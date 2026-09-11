"""过敏守护（Allergy Guard）AstrBot 插件主入口。

功能概述：
- 通过 LLM 函数工具（function calling），让用户以自然语言或图片的形式，
  无门槛地记录餐食、过敏/身体症状、睡眠、被褥/居住环境、经期与穿着情况；
- 提供 `/分析过敏` 指令，整理最近若干天的记录并调用（可独立配置的）LLM
  分析潜在致敏因素，生成报告；
- 自动创建一个专属人格「过敏守护助手」，引导用户使用本插件并主动调用记录工具。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import (
    AstrMessageEvent,
    MessageChain,
    MessageEventResult,
    filter,
)
from astrbot.api.star import Context, Star, StarTools, register

from .image_utils import extract_image_urls, save_image
from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    CATEGORY_BEDDING,
    CATEGORY_CLOTHING,
    CATEGORY_LABELS,
    CATEGORY_MEAL,
    CATEGORY_MENSTRUATION,
    CATEGORY_SLEEP,
    CATEGORY_SYMPTOM,
    HELP_TEXT,
    PERSONA_SYSTEM_PROMPT,
    build_analysis_prompt,
)
from .storage import AllergyStorage

# 默认插件名（当无法从框架获取 self.name 时使用）
_DEFAULT_PLUGIN_NAME = "allergy_guard"
_DEFAULT_PERSONA_ID = "allergy_guard"


@register(
    "allergy_guard",
    "zhoufan47",
    "记录饮食/穿着/居住/睡眠/经期与过敏症状，并用 LLM 分析潜在致敏因素",
    "1.0.0",
    "https://github.com/zhoufan47/astrbot_plugin_allergy_guard",
)
class AllergyGuardPlugin(Star):
    """过敏守护插件。"""

    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config if config is not None else {}

        # 数据目录：优先使用框架提供的官方数据目录，回退到 data/plugin_data/
        plugin_name = getattr(self, "name", None) or _DEFAULT_PLUGIN_NAME
        self.data_dir = self._resolve_data_dir(plugin_name)
        self.image_dir = self.data_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # 初始化存储
        self.storage = AllergyStorage(self.data_dir / "allergy_guard.db")
        logger.info(f"[过敏守护] 插件已加载，数据目录：{self.data_dir}")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def initialize(self):
        """插件初始化：创建/更新专属人格。"""
        await self._ensure_persona()

    async def terminate(self):
        """插件卸载/停用时调用。"""
        logger.info("[过敏守护] 插件已停用。")

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_data_dir(plugin_name: str) -> Path:
        """获取插件数据目录，兼容不同 AstrBot 版本。"""
        try:
            data_dir = StarTools.get_data_dir(plugin_name)
            data_dir = Path(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[过敏守护] StarTools.get_data_dir 不可用（{exc}），使用回退路径。")
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path

            data_dir = Path(get_astrbot_data_path()) / "plugin_data" / plugin_name
        except Exception:  # noqa: BLE001
            data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @staticmethod
    async def _maybe_await(result):
        """兼容同步/异步两种 API：若为协程则等待，否则直接返回。"""
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _user_key(self, event: AstrMessageEvent) -> str:
        """构建用于区分不同用户数据的唯一键（平台 + 发送者 ID）。"""
        try:
            platform = event.get_platform_name() or "unknown"
        except Exception:  # noqa: BLE001
            platform = "unknown"
        try:
            sender = event.get_sender_id() or ""
        except Exception:  # noqa: BLE001
            sender = ""
        if not sender:
            sender = getattr(event, "unified_msg_origin", "") or "unknown"
        return f"{platform}:{sender}"

    async def _save_record(self, event, category, summary, detail, save_img=True):
        """保存一条记录（可选保存随附图片），返回 (record, image_path, has_image)。"""
        user_key = self._user_key(event)
        image_urls = extract_image_urls(event)
        has_image = bool(image_urls)
        image_path = None
        if save_img and has_image and self.config.get("save_images", True):
            image_path = await save_image(image_urls[0], self.image_dir, user_key)
        source = "image" if has_image else "text"
        record = await self.storage.add_record(
            user_key, category, summary, detail, image_path, source
        )
        return record, image_path, has_image

    async def _ensure_persona(self):
        """创建或更新专属人格。"""
        if not self.config.get("auto_create_persona", True):
            return
        persona_id = (self.config.get("persona_id") or _DEFAULT_PERSONA_ID).strip()
        manager = getattr(self.context, "persona_manager", None)
        if manager is None:
            logger.warning("[过敏守护] 当前 AstrBot 版本不支持 persona_manager，跳过人格创建。")
            return
        try:
            existing = None
            try:
                existing = await self._maybe_await(manager.get_persona(persona_id))
            except ValueError:
                existing = None
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[过敏守护] 查询人格失败：{exc}")
                existing = None

            if existing is None:
                await self._maybe_await(
                    manager.create_persona(
                        persona_id=persona_id,
                        system_prompt=PERSONA_SYSTEM_PROMPT,
                        begin_dialogs=[],
                        tools=None,  # None 表示允许使用该人格时调用全部工具
                    )
                )
                logger.info(f"[过敏守护] 已创建专属人格：{persona_id}")
            else:
                await self._maybe_await(
                    manager.update_persona(
                        persona_id=persona_id,
                        system_prompt=PERSONA_SYSTEM_PROMPT,
                    )
                )
                logger.info(f"[过敏守护] 已更新专属人格：{persona_id}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[过敏守护] 创建/更新人格失败：{exc}")

    async def _resolve_provider_id(self, event):
        """确定分析所使用的模型提供商 ID（优先使用插件独立配置）。"""
        configured = (self.config.get("analysis_provider_id") or "").strip()
        if configured:
            return configured
        umo = event.unified_msg_origin
        try:
            pid = await self._maybe_await(
                self.context.get_current_chat_provider_id(umo=umo)
            )
            if pid:
                return pid
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[过敏守护] get_current_chat_provider_id 失败：{exc}")
        # 回退：旧版 API
        try:
            prov = await self._maybe_await(self.context.get_using_provider(umo=umo))
            if prov is not None:
                pid = getattr(prov, "provider_id", None) or getattr(prov, "id", None)
                if pid:
                    return pid
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[过敏守护] get_using_provider 回退失败：{exc}")
        return None

    async def _call_llm(self, provider_id, prompt, system_prompt) -> str:
        """调用 LLM 生成文本，兼容新旧 API。"""
        if hasattr(self.context, "llm_generate"):
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=system_prompt,
            )
        else:
            prov = await self._maybe_await(
                self.context.get_provider_by_id(provider_id=provider_id)
            )
            if prov is None:
                raise RuntimeError(f"未找到模型提供商：{provider_id}")
            resp = await prov.text_chat(prompt=prompt, system_prompt=system_prompt)

        text = getattr(resp, "completion_text", "") or ""
        if not text:
            chain = getattr(resp, "result_chain", None)
            if chain:
                text = "".join(
                    getattr(seg, "text", "") for seg in chain if getattr(seg, "type", "") == "plain"
                )
        return text

    # ------------------------------------------------------------------
    # LLM 函数工具：业务流程 1 与 2（自然语言 / 图片触发）
    # ------------------------------------------------------------------
    @filter.llm_tool(name="record_meal")
    async def record_meal(
        self,
        event: AstrMessageEvent,
        food_name: str = "",
        ingredients: str = "",
        seasonings: str = "",
        meal_time: str = "",
        description: str = "",
    ) -> MessageEventResult:
        """记录用户的一餐饮食。当用户发送食物/餐食照片，或用自然语言描述吃了什么（如"今天中午吃了……"）时调用此工具。请尽量从图片或描述中识别食物名称、主要食材与辅料/调味料。

        Args:
            food_name(string): 食物或这一餐的名称，例如"番茄炒蛋盖饭"
            ingredients(string): 推测的主要食材，用逗号分隔，例如"番茄,鸡蛋,米饭"；无法判断时留空
            seasonings(string): 推测的辅料或调味料，用逗号分隔，例如"盐,食用油,葱花"；无法判断时留空
            meal_time(string): 用餐时段，例如"早餐""午餐""晚餐""加餐"；不确定时留空
            description(string): 对这一餐的简要补充描述；没有时留空
        """
        detail = {
            "food_name": food_name,
            "ingredients": ingredients,
            "seasonings": seasonings,
            "meal_time": meal_time,
            "description": description,
        }
        summary = food_name or "一餐饮食"
        if ingredients:
            summary += f"（食材：{ingredients}）"
        try:
            record, image_path, _ = await self._save_record(
                event, CATEGORY_MEAL, summary, detail
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[过敏守护] 记录餐食失败：{exc}")
            yield event.plain_result("记录餐食时出了点问题，请稍后再试。")
            return
        msg = f"已成功记录餐食（{record['created_at_str']}）：{summary}"
        if image_path:
            msg += "，并保存了餐食照片"
        yield event.plain_result(msg + "。")

    @filter.llm_tool(name="record_symptom")
    async def record_symptom(
        self,
        event: AstrMessageEvent,
        symptom: str = "",
        body_part: str = "",
        severity: str = "",
        notes: str = "",
    ) -> MessageEventResult:
        """记录用户的过敏或身体不适症状。当用户描述身体出现的不适（如皮疹、瘙痒、红肿、荨麻疹、打喷嚏、流鼻涕、眼睛痒、腹痛、腹泻、恶心等），或发送了展示身体症状的照片时调用此工具。

        Args:
            symptom(string): 症状的具体描述，例如"手臂出现红色风团并瘙痒"
            body_part(string): 症状出现的身体部位，例如"手臂""脸部""全身"；不确定时留空
            severity(string): 严重程度，例如"轻微""中等""严重"；不确定时留空
            notes(string): 其他补充说明，例如持续时间、是否用药；没有时留空
        """
        detail = {
            "symptom": symptom,
            "body_part": body_part,
            "severity": severity,
            "notes": notes,
        }
        summary = symptom or "身体不适"
        if body_part:
            summary = f"{body_part}：{summary}"
        if severity:
            summary += f"（程度：{severity}）"
        try:
            record, image_path, _ = await self._save_record(
                event, CATEGORY_SYMPTOM, summary, detail
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[过敏守护] 记录症状失败：{exc}")
            yield event.plain_result("记录症状时出了点问题，请稍后再试。")
            return
        msg = f"已成功记录症状（{record['created_at_str']}）：{summary}"
        if image_path:
            msg += "，并保存了症状照片"
        yield event.plain_result(msg + "。请提醒用户如症状严重应及时就医。")

    @filter.llm_tool(name="record_sleep")
    async def record_sleep(
        self,
        event: AstrMessageEvent,
        location: str = "",
        quality: str = "",
        duration: str = "",
        bedding_change: str = "",
        notes: str = "",
    ) -> MessageEventResult:
        """记录用户的睡眠情况。当用户描述昨晚或今天的睡眠（睡了多久、睡得如何、在哪里睡、睡眠时是否更换了被褥等）时调用此工具。

        Args:
            location(string): 睡眠地点，例如"家中卧室""宿舍""酒店"；不确定时留空
            quality(string): 睡眠质量，例如"良好""一般""差""多梦易醒"；不确定时留空
            duration(string): 睡眠时长，例如"7小时"；不确定时留空
            bedding_change(string): 本次睡眠涉及的被褥更换情况，例如"换了新床单""盖了羽绒被"；无更换时留空
            notes(string): 其他补充信息，例如睡前环境、是否开空调；没有时留空
        """
        detail = {
            "location": location,
            "quality": quality,
            "duration": duration,
            "bedding_change": bedding_change,
            "notes": notes,
        }
        parts = []
        if duration:
            parts.append(f"时长{duration}")
        if quality:
            parts.append(f"质量{quality}")
        if location:
            parts.append(f"地点{location}")
        if bedding_change:
            parts.append(f"被褥：{bedding_change}")
        summary = "睡眠记录：" + "，".join(parts) if parts else "睡眠记录"
        try:
            record, _, _ = await self._save_record(
                event, CATEGORY_SLEEP, summary, detail, save_img=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[过敏守护] 记录睡眠失败：{exc}")
            yield event.plain_result("记录睡眠时出了点问题，请稍后再试。")
            return
        yield event.plain_result(
            f"已成功记录睡眠（{record['created_at_str']}）：{summary}。"
        )

    @filter.llm_tool(name="record_bedding")
    async def record_bedding(
        self,
        event: AstrMessageEvent,
        update_type: str = "",
        description: str = "",
        location: str = "",
    ) -> MessageEventResult:
        """记录被褥或居住环境的更新变化。当用户提到更换/清洗床单被套枕头、更换居住地，或居住环境发生变化（如新装修、换了房间、铺了新地毯、新添置家具）时调用此工具。

        Args:
            update_type(string): 更新类型，例如"更换床单""清洗被套""更换枕头""更换居住地"
            description(string): 更新的详细描述，例如"把旧棉絮换成了新的蚕丝被"
            location(string): 相关地点或环境，例如"主卧""出租屋"；不确定时留空
        """
        detail = {
            "update_type": update_type,
            "description": description,
            "location": location,
        }
        summary = update_type or "被褥/居住环境更新"
        if description:
            summary += f"：{description}"
        try:
            record, _, _ = await self._save_record(
                event, CATEGORY_BEDDING, summary, detail, save_img=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[过敏守护] 记录被褥/环境失败：{exc}")
            yield event.plain_result("记录被褥/居住环境时出了点问题，请稍后再试。")
            return
        yield event.plain_result(
            f"已成功记录被褥/居住环境更新（{record['created_at_str']}）：{summary}。"
        )

    @filter.llm_tool(name="record_menstruation")
    async def record_menstruation(
        self,
        event: AstrMessageEvent,
        phase: str = "",
        flow: str = "",
        symptoms: str = "",
        notes: str = "",
    ) -> MessageEventResult:
        """记录用户的经期情况。当用户提到月经/经期/例假/大姨妈相关信息（如经期第几天、经量、伴随症状）时调用此工具，并尽量判断经期所处的阶段或进度。

        Args:
            phase(string): 经期阶段或进度，例如"月经第一天""经期第3天""经期结束""排卵期""经前"
            flow(string): 经量情况，例如"少量""中等""大量"；不确定时留空
            symptoms(string): 伴随症状，例如"腹痛""腰酸""情绪波动"；没有时留空
            notes(string): 其他补充说明；没有时留空
        """
        detail = {
            "phase": phase,
            "flow": flow,
            "symptoms": symptoms,
            "notes": notes,
        }
        summary = phase or "经期记录"
        extra = []
        if flow:
            extra.append(f"经量{flow}")
        if symptoms:
            extra.append(f"伴随{symptoms}")
        if extra:
            summary += "（" + "，".join(extra) + "）"
        try:
            record, _, _ = await self._save_record(
                event, CATEGORY_MENSTRUATION, summary, detail, save_img=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[过敏守护] 记录经期失败：{exc}")
            yield event.plain_result("记录经期时出了点问题，请稍后再试。")
            return
        yield event.plain_result(
            f"已成功记录经期情况（{record['created_at_str']}）：{summary}。"
        )

    @filter.llm_tool(name="record_clothing")
    async def record_clothing(
        self,
        event: AstrMessageEvent,
        clothing: str = "",
        material: str = "",
        scene: str = "",
        notes: str = "",
    ) -> MessageEventResult:
        """记录用户的穿着情况。当用户描述今天穿了什么衣物、衣物的材质，或穿衣后出现皮肤不适（如发扎、发痒、起疹）时调用此工具。

        Args:
            clothing(string): 穿着的衣物描述，例如"羊毛毛衣""化纤运动服""新买的牛仔裤"
            material(string): 衣物主要材质，例如"羊毛""聚酯纤维""纯棉"；不确定时留空
            scene(string): 穿着场景或地点，例如"户外""家中""办公室"；不确定时留空
            notes(string): 其他补充说明，例如穿着后是否皮肤不适；没有时留空
        """
        detail = {
            "clothing": clothing,
            "material": material,
            "scene": scene,
            "notes": notes,
        }
        summary = clothing or "穿着记录"
        if material:
            summary += f"（材质：{material}）"
        try:
            record, image_path, _ = await self._save_record(
                event, CATEGORY_CLOTHING, summary, detail
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[过敏守护] 记录穿着失败：{exc}")
            yield event.plain_result("记录穿着时出了点问题，请稍后再试。")
            return
        msg = f"已成功记录穿着（{record['created_at_str']}）：{summary}"
        if image_path:
            msg += "，并保存了照片"
        yield event.plain_result(msg + "。")

    # ------------------------------------------------------------------
    # 指令：业务流程 3（分析过敏）
    # ------------------------------------------------------------------
    @filter.command("分析过敏")
    async def analyze_allergy(self, event: AstrMessageEvent, days: int = None):
        """整理最近一段时间（默认 14 天）的记录，用 AI 分析可能的致敏因素并生成报告。

        可选参数 days 指定统计天数，例如 /分析过敏 7
        """
        user_key = self._user_key(event)
        try:
            days = int(days) if days else int(self.config.get("analysis_days", 14) or 14)
        except (TypeError, ValueError):
            days = 14
        if days <= 0:
            days = 14

        records = await self.storage.get_recent_records(user_key, days)
        if not records:
            yield event.plain_result(
                f"最近 {days} 天还没有任何记录哦～\n"
                "平时直接用大白话告诉我你吃了什么、睡得怎样、有没有不舒服、"
                "换了什么被褥、经期情况或穿了什么（也可以发图片），我会帮你记录。"
                "攒够数据后再发送 /分析过敏 就能拿到分析报告啦。"
            )
            return

        provider_id = await self._resolve_provider_id(event)
        if not provider_id:
            yield event.plain_result(
                "未找到可用的大模型提供商，请先在 AstrBot 后台配置模型，"
                "或在本插件配置中指定「过敏分析使用的模型提供商」。"
            )
            return

        # 发送进度提示（非关键步骤，失败也不影响后续分析）
        try:
            await event.send(
                MessageChain(
                    [Comp.Plain(f"正在整理最近 {days} 天的 {len(records)} 条记录并分析，请稍候…")]
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[过敏守护] 发送进度提示失败：{exc}")

        user_prompt = build_analysis_prompt(records, days)
        system_prompt = ANALYSIS_SYSTEM_PROMPT
        extra = (self.config.get("analysis_extra_requirements") or "").strip()
        if extra:
            system_prompt += f"\n\n用户的额外要求：\n{extra}"

        try:
            report = await self._call_llm(provider_id, user_prompt, system_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[过敏守护] 分析过敏失败：{exc}")
            yield event.plain_result(f"分析失败：{exc}")
            return

        if not report or not report.strip():
            report = "模型未返回有效内容，请稍后重试或更换分析模型。"
        yield event.plain_result(report)

    # ------------------------------------------------------------------
    # 辅助指令组
    # ------------------------------------------------------------------
    @filter.command_group("过敏守护")
    async def allergy_guard(self):
        """过敏守护插件的辅助指令组。"""

    @allergy_guard.command("帮助")
    async def guard_help(self, event: AstrMessageEvent):
        """查看过敏守护插件的使用帮助。"""
        yield event.plain_result(HELP_TEXT)

    @allergy_guard.command("人格")
    async def guard_persona(self, event: AstrMessageEvent):
        """把当前会话切换为「过敏守护助手」专属人格。"""
        persona_id = (self.config.get("persona_id") or _DEFAULT_PERSONA_ID).strip()
        umo = event.unified_msg_origin
        conv_mgr = getattr(self.context, "conversation_manager", None)
        if conv_mgr is None:
            yield event.plain_result(f"当前版本不支持会话人格切换，请使用 /persona {persona_id} 手动切换。")
            return
        try:
            await self._ensure_persona()
            cid = await self._maybe_await(conv_mgr.get_curr_conversation_id(umo))
            await self._maybe_await(
                conv_mgr.update_conversation(umo, cid, persona_id=persona_id)
            )
            yield event.plain_result(
                f"已把当前会话切换为「过敏守护助手」人格（{persona_id}）。\n"
                "现在可以直接用大白话或发图片告诉我你的饮食、睡眠、症状等情况，我会帮你记录～"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[过敏守护] 切换人格失败：{exc}")
            yield event.plain_result(
                f"自动切换人格失败：{exc}\n你可以手动发送 /persona {persona_id} 进行切换。"
            )

    @allergy_guard.command("记录")
    async def guard_records(self, event: AstrMessageEvent, days: int = None):
        """查看最近记录的概览统计。"""
        user_key = self._user_key(event)
        try:
            days = int(days) if days else int(self.config.get("analysis_days", 14) or 14)
        except (TypeError, ValueError):
            days = 14
        if days <= 0:
            days = 14
        stats = await self.storage.get_stats(user_key, days)
        if not stats["total"]:
            yield event.plain_result(
                f"最近 {days} 天还没有记录。直接用大白话或发图片告诉我你的情况即可开始记录～"
            )
            return
        lines = [f"【最近 {days} 天记录概览】共 {stats['total']} 条："]
        for category, count in stats["counts"].items():
            label = CATEGORY_LABELS.get(category, category)
            lines.append(f"· {label}：{count} 条")
        last = stats.get("last")
        if last:
            lines.append(
                f"最近一条：[{last.get('created_at_str', '')}] {last.get('summary', '')}"
            )
        lines.append("\n发送 /分析过敏 可获取致敏因素分析报告。")
        yield event.plain_result("\n".join(lines))
