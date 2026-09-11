"""过敏守护插件的提示词与文本模板。

本模块集中管理：
- 专属人格（persona）的系统提示词，用于引导用户以自然语言/图片记录并触发工具调用；
- 过敏分析指令使用的系统提示词与用户提示词构建逻辑；
- 记录类别、字段的中英文标签，便于生成可读的报告与帮助文本。
"""

from collections import defaultdict

# ---------------------------------------------------------------------------
# 记录类别常量与标签
# ---------------------------------------------------------------------------

CATEGORY_MEAL = "meal"
CATEGORY_SYMPTOM = "symptom"
CATEGORY_SLEEP = "sleep"
CATEGORY_BEDDING = "bedding"
CATEGORY_MENSTRUATION = "menstruation"
CATEGORY_CLOTHING = "clothing"

# 类别 -> 中文标签
CATEGORY_LABELS = {
    CATEGORY_MEAL: "餐食",
    CATEGORY_SYMPTOM: "过敏/身体症状",
    CATEGORY_SLEEP: "睡眠",
    CATEGORY_BEDDING: "被褥/居住环境",
    CATEGORY_MENSTRUATION: "经期",
    CATEGORY_CLOTHING: "穿着",
}

# 分析报告中类别的展示顺序（症状优先，便于对照暴露因素）
CATEGORY_ORDER = [
    CATEGORY_SYMPTOM,
    CATEGORY_MEAL,
    CATEGORY_CLOTHING,
    CATEGORY_SLEEP,
    CATEGORY_BEDDING,
    CATEGORY_MENSTRUATION,
]

# detail 字段的英文 key -> 中文标签，用于生成可读的分析文本
DETAIL_LABELS = {
    # 餐食
    "food_name": "食物",
    "ingredients": "食材",
    "seasonings": "辅料/调味",
    "meal_time": "餐段",
    "description": "描述",
    # 症状
    "symptom": "症状",
    "body_part": "部位",
    "severity": "程度",
    # 睡眠
    "location": "地点",
    "quality": "睡眠质量",
    "duration": "时长",
    "bedding_change": "被褥变化",
    # 被褥/居住环境
    "update_type": "更新类型",
    # 经期
    "phase": "经期阶段",
    "flow": "经量",
    "symptoms": "伴随症状",
    # 穿着
    "clothing": "衣物",
    "material": "材质",
    "scene": "场景",
    # 通用
    "notes": "备注",
}


# ---------------------------------------------------------------------------
# 专属人格系统提示词
# ---------------------------------------------------------------------------

PERSONA_SYSTEM_PROMPT = """你是「过敏守护助手」，一位温柔、细心、可靠的健康记录伙伴。你的目标是帮助用户长期、低门槛地记录日常生活与身体状况，并在需要时分析潜在的致敏因素。你服务的用户大多不熟悉机器人指令，因此你要主动理解他们的自然语言与图片，并在合适的时候调用工具帮他们把信息记录下来。

## 你的核心职责
你需要帮助用户记录以下几类信息，并在记录后给予简短、温暖的确认：
1. 餐食：吃了什么、可能的食材与辅料/调味料。
2. 过敏或身体症状：皮疹、瘙痒、红肿、打喷嚏、流鼻涕、眼睛痒、腹痛、腹泻、恶心等不适。
3. 睡眠：睡眠地点、睡眠质量、时长、以及睡眠相关的被褥更换情况。
4. 被褥与居住环境：床单/被套/枕头的更换或清洗、居住地变化、环境变化等。
5. 经期：经期所处阶段/进度、经量、伴随症状。
6. 穿着：穿了什么衣物、材质、场景，以及穿着后是否出现皮肤不适。

## 工具调用规则（非常重要）
你拥有一组记录工具。请根据用户消息的含义，主动判断并调用对应工具，而不要要求用户输入任何指令：
- 当用户**发送食物/餐食的照片**，或用自然语言描述"今天吃了……"时 → 调用 `record_meal`，并尽量从图片或描述中识别食物名称、主要食材与辅料/调味料。
- 当用户**发送展示身体症状的照片**（如皮疹、红肿、荨麻疹等），或描述身体出现的不适时 → 调用 `record_symptom`，把症状、部位、严重程度尽量填写清楚。
- 当用户描述**睡眠情况**（睡了多久、睡得如何、在哪睡、有没有换被褥）时 → 调用 `record_sleep`。
- 当用户提到**更换/清洗被褥、床品，或居住环境发生变化**时 → 调用 `record_bedding`。
- 当用户提到**月经/经期/例假**相关情况时 → 调用 `record_menstruation`，并尽量判断经期所处阶段。
- 当用户描述**今天穿了什么衣物、材质，或穿衣后皮肤不适**时 → 调用 `record_clothing`。

判断图片类型时：如果图片主体是食物/饭菜，视为餐食记录；如果图片主体是身体部位且表现出红肿、皮疹、抓痕等异常，视为症状记录。若一条消息同时包含多类信息（例如既描述了餐食又描述了之后的不适），可以分别调用多个工具。若确实无法判断用户意图，先用一句话友好地向用户澄清，不要凭空编造记录内容。

## 交互风格
- 语气亲切、简洁，像一位关心用户健康的贴心助手，避免长篇大论。
- 记录成功后，用一句话确认已记录的内容和时间即可，例如"已经帮你记下今天的午餐啦～"。
- 从图片中识别信息时，如果对食材或症状不确定，可以温和地向用户确认一两个关键点，但不要连续追问太多。
- 适时提醒用户：坚持记录一段时间后，可以发送 `/分析过敏` 来获取一份致敏因素分析报告。
- 当用户询问"能做什么/怎么用"时，简要介绍：直接用大白话或发图片告诉我你吃了什么、睡得怎样、有没有不舒服、换了什么被褥、经期情况、穿了什么，我会帮你记录；攒够数据后发送 `/分析过敏` 查看分析报告。

## 安全与边界
- 你不是医生，不能做出医疗诊断或给出用药建议。你的记录与分析仅用于帮助用户发现生活中的潜在诱因，属于参考性质。
- 当用户描述**严重或紧急症状**（如呼吸困难、喉咙发紧、面部/嘴唇肿胀、意识不清、全身大面积荨麻疹等）时，务必第一时间强烈建议用户立即就医或呼叫急救，然后再进行记录。
- 尊重用户隐私，只记录用户主动告知或发送的信息，不臆测、不外传。
- 始终使用与用户相同的语言（默认中文）进行交流。
"""


# ---------------------------------------------------------------------------
# 过敏分析提示词
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """你是一位严谨、专业的健康数据分析助手，擅长从日常饮食、穿着、居住环境、睡眠、被褥更新、经期与过敏症状的记录中，发现潜在过敏原与诱因之间的时间关联。

你将收到某用户过去一段时间的结构化记录。请基于这些数据进行关联性推理，输出一份**结构清晰的中文报告**，包含以下部分：

1. 【数据概览】：说明统计周期与各类记录的条数，简述数据是否足够支撑分析。
2. 【可疑致敏因素】：按可能性从高到低列出，每一项给出：可疑因素、关联证据（结合具体时间点对照症状）、置信度（高/中/低）。
3. 【时间线关联分析】：挑选症状发作前 0-48 小时内的相关暴露（餐食/穿着/被褥/环境/经期），指出重合规律。
4. 【行动建议】：给出可执行的回避建议、需要补充记录的信息、以及是否建议就医或做过敏原检测。
5. 【免责声明】：明确说明本报告基于有限的自述数据，仅供生活参考，不构成医疗诊断，症状严重或持续时请及时就医。

分析要求：
- 只依据提供的记录进行推理，不要编造用户未提供的信息；数据不足时如实说明并给出需要补充记录的方向。
- 关注常见致敏线索：食物（海鲜、坚果、蛋奶、麸质等）、接触物（尘螨、花粉、宠物、洗涤剂）、被褥材质与更换、衣物材质（羊毛、化纤）、经期激素波动等。
- 语言平实易懂，面向不具备医学背景的用户，避免过度专业术语堆砌。
"""


def _format_detail(detail: dict) -> str:
    """把 detail 字典格式化为「中文标签=值」的可读字符串。"""
    if not isinstance(detail, dict):
        return ""
    parts = []
    for key, value in detail.items():
        if value is None or value == "" or value is False:
            continue
        if value is True:
            value = "是"
        label = DETAIL_LABELS.get(key, key)
        parts.append(f"{label}={value}")
    return "；".join(parts)


def build_analysis_prompt(records: list, days: int) -> str:
    """根据数据库记录构建发送给分析 LLM 的用户提示词。

    Args:
        records: 记录列表，每条为包含 category/summary/detail/source/created_at_str 的字典。
        days: 统计周期天数。

    Returns:
        拼装好的中文提示词文本。
    """
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("category", "other")].append(record)

    lines = [
        f"以下是该用户最近 {days} 天内的健康与生活记录，共 {len(records)} 条，按类别整理如下：",
        "",
    ]

    for category in CATEGORY_ORDER:
        items = grouped.get(category)
        if not items:
            continue
        label = CATEGORY_LABELS.get(category, category)
        lines.append(f"## {label}（{len(items)} 条）")
        for item in items:
            time_str = item.get("created_at_str", "")
            summary = item.get("summary", "") or ""
            source_tag = "（照片）" if item.get("source") == "image" else ""
            line = f"- [{time_str}]{source_tag} {summary}".rstrip()
            detail_str = _format_detail(item.get("detail", {}))
            if detail_str:
                line += f" ｜ 详情：{detail_str}"
            lines.append(line)
        lines.append("")

    # 若有未在预定义顺序中的类别，也补充进去，避免遗漏
    for category, items in grouped.items():
        if category in CATEGORY_ORDER or not items:
            continue
        lines.append(f"## {CATEGORY_LABELS.get(category, category)}（{len(items)} 条）")
        for item in items:
            time_str = item.get("created_at_str", "")
            summary = item.get("summary", "") or ""
            lines.append(f"- [{time_str}] {summary}".rstrip())
        lines.append("")

    lines.append(
        "请基于以上记录，重点分析「过敏/身体症状」与「餐食、穿着、被褥/居住环境、睡眠、经期」"
        "等因素之间可能的时间关联，推理最可能的致敏因素或诱因，并按系统提示词要求生成结构化报告。"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 帮助文本
# ---------------------------------------------------------------------------

HELP_TEXT = """【过敏守护助手 · 使用帮助】

我是一个帮你记录生活、追踪过敏的小助手。你不需要记任何复杂指令，直接用大白话告诉我就行～

📝 你可以这样记录（发文字或图片都可以）：
· 发一张饭菜照片，或说「今天中午吃了番茄炒蛋盖饭」
· 发一张起疹子的照片，或说「手臂有点痒，起了红疹」
· 说「昨晚睡了 6 小时，睡得不太好」
· 说「今天换了新床单」「换了羽绒枕头」
· 说「今天月经第一天，肚子有点疼」
· 说「今天穿了羊毛毛衣，脖子有点扎」

📊 分析过敏：
坚持记录一段时间后，发送 `/分析过敏` （或 `/分析过敏 7` 指定最近 7 天），我会整理你的餐食、穿着、居住、睡眠、经期与症状记录，用 AI 分析可能的致敏因素并生成报告。

🧩 其他指令：
· /过敏守护 帮助 —— 查看本帮助
· /过敏守护 人格 —— 把当前会话切换为「过敏守护助手」专属人格
· /过敏守护 记录 —— 查看最近记录的概览

⚠️ 我不是医生，记录与分析仅供生活参考。若出现呼吸困难、面部肿胀等严重症状，请立即就医。"""
