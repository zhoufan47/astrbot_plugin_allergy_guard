"""过敏守护插件的图片处理工具。

负责从消息事件中提取图片地址，并把图片（网络 URL / base64 / 本地文件）
保存到插件数据目录，供后续查看与追溯。
"""

from __future__ import annotations

import base64
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import aiohttp

from astrbot.api import logger

# 允许保存的图片扩展名
_VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def extract_image_urls(event) -> list:
    """从消息事件中提取图片地址列表。

    优先使用 AstrBot 提供的 ``event.image_urls`` 属性；若不可用或为空，
    则回退到遍历消息链中的 Image 组件，兼容不同版本与平台。
    """
    urls = []
    try:
        raw = getattr(event, "image_urls", None)
        if raw:
            urls = [u for u in list(raw) if u]
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[过敏守护] 读取 event.image_urls 失败: {exc}")
        urls = []

    if urls:
        return urls

    # 回退方案：遍历消息链
    try:
        for comp in event.get_messages():
            comp_type = getattr(comp, "type", None)
            class_name = comp.__class__.__name__
            if comp_type == "image" or class_name == "Image":
                for attr in ("url", "path", "file"):
                    value = getattr(comp, attr, None)
                    if value:
                        urls.append(value)
                        break
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[过敏守护] 遍历消息链提取图片失败: {exc}")

    return urls


def _guess_ext_from_content_type(content_type: str) -> str:
    content_type = (content_type or "").lower()
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    if "bmp" in content_type:
        return ".bmp"
    return ".jpg"


def _safe_user_dir_name(user_key: str) -> str:
    """把 user_key 转换为安全的目录名。"""
    safe = re.sub(r"[^0-9a-zA-Z_\-]", "_", user_key or "")[:64]
    return safe or "user"


async def save_image(url_or_path: str, dest_dir: Path, user_key: str) -> str | None:
    """把图片保存到本地目录，返回保存后的绝对路径；失败时返回 None。

    支持三种来源：
    - ``data:image/...;base64,`` 或 ``base64://`` 开头的 base64 数据；
    - ``http(s)://`` 网络图片；
    - 本地文件路径或 ``file://`` 路径。
    """
    if not url_or_path:
        return None

    try:
        user_dir = Path(dest_dir) / _safe_user_dir_name(user_key)
        user_dir.mkdir(parents=True, exist_ok=True)

        data = None
        ext = ".jpg"
        source = url_or_path.strip()

        if source.startswith("data:image"):
            header, _, b64_data = source.partition(",")
            ext = _guess_ext_from_content_type(header)
            data = base64.b64decode(b64_data)
        elif source.startswith("base64://"):
            data = base64.b64decode(source[len("base64://"):])
        elif source.startswith(("http://", "https://")):
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(source) as resp:
                    if resp.status == 200:
                        data = await resp.content.read()
                        ext = _guess_ext_from_content_type(
                            resp.headers.get("Content-Type", "")
                        )
                    else:
                        logger.warning(
                            f"[过敏守护] 下载图片失败，HTTP {resp.status}: {source[:80]}"
                        )
        else:
            local_path = source[len("file://"):] if source.startswith("file://") else source
            if os.path.exists(local_path):
                _, local_ext = os.path.splitext(local_path)
                if local_ext.lower() in _VALID_EXTS:
                    ext = ".jpg" if local_ext.lower() == ".jpeg" else local_ext.lower()
                with open(local_path, "rb") as fp:
                    data = fp.read()
            else:
                logger.warning(f"[过敏守护] 本地图片不存在: {local_path[:80]}")

        if not data:
            return None

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{ext}"
        out_path = user_dir / filename
        with open(out_path, "wb") as fp:
            fp.write(data)
        return str(out_path)
    except Exception as exc:  # noqa: BLE001 - 保存图片失败不应中断记录流程
        logger.warning(f"[过敏守护] 保存图片时出错: {exc}")
        return None
