"""过敏守护插件的数据存储层。

使用 Python 内置的 sqlite3 实现轻量级持久化，数据库文件保存在插件数据目录下，
避免插件更新/重装时数据丢失。所有对外的读写方法均为异步（内部通过线程池执行，
避免阻塞事件循环）。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

from astrbot.api import logger

# 建表语句：一条记录代表用户的一次餐食/症状/睡眠/被褥/经期/穿着记录
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS allergy_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    category TEXT NOT NULL,
    summary TEXT,
    detail TEXT,
    image_path TEXT,
    source TEXT,
    created_at INTEGER NOT NULL,
    created_at_str TEXT
)
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_allergy_user_time "
    "ON allergy_records(user_key, created_at)"
)


class AllergyStorage:
    """过敏守护记录存储管理器。"""

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        # sqlite3 连接非线程安全，使用锁串行化写操作
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # 内部同步方法
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化数据库表与索引。"""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(_CREATE_TABLE_SQL)
                    conn.execute(_CREATE_INDEX_SQL)
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:  # noqa: BLE001 - 存储初始化失败不应导致插件崩溃
            logger.error(f"[过敏守护] 初始化数据库失败: {exc}")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """把数据库行转换为字典，并解析 detail 中的 JSON。"""
        data = dict(row)
        detail_raw = data.get("detail")
        if detail_raw:
            try:
                data["detail"] = json.loads(detail_raw)
            except (json.JSONDecodeError, TypeError):
                data["detail"] = {}
        else:
            data["detail"] = {}
        return data

    def _add_record_sync(
        self,
        user_key: str,
        category: str,
        summary: str,
        detail: dict,
        image_path: str | None,
        source: str,
    ) -> dict:
        now = datetime.now()
        created_at = int(now.timestamp())
        created_at_str = now.strftime("%Y-%m-%d %H:%M")
        detail_json = json.dumps(detail or {}, ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO allergy_records "
                    "(user_key, category, summary, detail, image_path, source, "
                    "created_at, created_at_str) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_key,
                        category,
                        summary,
                        detail_json,
                        image_path,
                        source,
                        created_at,
                        created_at_str,
                    ),
                )
                conn.commit()
                record_id = cur.lastrowid
            finally:
                conn.close()
        return {
            "id": record_id,
            "user_key": user_key,
            "category": category,
            "summary": summary,
            "detail": detail or {},
            "image_path": image_path,
            "source": source,
            "created_at": created_at,
            "created_at_str": created_at_str,
        }

    def _get_recent_sync(self, user_key: str, days: int) -> list:
        since = int((datetime.now() - timedelta(days=days)).timestamp())
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, user_key, category, summary, detail, image_path, "
                    "source, created_at, created_at_str FROM allergy_records "
                    "WHERE user_key = ? AND created_at >= ? ORDER BY created_at ASC",
                    (user_key, since),
                ).fetchall()
            finally:
                conn.close()
        return [self._row_to_dict(row) for row in rows]

    def _get_stats_sync(self, user_key: str, days: int) -> dict:
        since = int((datetime.now() - timedelta(days=days)).timestamp())
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT category, COUNT(*) AS cnt FROM allergy_records "
                    "WHERE user_key = ? AND created_at >= ? GROUP BY category",
                    (user_key, since),
                ).fetchall()
                last = conn.execute(
                    "SELECT created_at_str, category, summary FROM allergy_records "
                    "WHERE user_key = ? ORDER BY created_at DESC LIMIT 1",
                    (user_key,),
                ).fetchone()
            finally:
                conn.close()
        counts = {row["category"]: row["cnt"] for row in rows}
        last_record = dict(last) if last else None
        return {"counts": counts, "total": sum(counts.values()), "last": last_record}

    # ------------------------------------------------------------------
    # 对外异步方法
    # ------------------------------------------------------------------
    async def add_record(
        self,
        user_key: str,
        category: str,
        summary: str,
        detail: dict | None = None,
        image_path: str | None = None,
        source: str = "text",
    ) -> dict:
        """新增一条记录，返回写入后的记录字典。"""
        return await asyncio.to_thread(
            self._add_record_sync,
            user_key,
            category,
            summary,
            detail or {},
            image_path,
            source,
        )

    async def get_recent_records(self, user_key: str, days: int) -> list:
        """获取指定用户最近 days 天内的全部记录（按时间升序）。"""
        return await asyncio.to_thread(self._get_recent_sync, user_key, days)

    async def get_stats(self, user_key: str, days: int) -> dict:
        """获取指定用户最近 days 天内的记录统计信息。"""
        return await asyncio.to_thread(self._get_stats_sync, user_key, days)
