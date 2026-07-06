"""Timezone helpers for generated reports and session timestamps."""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo


DEFAULT_TIMEZONE = "Asia/Kuala_Lumpur"


def app_timezone() -> ZoneInfo:
    timezone_name = os.environ.get("APP_TIMEZONE", DEFAULT_TIMEZONE)
    try:
        return ZoneInfo(timezone_name)
    except Exception:
        return ZoneInfo(DEFAULT_TIMEZONE)


def now_for_app() -> datetime:
    return datetime.now(app_timezone())


def timezone_offset_label(value: datetime) -> str:
    offset = value.strftime("%z")
    if not offset:
        return "GMT"
    return f"GMT{offset[:3]}:{offset[3:]}"


def formatted_now() -> str:
    timezone_name = os.environ.get("APP_TIMEZONE", DEFAULT_TIMEZONE)
    current = now_for_app()
    return f"{current.strftime('%Y-%m-%d %H:%M:%S')} ({timezone_name}, {timezone_offset_label(current)})"
