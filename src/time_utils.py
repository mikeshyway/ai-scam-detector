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


def formatted_now() -> str:
    timezone_name = os.environ.get("APP_TIMEZONE", DEFAULT_TIMEZONE)
    return f"{now_for_app().strftime('%Y-%m-%d %H:%M:%S')} ({timezone_name}, GMT+8)"
