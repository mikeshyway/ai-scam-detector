"""WebRTC ICE configuration helpers for local and hosted deployments."""

from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from typing import Iterable


DEFAULT_STUN_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
]


def _normalise_urls(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = [item.strip() for item in text.split(",")]
        return _normalise_urls(decoded)
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def normalise_ice_servers(servers: object) -> list[dict[str, object]]:
    """Normalise STUN/TURN entries returned by providers such as Twilio."""

    if not isinstance(servers, list):
        return []
    normalised: list[dict[str, object]] = []
    for server in servers:
        if not isinstance(server, dict):
            continue
        urls = _normalise_urls(server.get("urls") or server.get("url"))
        if not urls:
            continue
        item: dict[str, object] = {"urls": urls}
        username = str(server.get("username", "")).strip()
        credential = str(server.get("credential", "")).strip()
        if username:
            item["username"] = username
        if credential:
            item["credential"] = credential
        normalised.append(item)
    return normalised


def static_turn_servers(
    *,
    urls: object,
    username: str = "",
    credential: str = "",
) -> list[dict[str, object]]:
    """Build one TURN server entry from static deployment secrets."""

    turn_urls = [
        url
        for url in _normalise_urls(urls)
        if url.lower().startswith(("turn:", "turns:"))
    ]
    if not turn_urls or not username.strip() or not credential.strip():
        return []
    return [
        {
            "urls": turn_urls,
            "username": username.strip(),
            "credential": credential.strip(),
        }
    ]


def fetch_twilio_ice_servers(
    account_sid: str,
    auth_token: str,
    *,
    timeout: float = 10.0,
) -> list[dict[str, object]]:
    """Request short-lived STUN/TURN credentials from Twilio's Tokens API."""

    account_sid = account_sid.strip()
    auth_token = auth_token.strip()
    if not account_sid or not auth_token:
        return []

    safe_sid = urllib.parse.quote(account_sid, safe="")
    url = f"https://api.twilio.com/2010-04-01/Accounts/{safe_sid}/Tokens.json"
    encoded_auth = base64.b64encode(f"{account_sid}:{auth_token}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        url,
        data=b"",
        method="POST",
        headers={
            "Authorization": f"Basic {encoded_auth}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return normalise_ice_servers(payload.get("ice_servers"))


def build_rtc_configuration(
    *,
    turn_servers: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], str]:
    """Return a streamlit-webrtc RTC config and a human-readable mode."""

    supplied_servers = normalise_ice_servers(turn_servers or [])
    has_turn = any(
        str(url).lower().startswith(("turn:", "turns:"))
        for server in supplied_servers
        for url in _normalise_urls(server.get("urls"))
    )
    if has_turn:
        return {"iceServers": [*DEFAULT_STUN_SERVERS, *supplied_servers]}, "TURN relay configured"
    return {"iceServers": list(DEFAULT_STUN_SERVERS)}, "STUN only"


__all__ = [
    "build_rtc_configuration",
    "fetch_twilio_ice_servers",
    "normalise_ice_servers",
    "static_turn_servers",
]
