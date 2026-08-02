from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any
from pathlib import Path


APP_DIR = Path.home() / ".remote-clone-tool"
CONFIG_PATH = APP_DIR / "config.json"


@dataclass
class AppConfig:
    connection_command: str = ""
    project_path: str = ""
    console_vars: dict[str, Any] = field(default_factory=dict)
    form_vars: dict[str, str] = field(default_factory=dict)


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppConfig()
    return AppConfig(
        connection_command=str(
            payload.get("connection_command", payload.get("server_address", ""))
        ),
        project_path=str(payload.get("project_path", "")),
        console_vars=dict(payload.get("console_vars", {})) if isinstance(payload.get("console_vars", {}), dict) else {},
        form_vars={str(key): str(value) for key, value in payload.get("form_vars", {}).items()} if isinstance(payload.get("form_vars", {}), dict) else {},
    )


def save_config(config: AppConfig) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(asdict(config), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
