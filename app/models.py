from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServerTarget:
    username: str
    hostname: str
    port: int = 22


@dataclass
class CloneRequest:
    connection_command: str
    project_path: str
    password: str


@dataclass
class OperationResult:
    success: bool
    message: str
    details: str = ""
