from __future__ import annotations

import shlex
from contextlib import AbstractContextManager
from dataclasses import dataclass

from app.models import ServerTarget


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


def parse_connection_command(raw: str) -> ServerTarget:
    value = raw.strip()
    if not value:
        raise ValueError("服务器连接命令不能为空")

    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise ValueError("服务器连接命令格式无效") from exc

    if not parts:
        raise ValueError("服务器连接命令不能为空")
    if parts[0] != "ssh":
        raise ValueError("当前仅支持以 ssh 开头的服务器连接命令")

    port = 22
    target_text = ""
    idx = 1
    while idx < len(parts):
        token = parts[idx]
        if token in {"-p", "-P"}:
            if idx + 1 >= len(parts):
                raise ValueError("ssh 命令中的端口参数不完整")
            try:
                port = int(parts[idx + 1])
            except ValueError as exc:
                raise ValueError("ssh 命令中的端口必须是整数") from exc
            idx += 2
            continue
        if token.startswith("-"):
            idx += 1
            continue
        target_text = token
        idx += 1

    if "@" not in target_text:
        raise ValueError("ssh 命令中需要包含 username@hostname")

    username, hostname = target_text.split("@", 1)
    username = username.strip()
    hostname = hostname.strip()
    if not username or not hostname:
        raise ValueError("ssh 命令中的 username@hostname 不完整")

    return ServerTarget(username=username, hostname=hostname, port=port)


class SSHClientManager(AbstractContextManager["SSHClientManager"]):
    def __init__(self, target: ServerTarget, password: str) -> None:
        self.target = target
        self.password = password
        self.client: paramiko.SSHClient | None = None

    def __enter__(self) -> "SSHClientManager":
        try:
            import paramiko
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 paramiko 依赖，无法建立 SSH 连接；请先安装 requirements.txt") from exc

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.target.hostname,
            port=self.target.port,
            username=self.target.username,
            password=self.password,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
        )
        self.client = client
        return self

    def run(self, command: str) -> CommandResult:
        if self.client is None:
            raise RuntimeError("SSH client is not connected")
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout.read().decode("utf-8", errors="replace"),
            stderr=stderr.read().decode("utf-8", errors="replace"),
        )

    def quote(self, value: str) -> str:
        return shlex.quote(value)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
