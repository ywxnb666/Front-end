from __future__ import annotations

from app.models import CloneRequest, OperationResult
from app.services.ssh_client import SSHClientManager, parse_connection_command


REPO_URL = "https://gitee.com/yang-wenxiao-111/CISCN-MICAD.git"
REPO_DIR_NAME = "CISCN-MICAD"


class RemoteGitCloneService:
    def clone_repository(self, request: CloneRequest) -> OperationResult:
        if not request.project_path.strip():
            return OperationResult(success=False, message="项目地址不能为空")
        if not request.password:
            return OperationResult(success=False, message="密码不能为空")

        try:
            target = parse_connection_command(request.connection_command)
        except ValueError as exc:
            return OperationResult(success=False, message=str(exc))

        project_path = request.project_path.strip()

        try:
            with SSHClientManager(target, request.password) as ssh:
                quoted_project_path = ssh.quote(project_path)
                quoted_repo_name = ssh.quote(REPO_DIR_NAME)
                quoted_repo_url = ssh.quote(REPO_URL)
                command = (
                    f"mkdir -p {quoted_project_path} && "
                    f"cd {quoted_project_path} && "
                    f"if [ -d {quoted_repo_name}/.git ]; then "
                    f"echo '__REPO_EXISTS__'; "
                    f"else git clone {quoted_repo_url}; "
                    f"fi"
                )
                result = ssh.run(command)
        except Exception as exc:
            return OperationResult(
                success=False,
                message="连接服务器或执行 clone 失败",
                details=str(exc),
            )

        if result.exit_code != 0:
            return OperationResult(
                success=False,
                message="远程命令执行失败",
                details=result.stderr.strip() or result.stdout.strip(),
            )

        if "__REPO_EXISTS__" in result.stdout:
            return OperationResult(
                success=True,
                message="远程仓库已存在，未重复 clone",
                details=f"{project_path}/{REPO_DIR_NAME}",
            )

        return OperationResult(
            success=True,
            message="仓库 clone 成功",
            details=f"{project_path}/{REPO_DIR_NAME}",
        )
