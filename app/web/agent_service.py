from __future__ import annotations

import json
import queue
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any

from app.config import APP_DIR
from app.web.agent_schemas import ActionProposal, AgentChatRequest, AgentChatResponse
from app.web.agent_tools import (
    AgentToolContext,
    apply_preset as tool_apply_preset,
    propose_pipeline_run as tool_propose_pipeline_run,
    propose_remote_config_save as tool_propose_remote_config_save,
    read_app_state as tool_read_app_state,
    read_pipeline_status as tool_read_pipeline_status,
    read_recent_logs as tool_read_recent_logs,
    read_results as tool_read_results,
    safe_app_state_snapshot,
    save_history as tool_save_history,
    set_allowed_parameters as tool_set_allowed_parameters,
    validate_remote_setup as tool_validate_remote_setup,
)

try:
    from pydantic_ai import RunContext
except ModuleNotFoundError:  # The web app can still start and report the missing optional dependency.
    RunContext = Any  # type: ignore[misc,assignment]


SESSION_DIR = APP_DIR / "assistant_sessions"
SESSION_TOKEN_LIMIT = 131072
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
ACTION_TTL_SECONDS = 600

AGENT_SYSTEM_PROMPT = """你是 MLLM 能力泄漏风险检测平台的轻量执行 Agent。

你只能处理本平台的参数、配置、Pipeline、日志、风险结果、水印和历史归档。遇到无关问题，简短说明能力范围，不调用工具。

规则：
1. 优先读取状态，再决定是否调用工具；不要猜测当前参数、文件或运行状态。
2. 参数修改只能使用 set_allowed_parameters 或 apply_preset，禁止请求 API Key、SSH 密码或任意 Shell。
3. 用户明确要求修改白名单参数时可以直接修改，并在最终回答列出改动。
4. Pipeline 和服务器配置同步只能提出待确认动作，绝不能声称已经执行。
5. 用户要求运行完整流程时使用 full=true；要求部分流程时传入明确的阶段 ID。
6. 日志只读取解决问题所需的最近部分，避免重复读取。
7. 回答使用简洁中文，区分建议、已修改、等待确认、正在运行和执行完成。

参数白名单别名：样本数量、MAX_NEW_TOKENS、NUM_WORKERS、MAX_CONCURRENCY、教师语言/思考开关、控制集、Stage1/Stage2 学习率、LoRA、损失权重、评估周期、judge 数据集与 split、并行控制，以及 train_num、max_samples、control_max_samples、eval_max_samples、judge_sample_num、stage1_epochs、stage1_batch_size、stage1_grad_accum、stage1_max_length、stage2_epochs、period_num、stage2_batch_size、stage2_grad_accum、stage2_max_length、eval_max_new_tokens、use_4bit、freeze_vision_tower、stage2_wrong_image_enable、stage2_pair_use_answer_correctness、dataset_name、cuda_devices、teacher_api_base、victim_model、judge_model、judge_api_base、judge_eval_api_base、watermark 参数。
教师标注数量通常同时修改 train_num 和 max_samples；思维链评估抽样使用 judge_sample_num。Stage2 batch size 会同时作用于 Phase A 和 Phase B。
可用参数预设：demo、recommended、low_memory、演示推荐参数（也接受 demo_recommended）。演示推荐参数是当前前端的小样本展示配置。

Pipeline 阶段 ID：teacher_collect、teacher_eval、origin_eval、stage1_train、stage2_train、student_eval、reason_judge、risk_report。
"""

ASK_SYSTEM_PROMPT = """你是 MLLM 能力泄漏风险检测平台的内置助手。只回答本平台的功能、参数、运行流程、报错、风险评估和水印检测问题。不要假装运行了工具或访问了服务器；无关问题简短说明能力范围。使用简洁准确的中文回答。"""


def _estimate_tokens(text: str) -> int:
    ascii_count = sum(1 for char in text if ord(char) < 128)
    return max(1, (ascii_count + 3) // 4 + len(text) - ascii_count)


def _tail_within_tokens(text: str, limit: int) -> str:
    if _estimate_tokens(text) <= limit:
        return text
    low, high = 0, len(text)
    while low < high:
        length = (low + high + 1) // 2
        if _estimate_tokens(text[-length:]) <= limit:
            low = length
        else:
            high = length - 1
    return text[-low:] if low else ""


def _safe_session_id(value: str) -> str:
    text = str(value or "").strip()
    return text if SESSION_ID_RE.fullmatch(text) else uuid.uuid4().hex


def _enabled(value: Any, default: bool = True) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


class AgentSessionStore:
    def __init__(self, directory: Path = SESSION_DIR, token_limit: int = SESSION_TOKEN_LIMIT) -> None:
        self.directory = directory
        self.token_limit = max(8192, token_limit)
        self.lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{_safe_session_id(session_id)}.json"

    def load(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        with self.lock:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"session_id": session_id, "summary": "", "messages": []}
        if not isinstance(payload, dict):
            return {"session_id": session_id, "summary": "", "messages": []}
        messages = payload.get("messages", [])
        payload["messages"] = messages if isinstance(messages, list) else []
        payload["summary"] = str(payload.get("summary", ""))
        payload["session_id"] = session_id
        return payload

    def save(self, session: dict[str, Any]) -> None:
        messages = list(session.get("messages", []))
        summary = str(session.get("summary", ""))
        summary_budget = max(1024, self.token_limit // 4)
        summary = _tail_within_tokens(summary, summary_budget)
        while messages and self._message_tokens(messages, summary) > self.token_limit:
            removed = messages.pop(0)
            snippet = str(removed.get("content", ""))[:600].replace("\n", " ")
            summary = _tail_within_tokens(
                (summary + f"\n{removed.get('role', 'unknown')}: {snippet}").strip(),
                summary_budget,
            )
        payload = {
            "session_id": session["session_id"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "messages": messages,
        }
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._path(str(session["session_id"])).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, session_id: str) -> None:
        with self.lock:
            try:
                self._path(session_id).unlink()
            except FileNotFoundError:
                pass

    def _message_tokens(self, messages: list[dict[str, Any]], summary: str) -> int:
        total = _estimate_tokens(summary)
        for message in messages:
            total += _estimate_tokens(str(message.get("content", "")))
            total += _estimate_tokens(str(message.get("reasoning", "")))
        return total

    def prompt_history(self, session: dict[str, Any], reserve_tokens: int = 12000) -> str:
        budget = max(4096, self.token_limit - reserve_tokens)
        selected: list[str] = []
        used = _estimate_tokens(str(session.get("summary", "")))
        for message in reversed(session.get("messages", [])[:-1]):
            line = f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            cost = _estimate_tokens(line)
            if used + cost > budget:
                break
            selected.append(line)
            used += cost
        selected.reverse()
        summary = str(session.get("summary", "")).strip()
        sections = []
        if summary:
            sections.append("较早会话摘要:\n" + summary)
        if selected:
            sections.append("最近会话:\n" + "\n".join(selected))
        return "\n\n".join(sections)


class AgentService:
    def __init__(self, console: Any) -> None:
        self.console = console
        try:
            context_limit = int(console.console_vars.get("assistant_context_limit", SESSION_TOKEN_LIMIT))
        except (TypeError, ValueError):
            context_limit = SESSION_TOKEN_LIMIT
        self.sessions = AgentSessionStore(token_limit=max(8192, context_limit))
        self.actions: dict[str, tuple[str, ActionProposal]] = {}
        self.action_lock = threading.Lock()
        self.session_locks: dict[str, threading.Lock] = {}
        self.session_locks_lock = threading.Lock()

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = AgentChatRequest.model_validate(payload)
        except Exception as exc:
            return AgentChatResponse(ok=False, mode="ask", message=f"请求格式无效：{exc}", error=str(exc)).model_dump()
        session_id = _safe_session_id(request.session_id)
        user_message = self._latest_user_message(request.messages)
        if not user_message:
            return AgentChatResponse(ok=False, mode=request.mode, session_id=session_id, error="请输入问题后再发送", message="请输入问题后再发送").model_dump()
        with self._session_lock(session_id):
            session, history = self._start_exchange(session_id, request.mode, user_message)
            try:
                response = self._run_pydantic_agent(
                    request.mode,
                    user_message,
                    history,
                    session_id,
                    history_view=request.history_view,
                )
            except Exception as exc:
                response = self._error_response(request.mode, session_id, exc)
            self._finish_exchange(session, request.mode, response)
        return response.model_dump()

    def chat_stream(self, payload: dict[str, Any]) -> Iterator[str]:
        """Run an agent in a worker and expose PydanticAI events as NDJSON."""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def emit(event: dict[str, Any]) -> None:
            events.put(event)

        def worker() -> None:
            try:
                request = AgentChatRequest.model_validate(payload)
                session_id = _safe_session_id(request.session_id)
                user_message = self._latest_user_message(request.messages)
                if not user_message:
                    raise ValueError("请输入问题后再发送")
                with self._session_lock(session_id):
                    session, history = self._start_exchange(session_id, request.mode, user_message)
                    emit({"type": "meta", "session_id": session_id, "mode": request.mode})
                    try:
                        response = self._run_pydantic_agent(
                            request.mode,
                            user_message,
                            history,
                            session_id,
                            event_sink=emit,
                            history_view=request.history_view,
                        )
                    except Exception as exc:
                        response = self._error_response(request.mode, session_id, exc)
                        self._finish_exchange(session, request.mode, response)
                        emit({"type": "error", "message": response.message})
                    else:
                        self._finish_exchange(session, request.mode, response)
                        emit({"type": "done", "response": response.model_dump()})
            except Exception as exc:
                emit({"type": "error", "message": f"AI 助手请求失败：{exc}"})
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True, name="assistant-stream").start()
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"

    def clear_session(self, session_id: str) -> dict[str, Any]:
        safe_id = _safe_session_id(session_id)
        with self._session_lock(safe_id):
            self.sessions.delete(safe_id)
        with self.action_lock:
            self.actions = {key: value for key, value in self.actions.items() if value[0] != safe_id}
        return {"ok": True, "session_id": safe_id}

    def get_session(self, session_id: str) -> dict[str, Any]:
        safe_id = _safe_session_id(session_id)
        with self._session_lock(safe_id):
            session = self.sessions.load(safe_id)
        with self.action_lock:
            self._purge_expired_actions_locked()
            active_actions = set(self.actions)
        messages = []
        for item in session.get("messages", []):
            if not isinstance(item, dict):
                continue
            pending_action = item.get("pending_action") if isinstance(item.get("pending_action"), dict) else None
            if pending_action and str(pending_action.get("action_id", "")) not in active_actions:
                pending_action = None
            messages.append({
                "role": str(item.get("role", "assistant")),
                "content": str(item.get("content", "")),
                "reasoning": str(item.get("reasoning", "")),
                "mode": str(item.get("mode", "ask")),
                "tool_events": item.get("tool_events", []) if isinstance(item.get("tool_events"), list) else [],
                "pending_action": pending_action,
            })
        return {"ok": True, "session_id": safe_id, "messages": messages}

    def create_action(self, session_id: str, action_type: str, title: str, detail: str, payload: dict[str, Any]) -> ActionProposal:
        action = ActionProposal(
            action_id=uuid.uuid4().hex,
            action_type=action_type,
            title=title,
            detail=detail,
            payload=payload,
            expires_at=time.time() + ACTION_TTL_SECONDS,
        )
        with self.action_lock:
            self._purge_expired_actions_locked()
            self.actions[action.action_id] = (_safe_session_id(session_id), action)
        return action

    def resolve_action(self, action_id: str, session_id: str, confirm: bool) -> dict[str, Any]:
        with self.action_lock:
            self._purge_expired_actions_locked()
            stored = self.actions.get(str(action_id))
            if stored is None:
                return {"ok": False, "message": "操作不存在、已执行或已过期"}
            owner, action = stored
            if owner != _safe_session_id(session_id):
                return {"ok": False, "message": "操作不属于当前会话"}
            self.actions.pop(str(action_id), None)
        if not confirm:
            return {"ok": True, "message": "已取消操作", "action_id": action.action_id}
        if action.action_type == "pipeline":
            steps = [str(step) for step in action.payload.get("steps", [])]
            result = self.console.start_tasks(steps)
            return {**result, "action_id": action.action_id, "pipeline": result.get("pipeline", self.console.pipeline_status())}
        if action.action_type == "remote_config":
            result = self.console.save_remote_config()
            return {**result, "action_id": action.action_id, "config": self.console.snapshot()}
        return {"ok": False, "message": "不支持的操作类型"}

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self.session_locks_lock:
            return self.session_locks.setdefault(session_id, threading.Lock())

    def _sync_session_token_limit(self) -> None:
        with self.console.lock:
            raw_limit = self.console.console_vars.get("assistant_context_limit", SESSION_TOKEN_LIMIT)
        try:
            self.sessions.token_limit = max(8192, int(raw_limit))
        except (TypeError, ValueError):
            self.sessions.token_limit = SESSION_TOKEN_LIMIT

    def _start_exchange(self, session_id: str, mode: str, user_message: str) -> tuple[dict[str, Any], str]:
        self._sync_session_token_limit()
        session = self.sessions.load(session_id)
        session["messages"].append({"role": "user", "content": user_message, "mode": mode})
        return session, self.sessions.prompt_history(session)

    def _finish_exchange(self, session: dict[str, Any], mode: str, response: AgentChatResponse) -> None:
        session["messages"].append({
            "role": "assistant",
            "content": response.message,
            "reasoning": response.reasoning,
            "mode": mode,
            "tool_events": [event.model_dump() for event in response.tool_events],
            "pending_action": response.pending_action.model_dump() if response.pending_action else None,
        })
        self.sessions.save(session)

    @staticmethod
    def _error_response(mode: str, session_id: str, exc: Exception) -> AgentChatResponse:
        message = f"AI 助手请求失败：{exc}"
        return AgentChatResponse(ok=False, mode=mode, session_id=session_id, message=message, error=str(exc))

    def _purge_expired_actions_locked(self) -> None:
        now = time.time()
        self.actions = {key: value for key, value in self.actions.items() if value[1].expires_at >= now}

    def _run_pydantic_agent(
        self,
        mode: str,
        user_message: str,
        history: str,
        session_id: str,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        history_view: bool = False,
    ) -> AgentChatResponse:
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models import openai as openai_models
            from pydantic_ai.providers.openai import OpenAIProvider
        except ModuleNotFoundError as exc:
            if mode == "ask":
                return self._run_legacy_ask(user_message, history, session_id, history_view)
            raise RuntimeError("缺少 pydantic-ai 依赖，请安装 requirements.txt") from exc

        base_url, api_key, model_name, reasoning_enabled = self._assistant_connection()

        model_class = getattr(openai_models, "OpenAIChatModel", None) or getattr(openai_models, "OpenAIModel", None)
        if model_class is None:
            raise RuntimeError("当前 pydantic-ai 版本缺少 OpenAI-compatible Chat 模型")
        model = model_class(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))
        system_prompt = AGENT_SYSTEM_PROMPT if mode == "agent" else ASK_SYSTEM_PROMPT
        agent = Agent(model, deps_type=AgentToolContext, system_prompt=system_prompt)
        context = AgentToolContext(
            console=self.console,
            create_action=lambda action_type, title, detail, data: self.create_action(session_id, action_type, title, detail, data),
            allow_mutations=not history_view,
        )
        if mode == "agent":
            self._register_tools(agent)
        prompt = self._build_prompt(user_message, history, history_view)
        settings: dict[str, Any] = {
            "temperature": 0.2,
            "max_tokens": 1600,
        }
        if reasoning_enabled:
            # PydanticAI forwards the standard setting where supported; the
            # extra body covers dpsk-compatible gateways using this switch.
            settings["openai_reasoning_effort"] = "high"
            settings["extra_body"] = {"reasoning": {"enabled": True}}
        async def stream_handler(_run_context: Any, stream: Any) -> None:
            async for event in stream:
                streamed = self._stream_event_payload(event)
                if streamed is not None and event_sink is not None:
                    event_sink(streamed)

        result = agent.run_sync(
            prompt,
            deps=context,
            model_settings=settings,
            event_stream_handler=stream_handler if event_sink is not None else None,
        )
        output = getattr(result, "output", getattr(result, "data", ""))
        message = str(output).strip()
        reasoning = self._extract_reasoning(result)
        message, reasoning = self._split_embedded_reasoning(message, reasoning)
        changed_config = any(event.name in {"set_allowed_parameters", "apply_preset"} and event.status == "completed" for event in context.events)
        return AgentChatResponse(
            ok=True,
            mode=mode,
            session_id=session_id,
            message=message or "模型没有返回文本内容",
            reasoning=reasoning,
            tool_events=context.events,
            pending_action=context.pending_action,
            config=self.console.snapshot() if changed_config else None,
        )

    def _run_legacy_ask(self, user_message: str, history: str, session_id: str, history_view: bool = False) -> AgentChatResponse:
        """Keep Ask mode usable while an environment is installing PydanticAI."""
        base_url, api_key, model, reasoning_enabled = self._assistant_connection()
        prompt = self._build_prompt(user_message, history, history_view)
        request_payload = {
            "model": model,
            "messages": [{"role": "system", "content": ASK_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1600,
        }
        if reasoning_enabled:
            request_payload["reasoning"] = {"enabled": True}
        request = urllib.request.Request(
            base_url + "/chat/completions",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"连接失败：{exc}") from exc
        try:
            raw = result["choices"][0]["message"]
            content = raw.get("content", "")
            if isinstance(content, list):
                content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            reasoning = str(raw.get("reasoning_content", raw.get("reasoning", "")) or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("AI 助手返回格式无法识别") from exc
        content, reasoning = self._split_embedded_reasoning(str(content), reasoning)
        return AgentChatResponse(ok=True, mode="ask", session_id=session_id, message=content or "模型没有返回文本内容", reasoning=reasoning)

    def _safe_context_text(self) -> str:
        return json.dumps(safe_app_state_snapshot(self.console), ensure_ascii=False)

    def _assistant_connection(self) -> tuple[str, str, str, bool]:
        with self.console.lock:
            base_url = str(self.console.console_vars.get("assistant_api_base", "")).strip().rstrip("/")
            api_key = str(self.console.console_vars.get("assistant_api_key", "")).strip()
            model = str(self.console.console_vars.get("assistant_model", "dpsk-v4-flash")).strip() or "dpsk-v4-flash"
            reasoning_enabled = _enabled(self.console.console_vars.get("assistant_reasoning", True))
        if not base_url:
            raise ValueError("尚未配置 AI 助手 Base URL")
        if not api_key:
            raise ValueError("尚未配置 AI 助手 API Key")
        for suffix in ("/chat/completions", "/responses"):
            if base_url.lower().endswith(suffix):
                base_url = base_url[: -len(suffix)].rstrip("/")
        return base_url, api_key, model, reasoning_enabled

    def _build_prompt(self, user_message: str, history: str, history_view: bool = False) -> str:
        context = self._safe_context_text()
        if history_view:
            context += "\n当前界面正在查看历史归档：只允许读取，不得修改参数、归档、同步配置或运行 Pipeline。"
        sections = [f"当前平台配置摘要：\n{context}"]
        if history:
            sections.append(f"以下是当前会话上下文，仅用于保持连续性：\n{history}")
        sections.append(f"用户最新请求：\n{user_message}")
        return "\n\n".join(sections)

    def _register_tools(self, agent: Any) -> None:
        @agent.tool
        def read_app_state(ctx: RunContext[AgentToolContext]) -> dict[str, Any]:
            """Read the current application configuration without secrets."""
            return tool_read_app_state(ctx.deps)

        @agent.tool
        def read_pipeline_status(ctx: RunContext[AgentToolContext]) -> dict[str, Any]:
            """Read current Pipeline stages and running state."""
            return tool_read_pipeline_status(ctx.deps)

        @agent.tool
        def read_recent_logs(ctx: RunContext[AgentToolContext], limit: int = 80) -> dict[str, Any]:
            """Read recent Pipeline logs, at most 200 lines."""
            return tool_read_recent_logs(ctx.deps, limit)

        @agent.tool
        def read_results(ctx: RunContext[AgentToolContext]) -> dict[str, Any]:
            """Read risk, watermark and history result summaries."""
            return tool_read_results(ctx.deps)

        @agent.tool
        def validate_remote_setup(ctx: RunContext[AgentToolContext]) -> dict[str, Any]:
            """Validate configured SSH and required model, dataset and checkpoint paths."""
            return tool_validate_remote_setup(ctx.deps)

        @agent.tool
        def set_allowed_parameters(ctx: RunContext[AgentToolContext], patches: dict[str, Any]) -> dict[str, Any]:
            """Set allowlisted parameters using the exact aliases listed in the system prompt."""
            return tool_set_allowed_parameters(ctx.deps, patches)

        @agent.tool
        def apply_preset(ctx: RunContext[AgentToolContext], preset: str) -> dict[str, Any]:
            """Apply one of demo, recommended or low_memory parameter presets."""
            return tool_apply_preset(ctx.deps, preset)

        @agent.tool
        def propose_pipeline_run(ctx: RunContext[AgentToolContext], steps: list[str] | None = None, full: bool = False) -> dict[str, Any]:
            """Propose a full or partial Pipeline using exact stage IDs. A non-empty run requires confirmation."""
            return tool_propose_pipeline_run(ctx.deps, steps, full)

        @agent.tool
        def save_remote_config(ctx: RunContext[AgentToolContext]) -> dict[str, Any]:
            """Propose saving the frontend configuration to the remote server. Requires confirmation."""
            return tool_propose_remote_config_save(ctx.deps)

        @agent.tool
        def save_history(ctx: RunContext[AgentToolContext], label: str = "") -> dict[str, Any]:
            """Archive the current result snapshot locally without rerunning experiments."""
            return tool_save_history(ctx.deps, label)

    def _latest_user_message(self, messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "user":
                content = str(item.get("content", "")).strip()
                if content:
                    return content[:16000]
        return ""

    def _extract_reasoning(self, result: Any) -> str:
        chunks: list[str] = []
        try:
            messages = result.all_messages()
        except Exception:
            messages = []
        for message in messages:
            for part in getattr(message, "parts", []):
                kind = str(getattr(part, "part_kind", part.__class__.__name__)).lower()
                if "think" not in kind and "reason" not in kind:
                    continue
                content = getattr(part, "content", getattr(part, "text", ""))
                if content:
                    chunks.append(str(content))
        return "\n".join(chunks).strip()

    def _stream_event_payload(self, event: Any) -> dict[str, Any] | None:
        kind = str(getattr(event, "event_kind", ""))
        if kind == "part_start":
            part = getattr(event, "part", None)
            part_kind = str(getattr(part, "part_kind", ""))
            content = str(getattr(part, "content", "") or "")
            if content and part_kind == "thinking":
                return {"type": "reasoning_delta", "delta": content}
            if content and part_kind == "text":
                return {"type": "answer_delta", "delta": content}
        if kind == "part_delta":
            delta = getattr(event, "delta", None)
            delta_kind = str(getattr(delta, "part_delta_kind", ""))
            content = str(getattr(delta, "content_delta", "") or "")
            if content and delta_kind == "thinking":
                return {"type": "reasoning_delta", "delta": content}
            if content and delta_kind == "text":
                return {"type": "answer_delta", "delta": content}
        if kind == "function_tool_call":
            part = getattr(event, "part", None)
            name = str(getattr(part, "tool_name", "tool"))
            return {"type": "tool", "event": {"name": name, "status": "started", "summary": "正在调用"}}
        if kind == "function_tool_result":
            part = getattr(event, "part", None) or getattr(event, "result", None)
            name = str(getattr(part, "tool_name", "tool"))
            outcome = str(getattr(part, "outcome", "success"))
            status = "completed" if outcome == "success" else "failed"
            return {"type": "tool", "event": {"name": name, "status": status, "summary": "调用完成" if status == "completed" else "调用失败"}}
        return None

    def _split_embedded_reasoning(self, message: str, reasoning: str = "") -> tuple[str, str]:
        embedded = re.findall(
            r"<think(?:ing)?[^>]*>(.*?)</think(?:ing)?>",
            str(message),
            flags=re.IGNORECASE | re.DOTALL,
        )
        clean_message = re.sub(
            r"<think(?:ing)?[^>]*>.*?</think(?:ing)?>",
            "",
            str(message),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        parts = [str(reasoning).strip(), *(part.strip() for part in embedded)]
        clean_reasoning = "\n\n".join(part for part in parts if part)
        return clean_message, clean_reasoning
