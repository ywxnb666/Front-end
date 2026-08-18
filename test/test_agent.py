from __future__ import annotations

import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import AppConfig
from app.web.agent_service import AGENT_SYSTEM_PROMPT, ASK_SYSTEM_PROMPT, AgentService, AgentSessionStore
from app.web.agent_schemas import AgentChatResponse
from app.web.agent_tools import (
    AgentToolContext,
    apply_preset,
    propose_pipeline_run,
    read_app_state,
    read_recent_logs,
    set_allowed_parameters,
)
from app.web.server import WebConsole


class AgentTests(unittest.TestCase):
    def make_console(self, root: Path) -> WebConsole:
        with patch("app.web.server.load_config", return_value=AppConfig()), patch("app.web.server.save_config"):
            console = WebConsole(debug=True)
        console.save_app_config_locked = lambda: None  # type: ignore[method-assign]
        console.agent_service.sessions = AgentSessionStore(root / "sessions", token_limit=8192)
        return console

    def make_context(self, console: WebConsole) -> AgentToolContext:
        return AgentToolContext(
            console=console,
            create_action=lambda action_type, title, detail, payload: console.agent_service.create_action(
                "testsession", action_type, title, detail, payload
            ),
        )

    def test_product_knowledge_is_shared_by_ask_and_agent(self) -> None:
        for prompt in (ASK_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT):
            self.assertIn('左侧栏的“历史测评数据”区域', prompt)
            self.assertIn("~/.remote-clone-tool/history/", prompt)
            self.assertIn("不要建议用户在笔记本上直接打开服务器路径", prompt)

    def test_prompt_context_includes_local_history_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            console.history_list = lambda: [{
                "id": "archive-1",
                "label": "ScienceQA 完整评估",
                "created_at": "2026-08-18 12:00:00",
                "risk": "0.6386",
            }]
            context = json.loads(console.agent_service._safe_context_text())
            self.assertEqual(context["history_archive"]["count"], 1)
            self.assertEqual(context["history_archive"]["recent_items"][0]["id"], "archive-1")
            self.assertNotIn("assistant_api_key", context["console_vars"])

    def test_parameter_patch_is_allowlisted_and_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            context = self.make_context(console)
            result = set_allowed_parameters(context, {"train_num": 5, "use_4bit": True})
            self.assertEqual(result["count"], 2)
            self.assertEqual(console.form_vars["TRAIN_NUM"], "5")
            self.assertEqual(console.form_vars["USE_4BIT"], "1")

            with self.assertRaisesRegex(ValueError, "不允许修改参数"):
                set_allowed_parameters(context, {"teacher_api_key": "secret"})
            with self.assertRaisesRegex(ValueError, "必须在"):
                set_allowed_parameters(context, {"stage1_batch_size": 0})

    def test_presets_do_not_change_paths_or_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            console.console_vars["dataset_path"] = "/data/scienceqa"
            console.console_vars["teacher_api_key"] = "secret"
            result = apply_preset(self.make_context(console), "low_memory")
            self.assertEqual(result["preset"], "low_memory")
            self.assertEqual(console.console_vars["dataset_path"], "/data/scienceqa")
            self.assertEqual(console.console_vars["teacher_api_key"], "secret")
            self.assertEqual(console.form_vars["STAGE1_BATCH_SIZE"], "1")
            self.assertEqual(console.form_vars["USE_4BIT"], "1")

    def test_demo_recommended_preset_matches_current_frontend_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            preserved = {
                "teacher_api_base": "https://teacher.example/v1",
                "judge_api_base": "https://judge.example/v1",
                "assistant_api_base": "https://assistant.example/v1",
                "assistant_model": "custom-assistant",
                "assistant_context_limit": 65536,
                "cuda_devices": "2",
                "watermark_device": "cpu",
                "watermark_torch_dtype": "float32",
                "sim_duration": 27,
                "auto_refresh": False,
            }
            console.console_vars.update(preserved)
            console.form_vars["JUDGE_API_BASE"] = "https://judge-form.example/v1"
            result = apply_preset(self.make_context(console), "demo_recommended")
            self.assertEqual(result["preset"], "演示推荐参数")
            self.assertEqual(console.form_vars["TRAIN_NUM"], "1")
            self.assertEqual(console.form_vars["MAX_SAMPLES"], "1")
            self.assertEqual(console.form_vars["SCIENCEQA_CONTROL_MAX_SAMPLES"], "1")
            self.assertEqual(console.form_vars["EVAL_MAX_SAMPLES"], "1")
            self.assertEqual(console.console_vars["judge_sample_num"], "5")
            self.assertEqual(console.form_vars["STAGE1_MAX_LENGTH"], "1536")
            self.assertEqual(console.form_vars["STAGE2_MAX_LENGTH"], "1024")
            self.assertEqual(console.form_vars["USE_4BIT"], "0")
            self.assertEqual(console.form_vars["STAGE2_WRONG_IMAGE_ENABLE"], "0")
            self.assertEqual(console.form_vars["MAX_NEW_TOKENS"], "64")
            self.assertEqual(console.form_vars["NUM_WORKERS"], "1")
            self.assertEqual(console.form_vars["MAX_CONCURRENCY"], "4")
            self.assertEqual(console.form_vars["STAGE1_LR"], "3e-5")
            self.assertEqual(console.form_vars["STAGE2_LR"], "2e-5")
            self.assertEqual(console.form_vars["LORA_RANK"], "16")
            self.assertEqual(console.form_vars["STAGE2_EVAL_MAX_SAMPLES"], "1")
            self.assertEqual(console.form_vars["PARALLEL_CONTROLS"], "1")
            self.assertEqual(console.form_vars["SCIENCEQA_SEED"], "20240306")
            self.assertEqual(console.form_vars["JUDGE_API_BASE"], "https://judge-form.example/v1")
            for key, value in preserved.items():
                self.assertEqual(console.console_vars[key], value)

    def test_stage2_batch_size_updates_both_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            result = set_allowed_parameters(self.make_context(console), {"stage2_batch_size": 2})
            self.assertEqual(result["count"], 1)
            self.assertEqual(console.form_vars["PHASE_A_BATCH_SIZE"], "2")
            self.assertEqual(console.form_vars["PHASE_B_BATCH_SIZE"], "2")

    def test_state_tool_never_returns_api_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            console.console_vars["assistant_api_key"] = "assistant-secret"
            console.console_vars["teacher_api_key"] = "teacher-secret"
            console.form_vars["JUDGE_API_KEY"] = "judge-secret"
            result = read_app_state(self.make_context(console))
            serialized = str(result)
            self.assertNotIn("assistant-secret", serialized)
            self.assertNotIn("teacher-secret", serialized)
            self.assertNotIn("judge-secret", serialized)

    def test_pipeline_action_requires_confirmation_and_cannot_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            context = self.make_context(console)
            result = propose_pipeline_run(context, ["stage1_train", "stage2_train"])
            self.assertTrue(result["requires_confirmation"])
            self.assertIsNotNone(context.pending_action)
            self.assertIsNone(console.real_pipeline_rows)

            with patch.object(console, "start_tasks", return_value={"ok": True, "pipeline": {"running": True}}) as start:
                confirmed = console.agent_service.resolve_action(result["pending_action_id"], "testsession", True)
                replayed = console.agent_service.resolve_action(result["pending_action_id"], "testsession", True)
            start.assert_called_once_with(["stage1_train", "stage2_train"])
            self.assertTrue(confirmed["ok"])
            self.assertFalse(replayed["ok"])

    def test_pipeline_proposal_requires_explicit_scope_and_deduplicates_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            context = self.make_context(console)
            with self.assertRaisesRegex(ValueError, "明确指定"):
                propose_pipeline_run(context)

            result = propose_pipeline_run(context, ["stage1_train", "stage1_train", "stage2_train"])
            self.assertEqual([item["id"] for item in result["plan"]], ["stage1_train", "stage2_train"])

    def test_pipeline_proposal_rejects_running_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            with patch.object(console, "pipeline_status", return_value={"running": True}):
                with self.assertRaisesRegex(ValueError, "正在运行"):
                    propose_pipeline_run(self.make_context(console), ["stage1_train"])

    def test_history_view_context_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            context = self.make_context(console)
            context.allow_mutations = False
            original_train_num = console.form_vars["TRAIN_NUM"]
            with self.assertRaisesRegex(ValueError, "历史归档"):
                set_allowed_parameters(context, {"train_num": 9})
            with self.assertRaisesRegex(ValueError, "历史归档"):
                propose_pipeline_run(context, ["stage1_train"])
            self.assertEqual(console.form_vars["TRAIN_NUM"], original_train_num)

    def test_fully_reused_pipeline_does_not_create_empty_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            console.console_vars.update({
                "reuse_teacher_annotation": "1",
                "teacher_annotation_path": "/tmp/teacher.json",
                "reuse_teacher_baseline": "1",
                "teacher_baseline_path": "/tmp/baseline.json",
            })
            with patch.object(console, "remote_or_local_file_exists", return_value=True):
                context = self.make_context(console)
                result = propose_pipeline_run(context, ["teacher_collect", "teacher_eval"])
            self.assertFalse(result["requires_confirmation"])
            self.assertIsNone(context.pending_action)
            self.assertEqual(len(result["skipped"]), 2)

    def test_expired_or_foreign_action_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            action = console.agent_service.create_action("owner1234", "pipeline", "run", "detail", {"steps": []})
            self.assertFalse(console.agent_service.resolve_action(action.action_id, "other1234", True)["ok"])

            expired = console.agent_service.create_action("owner1234", "pipeline", "run", "detail", {"steps": []})
            expired.expires_at = time.time() - 1
            self.assertFalse(console.agent_service.resolve_action(expired.action_id, "owner1234", True)["ok"])

    def test_session_is_persisted_and_trimmed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentSessionStore(Path(temp_dir), token_limit=8192)
            session = {
                "session_id": "session1234",
                "summary": "",
                "messages": [{"role": "user", "content": "测" * 9000}, {"role": "assistant", "content": "保留最后消息"}],
            }
            store.save(session)
            loaded = store.load("session1234")
            self.assertTrue(loaded["summary"])
            self.assertEqual(loaded["messages"][-1]["content"], "保留最后消息")
            self.assertLessEqual(store._message_tokens(loaded["messages"], loaded["summary"]), store.token_limit)

    def test_session_response_does_not_restore_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            service: AgentService = console.agent_service
            service.sessions.save({
                "session_id": "session1234",
                "summary": "",
                "messages": [{"role": "assistant", "content": "hello", "reasoning": "private reasoning", "tool_events": []}],
            })
            restored = service.get_session("session1234")
            self.assertTrue(restored["ok"])
            self.assertEqual(restored["messages"][0]["reasoning"], "private reasoning")

    def test_embedded_reasoning_is_removed_from_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            message, reasoning = console.agent_service._split_embedded_reasoning(
                "<think>内部推理</think>最终答案",
                "供应商推理",
            )
            self.assertEqual(message, "最终答案")
            self.assertEqual(reasoning, "供应商推理\n\n内部推理")

    def test_registered_tool_names_match_agent_contract(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.names: list[str] = []

            def tool(self, function):
                self.names.append(function.__name__)
                return function

        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            agent = FakeAgent()
            console.agent_service._register_tools(agent)
            self.assertEqual(agent.names, [
                "read_app_state",
                "read_pipeline_status",
                "read_recent_logs",
                "read_results",
                "validate_remote_setup",
                "set_allowed_parameters",
                "apply_preset",
                "propose_pipeline_run",
                "save_remote_config",
                "save_history",
            ])

    def test_recent_logs_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            console.real_pipeline_logs = ["x" * 5000 for _ in range(250)]
            result = read_recent_logs(self.make_context(console), 500)
            self.assertLessEqual(len(result["logs"]), 200)
            self.assertLessEqual(sum(len(line) for line in result["logs"]), 30000)
            self.assertTrue(all(len(line) <= 2000 for line in result["logs"]))

    def test_pydantic_adapter_contract_with_tool_calling(self) -> None:
        class FakeResult:
            output = "<think>内部思考</think>参数已修改"

            @staticmethod
            def all_messages():
                return []

        class FakeAgent:
            last = None

            def __init__(self, model, deps_type, system_prompt) -> None:
                self.model = model
                self.tools = {}
                self.settings = None
                FakeAgent.last = self

            def tool(self, function):
                self.tools[function.__name__] = function
                return function

            def run_sync(self, prompt, deps, model_settings, event_stream_handler=None):
                self.settings = model_settings
                context = types.SimpleNamespace(deps=deps)
                if "set_allowed_parameters" in self.tools:
                    self.tools["set_allowed_parameters"](context, {"train_num": 7})
                return FakeResult()

        class FakeProvider:
            def __init__(self, base_url, api_key) -> None:
                self.base_url = base_url
                self.api_key = api_key

        class FakeModel:
            def __init__(self, name, provider) -> None:
                self.name = name
                self.provider = provider

        pydantic_ai = types.ModuleType("pydantic_ai")
        pydantic_ai.Agent = FakeAgent
        models = types.ModuleType("pydantic_ai.models")
        openai_models = types.ModuleType("pydantic_ai.models.openai")
        openai_models.OpenAIChatModel = FakeModel
        models.openai = openai_models
        providers = types.ModuleType("pydantic_ai.providers")
        openai_provider = types.ModuleType("pydantic_ai.providers.openai")
        openai_provider.OpenAIProvider = FakeProvider

        modules = {
            "pydantic_ai": pydantic_ai,
            "pydantic_ai.models": models,
            "pydantic_ai.models.openai": openai_models,
            "pydantic_ai.providers": providers,
            "pydantic_ai.providers.openai": openai_provider,
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict("sys.modules", modules):
            console = self.make_console(Path(temp_dir))
            console.console_vars.update({
                "assistant_api_base": "https://example.invalid/v1/chat/completions",
                "assistant_api_key": "secret",
                "assistant_model": "dpsk-v4-flash",
            })
            result = console.agent_service._run_pydantic_agent("agent", "采样改为 7", "", "session1234")
            self.assertTrue(result.ok)
            self.assertEqual(result.message, "参数已修改")
            self.assertEqual(result.reasoning, "内部思考")
            self.assertEqual(console.form_vars["TRAIN_NUM"], "7")
            self.assertIsNotNone(result.config)
            self.assertEqual(FakeAgent.last.model.provider.base_url, "https://example.invalid/v1")
            self.assertEqual(FakeAgent.last.settings["extra_body"], {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            })
            console.console_vars["assistant_reasoning"] = "0"  # Legacy value must be ignored.
            console.agent_service._run_pydantic_agent("agent", "采样改为 7", "", "session1234")
            self.assertEqual(FakeAgent.last.settings["extra_body"], {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            })
            console.agent_service._run_pydantic_agent("ask", "如何查看历史归档", "", "session1234")
            self.assertEqual(FakeAgent.last.settings["extra_body"], {"thinking": {"type": "disabled"}})

    def test_chat_stream_orders_reasoning_before_answer_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))

            def fake_run(mode, user_message, history, session_id, event_sink=None, history_view=False):
                self.assertIsNotNone(event_sink)
                self.assertFalse(history_view)
                event_sink({"type": "reasoning_delta", "delta": "先分析"})
                event_sink({"type": "answer_delta", "delta": "再回答"})
                return AgentChatResponse(
                    ok=True,
                    mode="agent",
                    session_id=session_id,
                    message="再回答",
                    reasoning="先分析",
                )

            payload = {
                "mode": "agent",
                "session_id": "streamsession",
                "messages": [{"role": "user", "content": "测试流式输出"}],
            }
            with patch.object(console.agent_service, "_run_pydantic_agent", side_effect=fake_run):
                events = [json.loads(line) for line in console.agent_service.chat_stream(payload)]
            self.assertEqual([event["type"] for event in events], ["meta", "reasoning_delta", "answer_delta", "done"])
            stored = console.agent_service.sessions.load("streamsession")
            self.assertEqual(stored["messages"][-1]["content"], "再回答")
            self.assertEqual(stored["messages"][-1]["reasoning"], "先分析")

    def test_chat_stream_persists_failed_exchange(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            payload = {
                "mode": "agent",
                "session_id": "failedsession",
                "messages": [{"role": "user", "content": "触发失败"}],
            }
            with patch.object(console.agent_service, "_run_pydantic_agent", side_effect=RuntimeError("mock failure")):
                events = [json.loads(line) for line in console.agent_service.chat_stream(payload)]
            self.assertEqual(events[-1]["type"], "error")
            stored = console.agent_service.sessions.load("failedsession")
            self.assertEqual([item["role"] for item in stored["messages"]], ["user", "assistant"])
            self.assertIn("mock failure", stored["messages"][-1]["content"])

    def test_context_limit_setting_is_refreshed_for_each_exchange(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            console.console_vars["assistant_context_limit"] = 32768
            with patch.object(
                console.agent_service,
                "_run_pydantic_agent",
                return_value=AgentChatResponse(ok=True, mode="ask", session_id="limitsession", message="ok"),
            ):
                console.agent_service.chat({
                    "mode": "ask",
                    "session_id": "limitsession",
                    "messages": [{"role": "user", "content": "hello"}],
                })
            self.assertEqual(console.agent_service.sessions.token_limit, 32768)

    def test_pydantic_stream_event_payloads(self) -> None:
        from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, ThinkingPartDelta

        with tempfile.TemporaryDirectory() as temp_dir:
            console = self.make_console(Path(temp_dir))
            reasoning = console.agent_service._stream_event_payload(
                PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="思考"))
            )
            answer = console.agent_service._stream_event_payload(
                PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="回答"))
            )
            self.assertEqual(reasoning, {"type": "reasoning_delta", "delta": "思考"})
            self.assertEqual(answer, {"type": "answer_delta", "delta": "回答"})


if __name__ == "__main__":
    unittest.main()
