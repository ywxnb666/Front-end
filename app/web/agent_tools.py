from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.web.agent_schemas import ActionProposal, ToolEvent


@dataclass
class AgentToolContext:
    console: Any
    create_action: Callable[[str, str, str, dict[str, Any]], ActionProposal]
    allow_mutations: bool = True
    events: list[ToolEvent] = field(default_factory=list)
    pending_action: ActionProposal | None = None

    def event(self, name: str, status: str, summary: str = "", data: dict[str, Any] | None = None) -> None:
        self.events.append(ToolEvent(name=name, status=status, summary=summary, data=data or {}))


def _require_mutations(context: AgentToolContext) -> None:
    if not context.allow_mutations:
        raise ValueError("当前正在查看历史归档，请返回实时数据后再修改参数或执行操作")


PARAMETER_ALIASES: dict[str, tuple[str, str | tuple[str, ...], str]] = {
    "train_num": ("form_vars", "TRAIN_NUM", "int"),
    "max_samples": ("form_vars", "MAX_SAMPLES", "int"),
    "control_max_samples": ("form_vars", "SCIENCEQA_CONTROL_MAX_SAMPLES", "int"),
    "eval_max_samples": ("form_vars", "EVAL_MAX_SAMPLES", "int"),
    "judge_sample_num": ("console_vars", "judge_sample_num", "int"),
    "stage1_epochs": ("form_vars", "STAGE1_EPOCHS", "int"),
    "stage1_batch_size": ("form_vars", "STAGE1_BATCH_SIZE", "int"),
    "stage1_grad_accum": ("form_vars", "STAGE1_GRAD_ACCUM", "int"),
    "stage1_max_length": ("form_vars", "STAGE1_MAX_LENGTH", "int"),
    "stage2_epochs": ("form_vars", "STAGE2_EPOCHS", "int"),
    "period_num": ("form_vars", "PERIOD_NUM", "int"),
    "stage2_batch_size": ("form_vars", ("PHASE_A_BATCH_SIZE", "PHASE_B_BATCH_SIZE"), "int"),
    "stage2_grad_accum": ("form_vars", "STAGE2_GRAD_ACCUM", "int"),
    "stage2_max_length": ("form_vars", "STAGE2_MAX_LENGTH", "int"),
    "eval_max_new_tokens": ("form_vars", "EVAL_MAX_NEW_TOKENS", "int"),
    "use_4bit": ("form_vars", "USE_4BIT", "bool"),
    "freeze_vision_tower": ("form_vars", "FREEZE_VISION_TOWER", "bool"),
    "stage2_wrong_image_enable": ("form_vars", "STAGE2_WRONG_IMAGE_ENABLE", "bool"),
    "stage2_pair_use_answer_correctness": ("form_vars", "STAGE2_PAIR_USE_ANSWER_CORRECTNESS", "bool"),
    "dataset_name": ("console_vars", "dataset_name", "text"),
    "cuda_devices": ("console_vars", "cuda_devices", "text"),
    "teacher_api_base": ("console_vars", "teacher_api_base", "text"),
    "victim_model": ("console_vars", "victim_model", "text"),
    "sim_duration": ("console_vars", "sim_duration", "int"),
    "auto_refresh": ("console_vars", "auto_refresh", "bool"),
    "judge_model": ("console_vars", "judge_model", "text"),
    "judge_api_base": ("console_vars", "judge_api_base", "text"),
    "judge_eval_api_base": ("form_vars", "JUDGE_API_BASE", "text"),
    "assistant_api_base": ("console_vars", "assistant_api_base", "text"),
    "assistant_model": ("console_vars", "assistant_model", "text"),
    "assistant_context_limit": ("console_vars", "assistant_context_limit", "int"),
    "watermark_model_name": ("console_vars", "watermark_model_name", "text"),
    "watermark_torch_dtype": ("console_vars", "watermark_torch_dtype", "text"),
    "watermark_device": ("console_vars", "watermark_device", "text"),
    "watermark_sample_fraction": ("console_vars", "watermark_sample_fraction", "text"),
    "watermark_sample_size": ("console_vars", "watermark_sample_size", "int"),
    "scienceqa_split": ("form_vars", "SCIENCEQA_SPLIT", "text"),
    "scienceqa_seed": ("form_vars", "SCIENCEQA_SEED", "text"),
    "teacher_lang": ("form_vars", "TEACHER_LANG", "text"),
    "teacher_enable_thinking": ("form_vars", "TEACHER_ENABLE_THINKING", "bool"),
    "collect_teacher_data": ("form_vars", "COLLECT_TEACHER_DATA", "bool"),
    "strict_teacher_distill": ("form_vars", "STRICT_TEACHER_DISTILL", "bool"),
    "num_workers": ("form_vars", "NUM_WORKERS", "int"),
    "max_new_tokens": ("form_vars", "MAX_NEW_TOKENS", "int"),
    "max_concurrency": ("form_vars", "MAX_CONCURRENCY", "int"),
    "scienceqa_control_split": ("form_vars", "SCIENCEQA_CONTROL_SPLIT", "text"),
    "scienceqa_controls": ("form_vars", "SCIENCEQA_CONTROLS", "text"),
    "stage1_lr": ("form_vars", "STAGE1_LR", "text"),
    "lora_rank": ("form_vars", "LORA_RANK", "int"),
    "lora_alpha": ("form_vars", "LORA_ALPHA", "int"),
    "stage1_field_weight_reasoning": ("form_vars", "STAGE1_FIELD_WEIGHT_REASONING", "text"),
    "stage1_field_weight_answer": ("form_vars", "STAGE1_FIELD_WEIGHT_ANSWER", "text"),
    "stage2_lr": ("form_vars", "STAGE2_LR", "text"),
    "tau1": ("form_vars", "TAU1", "text"),
    "stage2_eval_every_period": ("form_vars", "STAGE2_EVAL_EVERY_PERIOD", "int"),
    "stage2_eval_train_num": ("form_vars", "STAGE2_EVAL_TRAIN_NUM", "int"),
    "stage2_eval_max_samples": ("form_vars", "STAGE2_EVAL_MAX_SAMPLES", "int"),
    "require_valid_format": ("form_vars", "REQUIRE_VALID_FORMAT", "bool"),
    "judge_dataset_name": ("form_vars", "JUDGE_DATASET_NAME", "text"),
    "judge_split": ("form_vars", "SPLIT", "text"),
    "parallel_controls": ("form_vars", "PARALLEL_CONTROLS", "bool"),
    "dataset_path": ("console_vars", "dataset_path", "path"),
    "model_path": ("console_vars", "model_path", "path"),
    "stage1_ckpt": ("console_vars", "stage1_ckpt", "path"),
    "stage2_adapter": ("console_vars", "stage2_adapter", "path"),
    "result_dir": ("console_vars", "result_dir", "path"),
}

PARAMETER_RANGES: dict[str, tuple[int, int]] = {
    "train_num": (0, 100000),
    "max_samples": (0, 100000),
    "control_max_samples": (0, 100000),
    "eval_max_samples": (0, 100000),
    "judge_sample_num": (1, 100000),
    "stage1_epochs": (1, 1000),
    "stage1_batch_size": (1, 64),
    "stage1_grad_accum": (1, 1024),
    "stage1_max_length": (128, 32768),
    "stage2_epochs": (1, 1000),
    "period_num": (1, 1000),
    "stage2_batch_size": (1, 64),
    "stage2_grad_accum": (1, 1024),
    "stage2_max_length": (128, 32768),
    "eval_max_new_tokens": (1, 32768),
    "sim_duration": (1, 3600),
    "watermark_sample_size": (0, 100000),
    "num_workers": (1, 256),
    "max_new_tokens": (1, 32768),
    "max_concurrency": (1, 256),
    "lora_rank": (1, 1024),
    "lora_alpha": (1, 4096),
    "stage2_eval_every_period": (1, 1000),
    "stage2_eval_train_num": (0, 100000),
    "stage2_eval_max_samples": (0, 100000),
    "assistant_context_limit": (8192, 1048576),
}

PRESETS: dict[str, dict[str, Any]] = {
    "demo": {
        "train_num": 3,
        "max_samples": 3,
        "control_max_samples": 3,
        "eval_max_samples": 3,
        "judge_sample_num": 3,
        "stage1_epochs": 1,
        "stage2_epochs": 1,
        "period_num": 1,
    },
    "recommended": {
        "train_num": 0,
        "max_samples": 0,
        "control_max_samples": 200,
        "eval_max_samples": 200,
        "judge_sample_num": 500,
        "stage1_epochs": 3,
        "stage1_batch_size": 1,
        "stage1_grad_accum": 2,
        "stage2_epochs": 1,
        "period_num": 1,
        "stage2_grad_accum": 2,
        "use_4bit": False,
        "freeze_vision_tower": True,
    },
    # Small reproducible run settings. Connection, hardware, UI and assistant
    # configuration stay untouched so the preset remains portable.
    "演示推荐参数": {
        "train_num": 1,
        "max_samples": 1,
        "control_max_samples": 1,
        "eval_max_samples": 1,
        "judge_sample_num": 5,
        "stage1_epochs": 1,
        "stage1_batch_size": 1,
        "stage1_grad_accum": 2,
        "stage1_max_length": 1536,
        "stage2_epochs": 1,
        "period_num": 1,
        "stage2_batch_size": 1,
        "stage2_grad_accum": 2,
        "stage2_max_length": 1024,
        "eval_max_new_tokens": 512,
        "use_4bit": False,
        "freeze_vision_tower": True,
        "stage2_wrong_image_enable": False,
        "stage2_pair_use_answer_correctness": False,
        "victim_model": "qwen/qwen3.5-flash-02-23",
        "judge_model": "gpt-5.5",
        "watermark_model_name": "qwen2-vl",
        "watermark_sample_fraction": "0.2",
        "watermark_sample_size": 1,
        "scienceqa_split": "train",
        "scienceqa_seed": "20240306",
        "teacher_lang": "en",
        "teacher_enable_thinking": False,
        "collect_teacher_data": True,
        "strict_teacher_distill": True,
        "num_workers": 1,
        "max_new_tokens": 64,
        "max_concurrency": 4,
        "scienceqa_control_split": "test",
        "scienceqa_controls": "baseline,text_only_blank,hint_ablation,option_shuffle,random_image_swap,image_blur,image_downsample",
        "stage1_lr": "3e-5",
        "lora_rank": 16,
        "lora_alpha": 32,
        "stage1_field_weight_reasoning": "2.0",
        "stage1_field_weight_answer": "12.0",
        "stage2_lr": "2e-5",
        "tau1": "0.02",
        "stage2_eval_every_period": 1,
        "stage2_eval_train_num": 1,
        "stage2_eval_max_samples": 1,
        "require_valid_format": True,
        "judge_dataset_name": "scienceqa",
        "judge_split": "test",
        "parallel_controls": True,
    },
    "low_memory": {
        "stage1_batch_size": 1,
        "stage1_grad_accum": 4,
        "stage1_max_length": 1024,
        "stage2_batch_size": 1,
        "stage2_grad_accum": 4,
        "stage2_max_length": 768,
        "eval_max_new_tokens": 256,
        "use_4bit": True,
        "freeze_vision_tower": True,
    },
}

PRESET_ALIASES = {
    "demo_recommended": "演示推荐参数",
    "demo-recommended": "演示推荐参数",
    "演示推荐": "演示推荐参数",
}

PIPELINE_STEP_IDS = [
    "teacher_collect",
    "teacher_eval",
    "origin_eval",
    "stage1_train",
    "stage2_train",
    "student_eval",
    "reason_judge",
    "risk_report",
]

SAFE_CONSOLE_SECRET_KEYS = {"teacher_api_key", "judge_api_key", "assistant_api_key"}
IMPORTANT_FORM_KEYS = {
    "TRAIN_NUM", "MAX_SAMPLES", "SCIENCEQA_CONTROL_MAX_SAMPLES", "EVAL_MAX_SAMPLES",
    "STAGE1_BATCH_SIZE", "STAGE1_GRAD_ACCUM", "STAGE2_GRAD_ACCUM", "USE_4BIT",
    "FREEZE_VISION_TOWER", "SAMPLE_NUM",
}


def _coerce(alias: str, value: Any, kind: str) -> Any:
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"{alias} 必须是整数")
        parsed = int(value)
        low, high = PARAMETER_RANGES[alias]
        if parsed < low or parsed > high:
            raise ValueError(f"{alias} 必须在 {low} 到 {high} 之间")
        return str(parsed)
    if kind == "bool":
        if isinstance(value, bool):
            return "1" if value else "0"
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return "1"
        if text in {"0", "false", "no", "off"}:
            return "0"
        raise ValueError(f"{alias} 必须是布尔值")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{alias} 不能为空")
    return text


def _apply_patches(context: AgentToolContext, patches: dict[str, Any]) -> dict[str, Any]:
    console_vars: dict[str, Any] = {}
    form_vars: dict[str, Any] = {}
    changes: list[dict[str, Any]] = []
    for alias, value in patches.items():
        key = str(alias).strip().lower()
        if key not in PARAMETER_ALIASES:
            raise ValueError(f"不允许修改参数: {alias}")
        scope, target, kind = PARAMETER_ALIASES[key]
        coerced = _coerce(key, value, kind)
        targets = target if isinstance(target, tuple) else (target,)
        current_values = [
            context.console.console_vars.get(item) if scope == "console_vars" else context.console.form_vars.get(item)
            for item in targets
        ]
        changed_targets = [item for item, current in zip(targets, current_values) if str(current) != str(coerced)]
        if not changed_targets:
            continue
        destination = console_vars if scope == "console_vars" else form_vars
        for item in changed_targets:
            destination[item] = coerced
        previous: Any = current_values[0] if len(current_values) == 1 else dict(zip(targets, current_values))
        changes.append({"parameter": key, "from": previous, "to": coerced})
    if console_vars or form_vars:
        context.console.update_config({"console_vars": console_vars, "form_vars": form_vars})
    return {"changed": changes, "count": len(changes)}


def safe_app_state_snapshot(console_instance: Any) -> dict[str, Any]:
    with console_instance.lock:
        console = dict(console_instance.console_vars)
        form = dict(console_instance.form_vars)
    console = {key: value for key, value in console.items() if key not in SAFE_CONSOLE_SECRET_KEYS}
    form = {key: value for key, value in form.items() if "API_KEY" not in key.upper()}
    return {
        "console_vars": console,
        "important_form_vars": {key: form.get(key) for key in sorted(IMPORTANT_FORM_KEYS) if key in form},
    }


def read_app_state(context: AgentToolContext) -> dict[str, Any]:
    context.event("read_app_state", "started")
    result = {
        **safe_app_state_snapshot(context.console),
        "remote_configured": bool(context.console.connection_command.strip()),
        "connection": context.console.connection_command.split("@")[-1] if "@" in context.console.connection_command else "未配置",
    }
    context.event("read_app_state", "completed", "已读取脱敏配置")
    return result


def read_pipeline_status(context: AgentToolContext) -> dict[str, Any]:
    context.event("read_pipeline_status", "started")
    status = context.console.pipeline_status()
    result = {"running": status.get("running", False), "summary": status.get("summary", ""), "rows": status.get("rows", [])}
    context.event("read_pipeline_status", "completed", str(result["summary"]))
    return result


def read_recent_logs(context: AgentToolContext, limit: int = 80) -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    context.event("read_recent_logs", "started")
    raw_logs = context.console.pipeline_status().get("logs", [])[-limit:]
    logs: list[str] = []
    remaining = 30000
    for line in reversed(raw_logs):
        text = str(line)[:2000]
        if remaining <= 0:
            break
        logs.append(text[:remaining])
        remaining -= len(logs[-1])
    logs.reverse()
    context.event("read_recent_logs", "completed", f"读取 {len(logs)} 行日志")
    return {"logs": logs}


def read_results(context: AgentToolContext) -> dict[str, Any]:
    context.event("read_results", "started")
    result = {
        "dashboard": context.console.dashboard_payload(),
        "watermark": context.console.watermark_payload(),
        "history": context.console.history_list()[:20],
    }
    context.event("read_results", "completed", "已读取结果摘要")
    return result


def validate_remote_setup(context: AgentToolContext) -> dict[str, Any]:
    context.event("validate_remote_setup", "started")
    with context.console.lock:
        values = dict(context.console.console_vars)
        form = dict(context.console.form_vars)
    paths = {
        "root_dir": (values.get("root_dir", ""), "directory"),
        "dataset_path": (values.get("dataset_path", ""), "directory"),
        "model_path": (values.get("model_path", ""), "directory"),
        "python_bin": (values.get("python_bin", ""), "executable"),
        "stage1_ckpt": (values.get("stage1_ckpt", ""), "exists"),
        "stage2_adapter": (values.get("stage2_adapter", ""), "directory"),
        "stage2_result_json": (form.get("STAGE2", values.get("reason_stage2_json", "")), "file"),
        "stage3_result_json": (form.get("STAGE3", values.get("reason_stage3_json", "")), "file"),
    }
    checks: dict[str, bool] = {}
    errors: list[str] = []
    remote = context.console.should_run_remote()
    test_flags = {"directory": "-d", "executable": "-x", "file": "-f", "exists": "-e"}
    for name, (path, kind) in paths.items():
        text = str(path or "").strip()
        if not text:
            checks[name] = False
            continue
        try:
            if remote:
                result = context.console.run_remote_config_command(f"test {test_flags[kind]} {shlex.quote(text)}")
                checks[name] = result.exit_code == 0
                if result.exit_code != 0 and (result.stderr or result.stdout):
                    errors.append(f"{name}: {(result.stderr or result.stdout).strip()[:200]}")
            else:
                path_obj = Path(text)
                checks[name] = {
                    "directory": path_obj.is_dir,
                    "executable": lambda: path_obj.is_file() and path_obj.stat().st_mode & 0o111 != 0,
                    "file": path_obj.is_file,
                    "exists": path_obj.exists,
                }[kind]()
        except Exception as exc:
            checks[name] = False
            errors.append(f"{name}: {exc}")
    result = {
        "ssh_configured": remote,
        "checks": checks,
        "missing": [name for name, ok in checks.items() if not ok],
        "errors": errors,
        "ready": bool(checks) and all(checks.values()),
    }
    summary = "环境检查通过" if result["ready"] else f"环境检查发现 {len(result['missing'])} 项异常"
    context.event("validate_remote_setup", "completed", summary, result)
    return result


def set_allowed_parameters(context: AgentToolContext, patches: dict[str, Any]) -> dict[str, Any]:
    _require_mutations(context)
    context.event("set_allowed_parameters", "started")
    try:
        result = _apply_patches(context, patches)
    except Exception as exc:
        context.event("set_allowed_parameters", "failed", str(exc))
        raise
    context.event("set_allowed_parameters", "completed", f"修改 {result['count']} 项参数", result)
    return result


def apply_preset(context: AgentToolContext, preset: str) -> dict[str, Any]:
    _require_mutations(context)
    requested = str(preset).strip()
    name = PRESET_ALIASES.get(requested.lower(), requested)
    if name not in PRESETS:
        raise ValueError(f"未知预设: {preset}，可选: {', '.join(PRESETS)}")
    context.event("apply_preset", "started")
    result = _apply_patches(context, PRESETS[name])
    result["preset"] = name
    context.event("apply_preset", "completed", f"已应用 {name} 预设", result)
    return result


def propose_pipeline_run(context: AgentToolContext, steps: list[str] | None = None, full: bool = False) -> dict[str, Any]:
    _require_mutations(context)
    if context.pending_action is not None:
        return {"pending_action_id": context.pending_action.action_id, "requires_confirmation": True, "already_proposed": True}
    status = context.console.pipeline_status()
    if status.get("running"):
        raise ValueError("已有 Pipeline 正在运行，不能重复启动")
    allowed = set(PIPELINE_STEP_IDS)
    if full:
        requested = list(PIPELINE_STEP_IDS)
    elif steps:
        requested = list(dict.fromkeys(str(step).strip() for step in steps))
    else:
        raise ValueError("请明确指定 Pipeline 阶段，或设置 full=true 运行完整流程")
    unknown = [step for step in requested if step not in allowed]
    if unknown:
        raise ValueError(f"未知 Pipeline 阶段: {', '.join(unknown)}")
    definitions = context.console.task_definitions()
    plan = [{"id": step, "stage": definitions[step][0]} for step in requested]
    skipped: list[dict[str, str]] = []
    for step in ("teacher_collect", "teacher_eval"):
        if step not in requested:
            continue
        source = context.console.reuse_stage_source(step)
        if source:
            try:
                exists = context.console.remote_or_local_file_exists(source)
            except Exception:
                exists = False
            if exists:
                plan = [item for item in plan if item["id"] != step]
                skipped.append({"id": step, "reason": f"复用已有文件: {source}"})
    if not plan:
        summary = "请求阶段均已由现有文件复用，无需启动 Pipeline"
        context.event("propose_pipeline_run", "completed", summary, {"plan": [], "skipped": skipped})
        return {"plan": [], "skipped": skipped, "requires_confirmation": False, "message": summary}
    detail = "将运行: " + ("、".join(item["stage"] for item in plan) or "无")
    action = context.create_action("pipeline", "确认运行 Pipeline", detail, {"steps": [item["id"] for item in plan]})
    context.pending_action = action
    context.event("propose_pipeline_run", "pending", detail, {"plan": plan, "skipped": skipped})
    return {"pending_action_id": action.action_id, "plan": plan, "skipped": skipped, "requires_confirmation": True}


def propose_remote_config_save(context: AgentToolContext) -> dict[str, Any]:
    _require_mutations(context)
    if context.pending_action is not None:
        return {"pending_action_id": context.pending_action.action_id, "requires_confirmation": True, "already_proposed": True}
    action = context.create_action("remote_config", "确认同步服务器配置", "将当前脱敏之外的完整配置同步到服务器配置文件", {})
    context.pending_action = action
    context.event("save_remote_config", "pending", "等待用户确认")
    return {"pending_action_id": action.action_id, "requires_confirmation": True}


def save_history(context: AgentToolContext, label: str = "") -> dict[str, Any]:
    _require_mutations(context)
    context.event("save_history", "started")
    result = context.console.save_history(label=label)
    status = "completed" if result.get("ok") else "failed"
    context.event("save_history", status, str(result.get("message", "")), result)
    return result
