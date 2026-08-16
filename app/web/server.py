from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import APP_DIR, AppConfig, load_config, save_config
from app.models import CloneRequest
from app.services.git_service import REPO_DIR_NAME, RemoteGitCloneService
from app.services.ssh_client import SSHClientManager, parse_connection_command
from app.web.risk_evaluation import (
    COT_WEIGHTS,
    calculate_capability_leakage,
    calculate_watermark_erosion,
    extract_baseline_accuracy,
    extract_control_summary,
)


PIPELINE_STEPS: list[tuple[str, str, str]] = [
    ("teacher_collect", "教师 API 数据采集", "scripts2/teacher_model_data_collect.sh"),
    ("teacher_eval", "教师完整风险基线", "scripts2/run_full_eval_pipeline_teacher.sh"),
    ("origin_eval", "原始学生能力基线", "scripts2/run_full_eval_pipeline_origin.sh"),
    ("stage1_train", "Stage1 学生蒸馏", "scripts2/run_stage1.sh"),
    ("stage2_train", "Stage2 学生蒸馏", "scripts2/run_stage2.sh"),
    ("student_eval", "学生完整风险评估", "scripts2/run_full_eval_pipeline_fast.sh"),
    ("reason_judge", "思维链评估", "reason_judge/run_judge.sh"),
    ("risk_report", "风险报告聚合", "dashboard aggregation"),
]

OBSOLETE_FORM_KEYS = {
    "RUN_TEACHER_SPECIAL_BENCHMARKS",
    "TEACHER_BENCHMARKS",
    "TEACHER_MAX_SAMPLES_PER_BENCHMARK",
    "AI2D_DATASET",
    "AI2D_SPLIT",
    "CHARTQA_DATASET",
    "CHARTQA_SPLIT",
}

BOOL_ENV_KEYS = {
    "TEACHER_ENABLE_THINKING",
    "COLLECT_TEACHER_DATA",
    "STRICT_TEACHER_DISTILL",
    "SAMPLE_ONLY_CACHED_TEACHER",
    "REUSE_VQ_CODEBOOK",
    "REMOVE_VQ_CODEBOOK",
    "PARALLEL_CONTROLS",
    "RUN_TEACHER_SPECIAL_BENCHMARKS",
    "USE_LORA",
    "USE_4BIT",
    "FREEZE_VISION_TOWER",
    "RUN_STAGE1_EVAL",
    "STAGE2_WRONG_IMAGE_ENABLE",
    "STAGE2_PAIR_USE_ANSWER_CORRECTNESS",
    "STAGE2_TRAIN_PROJECTOR",
    "REUSE_STAGE1",
    "REQUIRE_VALID_FORMAT",
}

# ---------------------------------------------------------------------------
# Fabricated results for --debug (frontend demo only, never used in live mode).
# Numbers are hand-picked to stay self-consistent across origin, victim, and
# extracted-student evaluations.
# ---------------------------------------------------------------------------
SIM_CONTROL_ACCURACY: dict[str, tuple[float, float]] = {
    # control name: (teacher accuracy, student accuracy)
    "baseline": (0.8140, 0.6390),
    "text_only_blank": (0.4620, 0.4050),
    "hint_ablation": (0.7580, 0.5720),
    "option_shuffle": (0.7930, 0.6010),
    "random_image_swap": (0.6210, 0.5240),
    "image_blur": (0.7350, 0.5610),
    "image_downsample": (0.7710, 0.5880),
}

SIM_ORIGIN_ACCURACY: dict[str, float] = {
    "baseline": 0.4120,
    "text_only_blank": 0.3010,
    "hint_ablation": 0.3840,
    "option_shuffle": 0.3910,
    "random_image_swap": 0.3380,
    "image_blur": 0.3790,
    "image_downsample": 0.3980,
}

SIM_REASON_SUMMARY: dict[str, str] = {
    "n": "120",
    "stage2_reason_score": "3.4200",
    "stage3_reason_score": "3.8600",
    "stage3_visual_grounding": "4.1000",
    "stage3_logical_correctness": "3.9000",
    "stage3_answer_support": "4.0000",
    "stage3_question_relevance": "3.7000",
    "stage3_option_discrimination": "3.6000",
    "delta_reason_score": "0.4400",
    "stage2_win_rate": "0.5750",
}

SIM_WATERMARK_REPORT: dict[str, Any] = {
    "num_scored": 96,
    "num_total_records": 480,
    "split": "test",
    "metrics": {
        "mean_z": 4.8130,
        "median_z": 4.6540,
        "min_z": 1.2870,
        "max_z": 9.4120,
        "threshold_4_rate": 0.6420,
    },
}


HISTORY_DIR = APP_DIR / "history"
HISTORY_LOG_LINES = 200          # cap stored logs so snapshots stay small
SECRET_CONSOLE_KEYS = {"teacher_api_key", "judge_api_key", "assistant_api_key"}
SECRET_FORM_KEY_HINT = "API_KEY"

ASSISTANT_SYSTEM_PROMPT = """你是 MLLM 能力泄漏风险检测平台的内置 AI 助手，负责帮助用户理解和使用这个软件。

回答原则：
1. 使用简洁、准确的中文回答；用户询问参数时，说明它的作用、常见取值、对运行时间/显存/结果的影响。
2. 只根据本平台的功能回答。不要假装已经运行任务、访问服务器文件或看到了用户没有提供的日志。
3. 区分本地笔记本前端和远程服务器后端：路径、模型、数据集和检查点通常指服务器上的路径；配置和 API 密钥保存在前端所在电脑的本地配置中，SSH 连接认证仍由前端负责。
4. 当前平台流程通常是：教师标注采集 -> 教师风险基线 -> Stage1 学生蒸馏 -> Stage2 学生蒸馏 -> 学生完整风险评估 -> 思维链评估 -> 风险报告聚合。用户启用“复用已有教师结果”时，对应教师阶段会跳过；复用的 JSON 必须位于服务器且文件存在。
5. Debug 模式只模拟前端进度和日志，不执行真实后端实验；正常模式才通过 SSH 执行服务器上的脚本。历史记录是结果快照，不会重新运行任务。
6. 教师采集的 TRAIN_NUM/MAX_SAMPLES 控制训练标注数量；SCIENCEQA_CONTROL_MAX_SAMPLES 控制教师风险基线控制集数量；EVAL_MAX_SAMPLES 控制学生风险评估数量；思维链评估的 SAMPLE_NUM 控制 judge 抽样数量。batch size 是每步样本数，显存不足优先减小它、增加梯度累积或启用 4bit/冻结视觉塔。
7. 如果用户贴出报错，先指出最可能的阶段和直接原因，再给出前端可调整的参数或需要检查的服务器路径。不要建议修改 SSH 认证，除非用户明确要求。

平台功能：左侧配置 SSH、路径、教师模型和运行参数；“一键跑完整 Pipeline”执行全流程；状态页显示阶段进度和终端日志；数据页管理教师结果复用；训练页配置 Stage1/Stage2；完整评测页运行学生评测和思维链 judge；风险大盘展示控制集对比和风险指标；水印检测页运行 z-mean 检测；历史测评数据可以归档、选择查看和删除。

如果无法确定某个脚本的具体行为，明确说需要查看对应服务器日志或脚本，而不是编造结论。
只输出最终答案，不要输出思考过程、分析过程、草稿、推理步骤或名为 Thinking Process 的内容；不要使用 <think> 标签。"""


def _strip_secrets(console_vars: dict[str, Any], form_vars: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    """Snapshots are plain files on disk: keep API keys out of them."""
    console = {k: ("" if k in SECRET_CONSOLE_KEYS else v) for k, v in console_vars.items()}
    form = {k: ("" if SECRET_FORM_KEY_HINT in k.upper() else v) for k, v in form_vars.items()}
    return console, form


def _history_has_results(dashboard: dict[str, Any], watermark: dict[str, Any]) -> bool:
    metrics = dashboard.get("metrics", {}) if isinstance(dashboard, dict) else {}
    if any(str(metrics.get(key, "-")) not in ("", "-") for key in ("clr", "risk", "teacher_acc", "student_acc", "wer")):
        return True
    return bool(watermark.get("ok")) if isinstance(watermark, dict) else False


def _sim_control_suite(index: int) -> dict[str, Any]:
    """Shape a fake control-suite result file the way the real scripts write it."""
    summary = {name: {"accuracy": scores[index]} for name, scores in SIM_CONTROL_ACCURACY.items()}
    return {
        "metrics": {
            "baseline_accuracy": SIM_CONTROL_ACCURACY["baseline"][index],
            "control_summary": summary,
        }
    }


def sim_teacher_result() -> dict[str, Any]:
    return _sim_control_suite(0)


def sim_student_result() -> dict[str, Any]:
    return _sim_control_suite(1)


def sim_origin_result() -> dict[str, Any]:
    summary = {name: {"accuracy": score} for name, score in SIM_ORIGIN_ACCURACY.items()}
    return {"metrics": {"baseline_accuracy": SIM_ORIGIN_ACCURACY["baseline"], "control_summary": summary}}


def _startup_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _looks_like_repo(path: Path) -> bool:
    return path.is_dir() and (
        ((path / ".git").is_dir() and path.name == REPO_DIR_NAME)
        or (path / "vq_lord3").is_dir()
        or (path / "fastapi_vqlord").is_dir()
    )


def _startup_repo_path() -> Path | None:
    base_dir = _startup_base_dir()
    candidates = [
        base_dir,
        base_dir / REPO_DIR_NAME,
        base_dir.parent / REPO_DIR_NAME,
    ]
    for candidate in candidates:
        if _looks_like_repo(candidate):
            return candidate
    return None


def _remote_join(*parts: str) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    return posixpath.normpath(posixpath.join(*cleaned))


def _remote_parent(path: str) -> str:
    normalized = posixpath.normpath(path.strip().rstrip("/") or ".")
    parent = posixpath.dirname(normalized)
    return parent or "."


def _is_windows_path(path: str) -> bool:
    stripped = path.strip()
    return (len(stripped) >= 2 and stripped[1] == ":") or "\\" in stripped


def _path_join(*parts: str) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    if _is_windows_path(cleaned[0]):
        return str(PureWindowsPath(cleaned[0], *cleaned[1:]))
    return _remote_join(*cleaned)


def _path_parent(path: str) -> str:
    if _is_windows_path(path):
        parent = PureWindowsPath(path.strip().rstrip("\\/")).parent
        return str(parent) if str(parent) != "." else "."
    return _remote_parent(path)


def _local_path_exists(path: str) -> bool:
    if not _is_windows_path(path) and not path.startswith("/"):
        return False
    return Path(path).exists()


def _default_vla_mark_dir(repo_root: str) -> str:
    candidates = [
        _path_join(repo_root, "VLA-mark"),
        _path_join(_path_parent(repo_root), "VLA-mark"),
    ]
    for candidate in candidates:
        if _local_path_exists(candidate):
            return candidate
    return candidates[0]


def _default_project_root(repo_root: str) -> str:
    return _path_parent(repo_root)


def _default_python_bin(repo_root: str) -> str:
    configured = os.environ.get("PYTHON_BIN") or os.environ.get("PYTHON")
    if configured:
        return configured
    if _is_windows_path(repo_root):
        for candidate in (
            _path_join(repo_root, ".venv", "Scripts", "python.exe"),
            _path_join(repo_root, "venv", "Scripts", "python.exe"),
        ):
            if _local_path_exists(candidate):
                return candidate
        if not getattr(sys, "frozen", False):
            return sys.executable
        return "python"
    return "python3"


def _default_model_path(repo_root: str) -> str:
    configured = os.environ.get("MODEL_PATH")
    if configured:
        return configured
    return _path_join(_default_project_root(repo_root), "models")


def _default_dataset_path(repo_root: str) -> str:
    configured = os.environ.get("DATASET_PATH")
    if configured:
        return configured
    return _path_join(_default_project_root(repo_root), "datasets")


def _console_path_defaults(repo_root: str) -> dict[str, str]:
    repo_root = posixpath.normpath(repo_root.strip().rstrip("/") or REPO_DIR_NAME)
    result_dir = _path_join(repo_root, "vq_lord_test_results")
    stage2_adapter = _path_join(repo_root, "vq_lord_ckpts", "stage2", "stage2_lord_final")
    reason_judge_dir = _path_join(repo_root, "reason_judge")
    return {
        "root_dir": repo_root,
        "python_bin": _default_python_bin(repo_root),
        "model_path": _default_model_path(repo_root),
        "reason_judge_dir": reason_judge_dir,
        "vla_mark_dir": _default_vla_mark_dir(repo_root),
        "dataset_path": _default_dataset_path(repo_root),
        "stage1_ckpt": _path_join(repo_root, "vq_lord_ckpts", "stage1", "stage1_vision_epoch1"),
        "stage2_adapter": stage2_adapter,
        "result_dir": result_dir,
        "teacher_result_dir": _path_join(result_dir, "teacher_compare"),
        "stage2_codebook": _path_join(stage2_adapter, "vq_codebook.pt"),
        "reason_stage2_json": _path_join(result_dir, "stage2_test_generate_readable.json"),
        "reason_stage3_json": _path_join(result_dir, "stage3_test_generate_vq1_bucketed_parallel.json"),
        "reason_teacher_json": "",
        "reason_dataset_path": _default_dataset_path(repo_root),
        "reason_out_dir": "outputs/judge_latest",
        "watermark_input_json": _path_join(result_dir, "stage3_test_generate_vq1_bucketed_parallel.json"),
        "watermark_model_name": "qwen2-vl",
        "watermark_python_bin": "",
        "watermark_torch_dtype": "bfloat16",
        "watermark_device": "cuda",
        "watermark_sample_fraction": "0.2",
    }


def _legacy_stage2_adapter_path(path: object) -> bool:
    text = str(path or "").replace("\\", "/")
    return text.endswith("/stage2_sub1_period7") or text.endswith("/stage2/stage2_sub1_period7")


def default_console_vars() -> dict[str, Any]:
    repo_path = _startup_repo_path()
    repo_root = str(repo_path) if repo_path is not None else REPO_DIR_NAME
    defaults: dict[str, Any] = {
        "server_ip": "127.0.0.1",
        "server_port": "8011",
        "python_bin": _default_python_bin(repo_root),
        "model_path": _default_model_path(repo_root),
        "dataset_name": "scienceqa",
        "cuda_devices": "0",
        "teacher_api_key": "",
        "teacher_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "victim_model": "qwen3.5-flash-2026-02-23",
        "sim_duration": 18,
        "auto_refresh": True,
        "judge_model": "gpt-5.5",
        "judge_api_base": "",
        "judge_api_key": "",
        "judge_sample_num": 500,
        "assistant_api_base": os.environ.get("ASSISTANT_API_BASE", "https://api.openai.com/v1"),
        "assistant_api_key": os.environ.get("ASSISTANT_API_KEY", ""),
        "assistant_model": os.environ.get("ASSISTANT_MODEL", "gpt-4o-mini"),
        "wm_base_before_score": "",
        "wm_extracted_score": "",
        "wm_test_score": "",
        "watermark_input_json": "",
        "watermark_model_name": "qwen2-vl",
        "watermark_python_bin": "",
        "watermark_torch_dtype": "bfloat16",
        "watermark_device": "cuda",
        "watermark_sample_fraction": "0.2",
        "watermark_sample_size": "0",
        "reuse_teacher_annotation": "0",
        "teacher_annotation_path": "",
        "reuse_teacher_baseline": "0",
        "teacher_baseline_path": "",
    }
    defaults.update(_console_path_defaults(repo_root))
    return defaults


def default_form_vars(console: dict[str, Any]) -> dict[str, str]:
    return {
        "SCIENCEQA_SPLIT": "train",
        "TRAIN_NUM": "0",
        "MAX_SAMPLES": "0",
        "SCIENCEQA_SEED": "20240306",
        "TEACHER_LANG": "en",
        "TEACHER_ENABLE_THINKING": "False",
        "COLLECT_TEACHER_DATA": "True",
        "STRICT_TEACHER_DISTILL": "True",
        "NUM_WORKERS": "4",
        "MAX_NEW_TOKENS": "64",
        "MAX_CONCURRENCY": "4",
        "SCIENCEQA_CONTROL_SPLIT": "test",
        "SCIENCEQA_CONTROL_MAX_SAMPLES": "0",
        "SCIENCEQA_CONTROLS": "baseline,text_only_blank,hint_ablation,option_shuffle,random_image_swap,image_blur,image_downsample",
        "TEACHER_RESULT_DIR": str(console.get("teacher_result_dir", "")),
        "STAGE1_CKPT_PATH": str(console.get("stage1_ckpt", "")),
        "STAGE2_FINAL_ADAPTER_PATH": str(console.get("stage2_adapter", "")),
        "STAGE1_EPOCHS": "3",
        "STAGE1_BATCH_SIZE": "1",
        "STAGE1_GRAD_ACCUM": "2",
        "STAGE1_LR": "3e-5",
        "STAGE1_MAX_LENGTH": "1536",
        "USE_4BIT": "False",
        "FREEZE_VISION_TOWER": "True",
        "LORA_RANK": "16",
        "LORA_ALPHA": "32",
        "STAGE1_FIELD_WEIGHT_REASONING": "2.0",
        "STAGE1_FIELD_WEIGHT_ANSWER": "12.0",
        "STAGE2_EPOCHS": "1",
        "PERIOD_NUM": "1",
        "STAGE2_GRAD_ACCUM": "2",
        "STAGE2_LR": "2e-5",
        "TAU1": "0.02",
        "STAGE2_MAX_LENGTH": "1024",
        "PHASE_A_BATCH_SIZE": "1",
        "PHASE_B_BATCH_SIZE": "1",
        "STAGE2_EVAL_EVERY_PERIOD": "1",
        "STAGE2_EVAL_TRAIN_NUM": "200",
        "STAGE2_EVAL_MAX_SAMPLES": "200",
        "EVAL_MAX_SAMPLES": "200",
        "STAGE2_WRONG_IMAGE_ENABLE": "True",
        "STAGE2_PAIR_USE_ANSWER_CORRECTNESS": "False",
        "ADAPTER_PATH": str(console.get("stage2_adapter", "")),
        "VQ_CODEBOOK_PATH": str(console.get("stage2_codebook", "")),
        "RESULT_DIR": str(console.get("result_dir", "")),
        "EVAL_MAX_NEW_TOKENS": "512",
        "STAGE2": str(console.get("reason_stage2_json", "")),
        "STAGE3": str(console.get("reason_stage3_json", "")),
        "TEACHER": "",
        "DATASET": str(console.get("reason_dataset_path", "")),
        "OUT_DIR": str(console.get("reason_out_dir", "")),
        "JUDGE_MODEL": str(console.get("judge_model", "gpt-5.5")),
        "JUDGE_API_BASE": "",
        "JUDGE_API_KEY": "",
        "SAMPLE_NUM": str(console.get("judge_sample_num", "500")),
        "JUDGE_DATASET_NAME": "scienceqa",
        "SPLIT": "test",
        "REQUIRE_VALID_FORMAT": "True",
    }


def normalize_openai_base_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)].rstrip("/")
    return text


def normalize_bool_text(value: object) -> str | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "1"
    if text in {"0", "false", "no", "n", "off"}:
        return "0"
    return None


def safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_metric(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


class WebConsole:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.lock = threading.Lock()
        self.clone_service = RemoteGitCloneService()
        self.config = load_config()
        self.console_vars = default_console_vars()
        self.console_vars.update({k: v for k, v in self.config.console_vars.items() if k in self.console_vars})
        self.form_vars = default_form_vars(self.console_vars)
        self.form_vars.update({str(k): str(v) for k, v in self.config.form_vars.items() if str(k) not in OBSOLETE_FORM_KEYS})
        self._migrate_legacy_stage2_paths_locked()
        self._refresh_derived_paths()
        self._sync_console_to_form_paths()
        self.connection_command = self.config.connection_command
        self.project_path = self.config.project_path
        self.ssh_password = ""
        self.real_pipeline_running = False
        self.real_pipeline_rows: list[dict[str, str]] | None = None
        self.real_pipeline_logs: list[str] = []
        self.local_real_log_path: Path | None = None
        self.real_pipeline_progress = 0.0
        self.pipeline_task: dict[str, Any] | None = None
        self.pipeline_started = False
        # debug only: step ids whose simulation has finished, so the dashboard /
        # watermark endpoints know which fabricated numbers may be revealed.
        self.sim_completed: set[str] = set()
        self.sim_archive_pending = False
        self.save_app_config()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "debug": self.debug,
                "connection_command": self.connection_command,
                "project_path": self.project_path,
                "console_vars": dict(self.console_vars),
                "form_vars": dict(self.form_vars),
                "pipeline": self.pipeline_status_locked(),
                "startup_repo": str(_startup_repo_path() or ""),
            }

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.connection_command = str(payload.get("connection_command", self.connection_command)).strip()
            self.project_path = str(payload.get("project_path", self.project_path)).strip()
            password = str(payload.get("password", ""))
            if password:
                self.ssh_password = password
            console = payload.get("console_vars")
            if isinstance(console, dict):
                for key, value in console.items():
                    if key in self.console_vars:
                        self.console_vars[str(key)] = value
            self._refresh_derived_paths()
            form = payload.get("form_vars")
            if isinstance(form, dict):
                for key, value in form.items():
                    key_text = str(key)
                    if key_text not in OBSOLETE_FORM_KEYS:
                        self.form_vars[key_text] = str(value)
            self._migrate_legacy_stage2_paths_locked()
            self._sync_console_to_form_paths()
            self.save_app_config_locked()
            return {"ok": True, "config": self.snapshot_locked()}

    def _migrate_legacy_stage2_paths_locked(self) -> None:
        root_dir = str(self.console_vars.get("root_dir", "")).strip()
        if not root_dir:
            return
        stage2_final = _path_join(root_dir, "vq_lord_ckpts", "stage2", "stage2_lord_final")
        if _legacy_stage2_adapter_path(self.console_vars.get("stage2_adapter")):
            self.console_vars["stage2_adapter"] = stage2_final
        for key in ("STAGE2_FINAL_ADAPTER_PATH", "ADAPTER_PATH"):
            if _legacy_stage2_adapter_path(self.form_vars.get(key)):
                self.form_vars[key] = stage2_final
        legacy_codebook = str(self.form_vars.get("VQ_CODEBOOK_PATH", "")).replace("\\", "/")
        if legacy_codebook.endswith("/stage2_sub1_period7/vq_codebook.pt") or legacy_codebook.endswith("/stage2/stage2_sub1_period7/vq_codebook.pt"):
            self.form_vars["VQ_CODEBOOK_PATH"] = _path_join(stage2_final, "vq_codebook.pt")

    def _refresh_derived_paths(self) -> None:
        result_dir = str(self.console_vars.get("result_dir", ""))
        stage2_adapter = str(self.console_vars.get("stage2_adapter", ""))
        if result_dir:
            self.console_vars["teacher_result_dir"] = _path_join(result_dir, "teacher_compare")
        if stage2_adapter:
            self.console_vars["stage2_codebook"] = _path_join(stage2_adapter, "vq_codebook.pt")
        if not str(self.console_vars.get("reason_stage2_json", "")).strip():
            self.console_vars["reason_stage2_json"] = _path_join(result_dir, "stage2_test_generate_readable.json")
        if not str(self.console_vars.get("reason_stage3_json", "")).strip():
            self.console_vars["reason_stage3_json"] = _path_join(result_dir, "stage3_test_generate_vq1_bucketed_parallel.json")
        if not str(self.console_vars.get("reason_dataset_path", "")).strip():
            self.console_vars["reason_dataset_path"] = str(self.console_vars.get("dataset_path", ""))
        if not str(self.console_vars.get("reason_out_dir", "")).strip():
            self.console_vars["reason_out_dir"] = "outputs/judge_latest"

    def _sync_console_to_form_paths(self) -> None:
        sync = {
            "TEACHER_RESULT_DIR": "teacher_result_dir",
            "STAGE1_CKPT_PATH": "stage1_ckpt",
            "STAGE2_FINAL_ADAPTER_PATH": "stage2_adapter",
            "ADAPTER_PATH": "stage2_adapter",
            "VQ_CODEBOOK_PATH": "stage2_codebook",
            "RESULT_DIR": "result_dir",
            "STAGE2": "reason_stage2_json",
            "STAGE3": "reason_stage3_json",
            "TEACHER": "reason_teacher_json",
            "DATASET": "reason_dataset_path",
            "OUT_DIR": "reason_out_dir",
            "JUDGE_MODEL": "judge_model",
            "SAMPLE_NUM": "judge_sample_num",
        }
        for form_key, console_key in sync.items():
            value = str(self.console_vars.get(console_key, ""))
            if value:
                self.form_vars[form_key] = value

    def save_app_config(self) -> None:
        with self.lock:
            self.save_app_config_locked()

    def save_app_config_locked(self) -> None:
        protected_form_keys = {
            "STAGE1_CKPT_PATH",
            "STAGE2_FINAL_ADAPTER_PATH",
            "ADAPTER_PATH",
            "VQ_CODEBOOK_PATH",
            "RESULT_DIR",
        }
        save_config(
            AppConfig(
                connection_command=self.connection_command,
                project_path=self.project_path,
                console_vars=dict(self.console_vars),
                form_vars={k: v for k, v in self.form_vars.items() if k not in OBSOLETE_FORM_KEYS and k not in protected_form_keys},
            )
        )

    def snapshot_locked(self) -> dict[str, Any]:
        return {
            "debug": self.debug,
            "connection_command": self.connection_command,
            "project_path": self.project_path,
            "console_vars": dict(self.console_vars),
            "form_vars": dict(self.form_vars),
            "pipeline": self.pipeline_status_locked(),
            "startup_repo": str(_startup_repo_path() or ""),
        }

    def should_run_remote(self) -> bool:
        return bool(self.connection_command.strip())

    def clone_repository(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.update_config(payload)
        password = str(payload.get("password", "")) or self.ssh_password
        request = CloneRequest(
            connection_command=self.connection_command,
            project_path=self.project_path,
            password=password,
        )
        result = self.clone_service.clone_repository(request)
        if result.success:
            with self.lock:
                if password:
                    self.ssh_password = password
                if result.details:
                    self.console_vars.update(_console_path_defaults(result.details))
                    self._sync_console_to_form_paths()
                    self.save_app_config_locked()
            if result.details:
                self.load_remote_config(result.details)
        return {"success": result.success, "message": result.message, "details": result.details, "config": self.snapshot()}

    def remote_config_path(self, repo_path: str | None = None) -> str:
        root_dir = (repo_path or str(self.console_vars.get("root_dir", ""))).strip()
        return _path_join(root_dir, ".remote-console-config.json")

    def run_remote_config_command(self, command: str):
        target = parse_connection_command(self.connection_command)
        with SSHClientManager(target, self.ssh_password) as ssh:
            return ssh.run(command)

    def load_remote_config(self, repo_path: str | None = None) -> dict[str, Any]:
        if not self.should_run_remote():
            return {"ok": False, "message": "no ssh command configured"}
        config_path = self.remote_config_path(repo_path)
        command = f"if [ -f {shlex.quote(config_path)} ]; then cat {shlex.quote(config_path)}; else echo __REMOTE_CONFIG_MISSING__; fi"
        try:
            result = self.run_remote_config_command(command)
        except Exception as exc:
            self.append_log(f"[config] failed to load remote config: {exc}")
            return {"ok": False, "message": str(exc)}
        if result.exit_code != 0:
            message = result.stderr.strip() or result.stdout.strip()
            self.append_log(f"[config] failed to load remote config: {message}")
            return {"ok": False, "message": message}
        if "__REMOTE_CONFIG_MISSING__" in result.stdout:
            self.append_log(f"[config] remote config missing, uploading current config to {config_path}")
            return self.save_remote_config(repo_path)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            self.append_log(f"[config] remote config is invalid JSON: {exc}")
            return {"ok": False, "message": str(exc)}
        with self.lock:
            console = payload.get("console_vars", {})
            if isinstance(console, dict):
                for key, value in console.items():
                    if key in self.console_vars:
                        self.console_vars[str(key)] = value
            form = payload.get("form_vars", {})
            if isinstance(form, dict):
                for key, value in form.items():
                    if str(key) not in OBSOLETE_FORM_KEYS:
                        self.form_vars[str(key)] = str(value)
            self._migrate_legacy_stage2_paths_locked()
            self._refresh_derived_paths()
            self._sync_console_to_form_paths()
            self.save_app_config_locked()
        self.append_log("[config] loaded remote frontend config")
        return {"ok": True, "config": self.snapshot()}

    def save_remote_config(self, repo_path: str | None = None) -> dict[str, Any]:
        if not self.should_run_remote():
            return {"ok": False, "message": "no ssh command configured"}
        root_dir = (repo_path or str(self.console_vars.get("root_dir", ""))).strip()
        if not root_dir:
            return {"ok": False, "message": "ROOT_DIR is empty"}
        config_path = self.remote_config_path(root_dir)
        payload = json.dumps(
            {
                "connection_command": self.connection_command,
                "project_path": self.project_path,
                "console_vars": self.console_vars,
                "form_vars": {k: v for k, v in self.form_vars.items() if k not in OBSOLETE_FORM_KEYS},
            },
            ensure_ascii=False,
            indent=2,
        )
        command = f"mkdir -p {shlex.quote(root_dir)} && printf %s {shlex.quote(payload)} > {shlex.quote(config_path)}"
        try:
            result = self.run_remote_config_command(command)
        except Exception as exc:
            self.append_log(f"[config] failed to save remote config: {exc}")
            return {"ok": False, "message": str(exc)}
        if result.exit_code != 0:
            message = result.stderr.strip() or result.stdout.strip()
            self.append_log(f"[config] failed to save remote config: {message}")
            return {"ok": False, "message": message}
        self.append_log(f"[config] saved remote frontend config: {config_path}")
        return {"ok": True, "message": config_path}

    def task_definitions(self) -> dict[str, tuple[str, str]]:
        definitions = {step_id: (label, script) for step_id, label, script in PIPELINE_STEPS}
        definitions["watermark_detect"] = ("水印失效风险评估", "VLA-mark/detect_vq_lord_result_watermark.py")
        return definitions

    def task_env(self, step_id: str | None = None) -> dict[str, str]:
        with self.lock:
            console = dict(self.console_vars)
            form = dict(self.form_vars)
        root_dir = str(console.get("root_dir", "")).strip()
        dataset_name = str(console.get("dataset_name", "") or "scienceqa").strip()
        dataset_path = str(console.get("dataset_path", "")).strip()
        teacher_key = str(console.get("teacher_api_key", "")).strip()
        teacher_base = normalize_openai_base_url(str(console.get("teacher_api_base", "")))
        judge_key = str(form.get("JUDGE_API_KEY", "") or console.get("judge_api_key", "") or teacher_key).strip()
        judge_base = normalize_openai_base_url(
            str(form.get("JUDGE_API_BASE", "") or console.get("judge_api_base", "") or teacher_base)
        )
        env = {
            "PYTHONUNBUFFERED": "1",
            # scripts2/common.sh exports PYTHONPATH inside align_vq_setup_env(), but the
            # teacher/fast eval pipelines never call it -> `import vq_lord3` fails there.
            # Passing it explicitly fixes every script regardless of remote script version.
            "PYTHONPATH": root_dir,
            "ROOT_DIR": root_dir,
            "PYTHON_BIN": str(console.get("python_bin", "")).strip(),
            "MODEL_PATH": str(console.get("model_path", "")).strip(),
            "DATASET_NAME": dataset_name,
            "DATASET_TAG": dataset_name,
            "TRAIN_DATASET_NAME": dataset_name,
            "DATASET_PATH": dataset_path,
            "SCIENCEQA_PATH": dataset_path,
            "CUDA_VISIBLE_DEVICES": str(console.get("cuda_devices", "")).strip(),
            "TEACHER_API_KEY": teacher_key,
            "TEACHER_API_BASE": teacher_base,
            "OPENAI_API_KEY": judge_key,
            "OPENAI_BASE_URL": judge_base,
            "JUDGE_API_KEY": judge_key,
            "JUDGE_API_BASE": judge_base,
            "VICTIM_MODEL": str(console.get("victim_model", "")).strip(),
            "STAGE1_CKPT_PATH": str(console.get("stage1_ckpt", "")).strip(),
            "STAGE2_FINAL_ADAPTER_PATH": str(console.get("stage2_adapter", "")).strip(),
            "ADAPTER_PATH": str(console.get("stage2_adapter", "")).strip(),
            "VQ_CODEBOOK_PATH": str(console.get("stage2_codebook", "")).strip(),
            "RESULT_DIR": str(console.get("result_dir", "")).strip(),
            "JUDGE_MODEL": str(console.get("judge_model", "")).strip(),
            "SAMPLE_NUM": str(console.get("judge_sample_num", "")).strip(),
            "STAGE2": str(console.get("reason_stage2_json", "")).strip(),
            "STAGE3": str(console.get("reason_stage3_json", "")).strip(),
            "TEACHER": str(console.get("reason_teacher_json", "")).strip(),
            "DATASET": str(console.get("reason_dataset_path", "")).strip(),
            "OUT_DIR": str(console.get("reason_out_dir", "")).strip(),
            "REQUIRE_VALID_FORMAT": "1",
        }
        annotation_path = str(console.get("teacher_annotation_path", "")).strip()
        reuse_annotation = normalize_bool_text(console.get("reuse_teacher_annotation", "0")) == "1"
        if reuse_annotation and annotation_path:
            # Stage1/Stage2 need the selected annotation cache after collection is skipped.
            env["TEACHER_CACHE_PATH"] = annotation_path
        protected = {
            "PYTHONPATH",
            "ROOT_DIR",
            "PYTHON_BIN",
            "MODEL_PATH",
            "DATASET_NAME",
            "DATASET_PATH",
            "SCIENCEQA_PATH",
            "CUDA_VISIBLE_DEVICES",
            "TEACHER_API_KEY",
            "TEACHER_API_BASE",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "STAGE1_CKPT_PATH",
            "STAGE2_FINAL_ADAPTER_PATH",
            "ADAPTER_PATH",
            "VQ_CODEBOOK_PATH",
            "RESULT_DIR",
        }
        for key, value in form.items():
            if str(value).strip() and key not in protected and key not in OBSOLETE_FORM_KEYS:
                env[key] = str(value).strip()
        for key in BOOL_ENV_KEYS:
            if key in env:
                normalized = normalize_bool_text(env[key])
                if normalized is not None:
                    env[key] = normalized
        if step_id == "stage1_train":
            self._map_form_env(env, "STAGE1_EPOCHS", "EPOCHS", form)
            self._map_form_env(env, "STAGE1_BATCH_SIZE", "BATCH_SIZE", form)
            self._map_form_env(env, "STAGE1_LR", "LR", form)
            self._map_form_env(env, "STAGE1_MAX_LENGTH", "MAX_LENGTH", form)
        elif step_id == "stage2_train":
            self._map_form_env(env, "STAGE2_EPOCHS", "EPOCHS", form)
            self._map_form_env(env, "STAGE2_LR", "LR", form)
            self._map_form_env(env, "STAGE2_MAX_LENGTH", "MAX_LENGTH", form)
        elif step_id == "student_eval":
            self._map_form_env(env, "EVAL_MAX_NEW_TOKENS", "MAX_NEW_TOKENS", form)
        elif step_id == "reason_judge" and form.get("JUDGE_DATASET_NAME"):
            env["DATASET_NAME"] = form["JUDGE_DATASET_NAME"]
        return {key: value for key, value in env.items() if value}

    def _map_form_env(self, env: dict[str, str], source: str, target: str, form: dict[str, str]) -> None:
        value = env.get(source) or form.get(source)
        if value:
            env[target] = value

    def resolve_task_command(self, step_id: str) -> tuple[list[str] | None, str, str]:
        definitions = self.task_definitions()
        label, script = definitions[step_id]
        with self.lock:
            console = dict(self.console_vars)
            form = dict(self.form_vars)
        root_dir = str(console.get("root_dir", "")).strip()
        if script == "dashboard aggregation":
            return None, root_dir, label
        if step_id == "origin_eval":
            python_bin = str(console.get("python_bin", "")).strip() or sys.executable
            result_dir = str(console.get("result_dir", "")).strip()
            control_path = _path_join(result_dir, "origin_scienceqa_control_suite_full_fast.json")
            report_path = _path_join(result_dir, "origin_mm_eval_suite_report_full_fast.json")
            controls = str(form.get("SCIENCEQA_CONTROLS", "")).strip() or "baseline,text_only_blank,hint_ablation,option_shuffle,random_image_swap,image_blur,image_downsample"
            max_samples = str(form.get("SCIENCEQA_CONTROL_MAX_SAMPLES", "0")).strip() or "0"
            split = str(form.get("SCIENCEQA_CONTROL_SPLIT", "test")).strip() or "test"
            evaluate = [
                python_bin,
                "-u",
                _path_join(root_dir, "vq_lord3", "evaluation", "final", "mm_scienceqa_control_eval_fast.py"),
                "--model_path",
                str(console.get("model_path", "")).strip(),
                "--adapter_path",
                "",
                "--student_model_type",
                str(form.get("STUDENT_MODEL_TYPE", "qwen2_vl")).strip() or "qwen2_vl",
                "--use_4bit",
                "0",
                "--use_vq",
                "0",
                "--vq_codebook_size",
                "1024",
                "--vq_codebook_path",
                "",
                "--scienceqa_path",
                str(console.get("dataset_path", "")).strip(),
                "--split",
                split,
                "--max_samples",
                max_samples,
                "--controls",
                controls,
                "--prompt_style",
                "legacy",
                "--answer_mode",
                "logits",
                "--max_new_tokens",
                "64",
                "--save_path",
                control_path,
            ]
            aggregate = [
                python_bin,
                "-u",
                _path_join(root_dir, "vq_lord3", "evaluation", "final", "mm_eval_suite_report.py"),
                "--control_result",
                control_path,
                "--save_json",
                report_path,
                "--save_md",
                _path_join(result_dir, "origin_mm_eval_suite_report_full_fast.md"),
            ]
            shell_command = f"{shlex.join(evaluate)} && {shlex.join(aggregate)}"
            return ["bash", "-lc", shell_command], root_dir, label
        if step_id == "reason_judge":
            cwd = str(console.get("reason_judge_dir", "")).strip()
            return ["bash", _path_join(cwd, "run_judge.sh")], cwd, label
        if step_id == "watermark_detect":
            cwd = str(console.get("vla_mark_dir", "")).strip()
            output_path = self.watermark_output_path()
            python_bin = str(console.get("watermark_python_bin", "")).strip() or str(console.get("python_bin", "")) or sys.executable
            result_path = (
                str(console.get("watermark_input_json", "")).strip()
                or os.environ.get("WATERMARK_RESULT_PATH")
                or _path_join(str(console.get("result_dir", "")), "stage3_test_generate_vq1_bucketed_parallel.json")
            )
            sample_size = str(console.get("watermark_sample_size", "0")).strip() or "0"
            sample_fraction = str(console.get("watermark_sample_fraction", "0.2")).strip() or "0.2"
            model_name = str(console.get("watermark_model_name", "qwen2-vl")).strip() or "qwen2-vl"
            torch_dtype = str(console.get("watermark_torch_dtype", "bfloat16")).strip() or "bfloat16"
            device = str(console.get("watermark_device", "cuda")).strip() or "cuda"
            command = [
                python_bin,
                "-u",
                _path_join(cwd, "detect_vq_lord_result_watermark.py"),
                "--result_path",
                result_path,
                "--scienceqa_path",
                str(console.get("dataset_path", "")),
                "--output_path",
                output_path,
                "--model_path",
                str(console.get("model_path", "")),
                "--model_name",
                model_name,
                "--sample_size",
                sample_size,
                "--sample_fraction",
                sample_fraction,
                "--torch_dtype",
                torch_dtype,
                "--device",
                device,
            ]
            return command, cwd, label
        return ["bash", _path_join(root_dir, script)], root_dir, label

    def reuse_stage_source(self, step_id: str) -> str | None:
        with self.lock:
            console = dict(self.console_vars)
        settings = {
            "teacher_collect": ("reuse_teacher_annotation", "teacher_annotation_path"),
            "teacher_eval": ("reuse_teacher_baseline", "teacher_baseline_path"),
        }
        setting = settings.get(step_id)
        if setting is None:
            return None
        enabled_key, path_key = setting
        if normalize_bool_text(console.get(enabled_key, "0")) != "1":
            return None
        return str(console.get(path_key, "")).strip()

    def remote_or_local_file_exists(self, path: str) -> bool:
        if not path:
            return False
        if self.should_run_remote():
            result = self.run_remote_config_command(f"test -f {shlex.quote(path)}")
            return result.exit_code == 0
        return Path(path).is_file()

    def prepare_reused_stage(self, step_id: str, source_path: str) -> bool:
        if not source_path:
            self.append_log("[reuse] 已启用复用，但未填写 JSON 路径")
            return False
        try:
            exists = self.remote_or_local_file_exists(source_path)
        except Exception as exc:
            self.append_log(f"[reuse] 检查文件失败: {exc}")
            return False
        if not exists:
            self.append_log(f"[reuse] 文件不存在，无法跳过阶段: {source_path}")
            return False

        if step_id == "teacher_eval":
            with self.lock:
                result_dir = str(self.console_vars.get("result_dir", "")).strip()
            target = _path_join(result_dir, "scienceqa_control_suite_teacher_full.json")
            same_path = posixpath.normpath(source_path) == posixpath.normpath(target)
            if not same_path and self.should_run_remote():
                command = (
                    f"mkdir -p {shlex.quote(_path_parent(target))} && "
                    f"cp {shlex.quote(source_path)} {shlex.quote(target)}"
                )
                try:
                    result = self.run_remote_config_command(command)
                except Exception as exc:
                    self.append_log(f"[reuse] 教师基线 JSON 复制失败: {exc}")
                    return False
                if result.exit_code != 0:
                    message = result.stderr.strip() or result.stdout.strip()
                    self.append_log(f"[reuse] 教师基线 JSON 复制失败: {message}")
                    return False
            elif not same_path:
                try:
                    source = Path(source_path).resolve()
                    destination = Path(target).resolve()
                    if source != destination:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(source.read_bytes())
                except OSError as exc:
                    self.append_log(f"[reuse] 教师基线 JSON 复制失败: {exc}")
                    return False

        self.append_log(f"[reuse] 使用已有文件跳过阶段: {source_path}")
        return True

    def start_tasks(self, step_ids: list[str]) -> dict[str, Any]:
        with self.lock:
            if self.real_pipeline_running:
                return {"ok": False, "message": "已有任务正在运行"}
            self.save_app_config_locked()
            if self.debug:
                self.start_simulation_locked(step_ids)
                return {"ok": True, "pipeline": self.pipeline_status_locked()}
            self.local_real_log_path = self.create_local_real_log_file()
            definitions = self.task_definitions()
            self.real_pipeline_rows = [
                {"id": step_id, "stage": definitions[step_id][0], "status": "pending", "progress": "0%", "script": definitions[step_id][1]}
                for step_id in step_ids
            ]
            self.real_pipeline_logs = ["[real] starting task runner"]
            if self.local_real_log_path is not None:
                self.real_pipeline_logs.append(f"[local-log] {self.local_real_log_path}")
            self.real_pipeline_progress = 0.0
            self.real_pipeline_running = True
        thread = threading.Thread(target=self.run_real_tasks, args=(step_ids,), daemon=True)
        thread.start()
        return {"ok": True, "pipeline": self.pipeline_status()}

    def start_simulation_locked(self, step_ids: list[str]) -> None:
        per_step_seconds = max(1, int(float(self.console_vars.get("sim_duration", 18) or 18)))
        self.pipeline_task = {
            "started_at": time.time(),
            "duration": float(per_step_seconds * max(1, len(step_ids))),
            "per_step": float(per_step_seconds),
            "step_ids": step_ids,
        }
        self.pipeline_started = True

    def create_local_real_log_file(self) -> Path | None:
        try:
            log_dir = APP_DIR / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"real_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
            path.write_text("[real] starting task runner\n", encoding="utf-8")
            return path
        except OSError:
            return None

    def run_real_tasks(self, step_ids: list[str]) -> None:
        total = max(1, len(step_ids))
        if self.should_run_remote():
            self.save_remote_config()
        for index, step_id in enumerate(step_ids):
            self.set_real_step_state(step_id, "running", "0%")
            reuse_source = self.reuse_stage_source(step_id)
            if reuse_source is not None:
                if not self.prepare_reused_stage(step_id, reuse_source):
                    self.set_real_step_state(step_id, "failed", "0%")
                    break
                self.set_real_step_state(step_id, "success", "100%")
                with self.lock:
                    self.real_pipeline_progress = (index + 1) / total
                continue
            env = self.task_env(step_id)
            command, cwd, label = self.resolve_task_command(step_id)
            if command is None:
                try:
                    if step_id == "risk_report":
                        self.save_risk_report()
                except Exception as exc:
                    self.append_log(f"[real] {label} failed: {exc}")
                    self.set_real_step_state(step_id, "failed", "0%")
                    break
                self.append_log(f"[real] {label}: aggregation completed")
                self.set_real_step_state(step_id, "success", "100%")
                with self.lock:
                    self.real_pipeline_progress = (index + 1) / total
                continue
            self.append_log(f"[real] start {label}: {' '.join(command)}")
            try:
                if self.should_run_remote():
                    exit_code = self.run_remote_command(command, cwd, env)
                else:
                    exit_code = self.run_local_command(command, cwd, env)
            except Exception as exc:
                self.append_log(f"[real] failed to start {label}: {exc}")
                self.set_real_step_state(step_id, "failed", "0%")
                break
            if exit_code != 0:
                self.append_log(f"[real] {label} failed with exit code {exit_code}")
                self.set_real_step_state(step_id, "failed", "0%")
                break
            self.set_real_step_state(step_id, "success", "100%")
            with self.lock:
                self.real_pipeline_progress = (index + 1) / total
        else:
            self.append_log("[real] all tasks completed")
        with self.lock:
            self.real_pipeline_running = False
        self.save_history(only_if_results=True)

    def remote_shell_command(self, command: list[str], cwd: str, env: dict[str, str]) -> str:
        assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items() if value)
        command_text = " ".join(shlex.quote(part) for part in command)
        if assignments:
            command_text = f"env {assignments} {command_text}"
        return f"cd {shlex.quote(cwd)} && {command_text}"

    def masked_env_for_log(self, env: dict[str, str]) -> dict[str, str]:
        secret_names = {"TOKEN", "ACCESS_TOKEN", "AUTH_TOKEN", "PASSWORD", "PASSWD", "SECRET"}
        masked = {}
        for key, value in env.items():
            key_upper = key.upper()
            is_secret = (
                key_upper in secret_names
                or key_upper.endswith("_API_KEY")
                or key_upper.endswith("_KEY")
                or key_upper.endswith("_SECRET")
                or key_upper.endswith("_PASSWORD")
                or key_upper.endswith("_TOKEN")
            )
            masked[key] = "***" if is_secret else value
        return masked

    def run_remote_command(self, command: list[str], cwd: str, env: dict[str, str]) -> int:
        target = parse_connection_command(self.connection_command)
        remote_command = self.remote_shell_command(command, cwd, env)
        display_command = self.remote_shell_command(command, cwd, self.masked_env_for_log(env))
        self.append_log(f"[ssh] {target.username}@{target.hostname}:{target.port} $ {display_command}")
        with SSHClientManager(target, self.ssh_password) as ssh:
            if ssh.client is None:
                raise RuntimeError("SSH client is not connected")
            _, stdout, stderr = ssh.client.exec_command(remote_command, get_pty=True)
            channel = stdout.channel
            while not channel.exit_status_ready():
                self.drain_ssh_stream(stdout)
                self.drain_ssh_stream(stderr)
                time.sleep(0.1)
            self.drain_ssh_stream(stdout)
            self.drain_ssh_stream(stderr)
            return channel.recv_exit_status()

    def drain_ssh_stream(self, stream: Any) -> None:
        channel = stream.channel
        while channel.recv_ready():
            self.append_log_chunk(channel.recv(4096).decode("utf-8", errors="replace"))
        while channel.recv_stderr_ready():
            self.append_log_chunk(channel.recv_stderr(4096).decode("utf-8", errors="replace"))

    def run_local_command(self, command: list[str], cwd: str, env: dict[str, str]) -> int:
        local_env = os.environ.copy()
        inherited_pythonpath = local_env.get("PYTHONPATH", "")
        local_env.update(env)
        # prepend rather than clobber, so a locally exported PYTHONPATH survives
        if env.get("PYTHONPATH") and inherited_pythonpath:
            local_env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + inherited_pythonpath
        process = subprocess.Popen(
            command,
            cwd=cwd or None,
            env=local_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            self.append_log(line.rstrip())
        return process.wait()

    def set_real_step_state(self, step_id: str, status: str, progress: str) -> None:
        with self.lock:
            if self.real_pipeline_rows is None:
                return
            for row in self.real_pipeline_rows:
                if row.get("id") == step_id:
                    row["status"] = status
                    row["progress"] = progress
                    break

    def append_log_chunk(self, chunk: str) -> None:
        for line in chunk.splitlines():
            self.append_log(line)

    def append_log(self, line: str) -> None:
        with self.lock:
            self.real_pipeline_logs.append(line)
            self.real_pipeline_logs = self.real_pipeline_logs[-1200:]
            path = self.local_real_log_path
        if path is not None:
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def pipeline_status(self) -> dict[str, Any]:
        with self.lock:
            status = self.pipeline_status_locked()
            archive = self.sim_archive_pending
            self.sim_archive_pending = False
        if archive:                       # debug sim finished: archive outside the lock
            self.save_history(only_if_results=True)
        return status

    def pipeline_status_locked(self) -> dict[str, Any]:
        if self.debug:
            return self.sim_pipeline_status_locked()
        rows = self.real_pipeline_rows
        if rows is None:
            rows = [{"id": step_id, "stage": label, "status": "pending", "progress": "0%", "script": script} for step_id, label, script in PIPELINE_STEPS]
        summary = "暂无真实任务"
        if self.real_pipeline_running:
            running = next((row["stage"] for row in rows if row["status"] == "running"), "准备中")
            summary = f"真实任务运行中：{self.real_pipeline_progress * 100:.0f}% · 当前阶段：{running}"
        elif self.real_pipeline_rows is not None and any(row["status"] == "failed" for row in rows):
            failed = next((row["stage"] for row in rows if row["status"] == "failed"), "未知阶段")
            summary = f"真实任务失败：{failed}"
        elif self.real_pipeline_rows is not None:
            summary = f"真实任务完成：{self.real_pipeline_progress * 100:.0f}%"
        return {
            "debug": False,
            "running": self.real_pipeline_running,
            "progress": self.real_pipeline_progress,
            "summary": summary,
            "rows": rows,
            "logs": self.real_pipeline_logs or ["No real tasks yet."],
            "local_log_path": str(self.local_real_log_path or ""),
        }

    def sim_pipeline_status_locked(self) -> dict[str, Any]:
        definitions = self.task_definitions()
        step_ids = [step_id for step_id, _, _ in PIPELINE_STEPS]
        if self.pipeline_task is None:
            rows = [{"id": step_id, "stage": definitions[step_id][0], "status": "pending", "progress": "0%", "script": definitions[step_id][1]} for step_id in step_ids]
            return {
                "debug": True,
                "running": False,
                "progress": 0.0,
                "summary": "暂无仿真任务",
                "rows": rows,
                "logs": ["No simulated tasks yet."],
                "local_log_path": "",
            }
        step_ids = list(self.pipeline_task.get("step_ids") or step_ids)
        elapsed = max(0.0, time.time() - float(self.pipeline_task["started_at"]))
        duration = max(0.001, float(self.pipeline_task["duration"]))
        per_step = max(0.001, float(self.pipeline_task["per_step"]))
        total_progress = min(1.0, elapsed / duration)
        rows = []
        for index, step_id in enumerate(step_ids):
            label, script = definitions[step_id]
            local_elapsed = elapsed - index * per_step
            if local_elapsed <= 0:
                status = "pending"
                progress = 0.0
            elif local_elapsed >= per_step:
                status = "success"
                progress = 1.0
            else:
                status = "running"
                progress = min(1.0, local_elapsed / per_step)
            if status == "success":
                self.sim_completed.add(step_id)
            rows.append({"id": step_id, "stage": label, "status": status, "progress": f"{progress * 100:.0f}%", "script": script})
        running = total_progress < 1.0
        if not running and not self.pipeline_task.get("archived"):
            self.pipeline_task["archived"] = True
            self.sim_archive_pending = True
        summary = "完整 Pipeline 仿真完成：100%" if not running else f"完整 Pipeline 仿真运行中：{total_progress * 100:.0f}%"
        logs = ["[simulate] mode=sequential full pipeline", f"[simulate] progress={total_progress * 100:.0f}%"]
        logs.extend(f"[simulate] {row['status']:>7} {row['progress']:>4} :: {row['script']}" for row in rows)
        if not running:
            logs.append("[simulate] completed successfully")
        return {
            "debug": True,
            "running": running,
            "progress": total_progress,
            "summary": summary,
            "rows": rows,
            "logs": logs,
            "local_log_path": "",
        }

    # ---------------- history archive (stored next to the frontend, in APP_DIR) ----
    def history_new_id(self) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        candidate = stamp
        counter = 2
        while (HISTORY_DIR / f"{candidate}.json").exists():
            candidate = f"{stamp}_{counter}"
            counter += 1
        return candidate

    def build_history_record(self, label: str = "") -> dict[str, Any]:
        dashboard = self.dashboard_payload()
        watermark = self.watermark_payload()
        pipeline = self.pipeline_status()
        with self.lock:
            console, form = _strip_secrets(dict(self.console_vars), dict(self.form_vars))
        rows = pipeline.get("rows") or []
        stages_ok = sum(1 for row in rows if row.get("status") == "success")
        return {
            "id": "",
            "label": label,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "debug": self.debug,
            "stages_ok": stages_ok,
            "stages_total": len(rows),
            "console_vars": console,
            "form_vars": form,
            "dashboard": dashboard,
            "watermark": watermark,
            "pipeline": {
                "summary": pipeline.get("summary", ""),
                "progress": pipeline.get("progress", 0.0),
                "rows": rows,
                "logs": (pipeline.get("logs") or [])[-HISTORY_LOG_LINES:],
            },
        }

    def save_history(self, label: str = "", only_if_results: bool = False) -> dict[str, Any]:
        record = self.build_history_record(label)
        if only_if_results and not _history_has_results(record["dashboard"], record["watermark"]):
            return {"ok": False, "message": "本次运行没有可归档的结果"}
        if not record["label"]:
            prefix = "[仿真] " if self.debug else ""
            record["label"] = f"{prefix}{record['created_at']} · {record['stages_ok']}/{record['stages_total']} 阶段完成"
        try:
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            record["id"] = self.history_new_id()
            path = HISTORY_DIR / f"{record['id']}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "message": f"归档失败: {exc}"}
        self.append_log(f"[history] saved {path}")
        return {"ok": True, "message": f"已归档：{record['label']}", "id": record["id"], "items": self.history_list()}

    def history_list(self) -> list[dict[str, Any]]:
        items = []
        try:
            paths = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
        except OSError:
            return []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("dashboard", {}).get("metrics", {}) if isinstance(payload.get("dashboard"), dict) else {}
            items.append({
                "id": payload.get("id") or path.stem,
                "label": payload.get("label") or path.stem,
                "created_at": payload.get("created_at", ""),
                "debug": bool(payload.get("debug")),
                "stages_ok": payload.get("stages_ok", 0),
                "stages_total": payload.get("stages_total", 0),
                "risk": metrics.get("risk", "-"),
            })
        return items

    def history_get(self, history_id: str) -> dict[str, Any]:
        path = HISTORY_DIR / f"{Path(history_id).name}.json"
        if not path.is_file():
            return {"ok": False, "message": "找不到该历史记录"}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "message": f"读取失败: {exc}"}
        if not isinstance(payload, dict):
            return {"ok": False, "message": "历史记录格式不正确"}
        payload["ok"] = True
        return payload

    def history_delete(self, history_id: str) -> dict[str, Any]:
        path = HISTORY_DIR / f"{Path(history_id).name}.json"
        if not path.is_file():
            return {"ok": False, "message": "找不到该历史记录", "items": self.history_list()}
        try:
            path.unlink()
        except OSError as exc:
            return {"ok": False, "message": f"删除失败: {exc}", "items": self.history_list()}
        return {"ok": True, "message": "已删除该历史记录", "items": self.history_list()}

    def assistant_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            console = dict(self.console_vars)
            form = dict(self.form_vars)
        base_url = normalize_openai_base_url(str(console.get("assistant_api_base", "")))
        api_key = str(console.get("assistant_api_key", "")).strip()
        model = str(console.get("assistant_model", "gpt-4o-mini")).strip() or "gpt-4o-mini"
        if not base_url:
            return {"ok": False, "message": "尚未配置 AI 助手 Base URL"}
        if not api_key:
            return {"ok": False, "message": "尚未配置 AI 助手 API Key"}
        raw_messages = payload.get("messages", []) if isinstance(payload, dict) else []
        messages: list[dict[str, str]] = []
        if isinstance(raw_messages, list):
            for item in raw_messages[-12:]:
                if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
                    continue
                content = str(item.get("content", "")).strip()
                if content:
                    messages.append({"role": str(item["role"]), "content": content[:4000]})
        if not messages or messages[-1]["role"] != "user":
            return {"ok": False, "message": "请输入问题后再发送"}
        context = {
            "dataset_name": console.get("dataset_name"),
            "dataset_path": console.get("dataset_path"),
            "model_path": console.get("model_path"),
            "cuda_devices": console.get("cuda_devices"),
            "teacher_model": console.get("victim_model"),
            "train_num": form.get("TRAIN_NUM"),
            "max_samples": form.get("MAX_SAMPLES"),
            "control_max_samples": form.get("SCIENCEQA_CONTROL_MAX_SAMPLES"),
            "eval_max_samples": form.get("EVAL_MAX_SAMPLES"),
            "sample_num": console.get("judge_sample_num"),
            "stage1_batch_size": form.get("STAGE1_BATCH_SIZE"),
            "stage1_grad_accum": form.get("STAGE1_GRAD_ACCUM"),
            "stage2_batch_size": form.get("PHASE_A_BATCH_SIZE"),
            "stage2_grad_accum": form.get("STAGE2_GRAD_ACCUM"),
            "use_4bit": form.get("USE_4BIT"),
            "freeze_vision_tower": form.get("FREEZE_VISION_TOWER"),
        }
        system = ASSISTANT_SYSTEM_PROMPT + "\n\n当前前端配置上下文（仅用于解释，不要在回答中泄露 API Key）：\n" + json.dumps(context, ensure_ascii=False)
        request_payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.2,
            "max_tokens": 900,
            "reasoning": {"enabled": False},
        }
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
            return {"ok": False, "message": f"AI 助手请求失败（HTTP {exc.code}）：{detail}"}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"ok": False, "message": f"AI 助手连接失败：{exc}"}
        try:
            message = result["choices"][0]["message"].get("content", "")
            if isinstance(message, list):
                message = "".join(str(part.get("text", "")) for part in message if isinstance(part, dict))
            message = str(message).strip()
            message = re.sub(r"<think(?:ing)?[^>]*>.*?</think(?:ing)?>", "", message, flags=re.IGNORECASE | re.DOTALL).strip()
            if re.search(r"thinking\s+process\s*:", message, flags=re.IGNORECASE):
                final = re.split(r"(?:final\s+answer|最终答案|答案)\s*:\s*", message, maxsplit=1, flags=re.IGNORECASE)
                message = final[-1].strip() if len(final) > 1 else "模型返回了未隐藏的思考内容，请关闭模型的 thinking/reasoning 选项后重试。"
        except (KeyError, IndexError, TypeError):
            return {"ok": False, "message": "AI 助手返回格式无法识别"}
        return {"ok": True, "message": message or "模型没有返回文本内容"}

    def sim_sync_completed(self) -> set[str]:
        """Refresh (and return) the set of finished simulated steps."""
        with self.lock:
            if self.pipeline_task is not None:
                self.sim_pipeline_status_locked()   # side effect: fills self.sim_completed
            return set(self.sim_completed)

    def resolve_reason_out_dir(self) -> str:
        out_dir = str(self.console_vars.get("reason_out_dir", "")).strip() or "outputs/judge_latest"
        if out_dir.startswith("/"):
            return out_dir
        return _path_join(str(self.console_vars.get("reason_judge_dir", "")), out_dir)

    def watermark_output_path(self) -> str:
        with self.lock:
            cwd = str(self.console_vars.get("vla_mark_dir", "")).strip()
        return _path_join(cwd, "outputs", "watermark_detect_vqlord.json")

    def watermark_payload(self) -> dict[str, Any]:
        with self.lock:
            input_json = str(self.console_vars.get("watermark_input_json", "")).strip()
        output_path = self.watermark_output_path()
        if self.debug:
            done = "watermark_detect" in self.sim_sync_completed()
            report: dict[str, Any] = dict(SIM_WATERMARK_REPORT) if done else {}
        else:
            report = self.read_json_result(output_path)
        metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
        erosion = self.watermark_erosion_payload()
        return {
            "ok": bool(metrics),
            "input_json": input_json,
            "output_path": output_path,
            "num_scored": report.get("num_scored") if isinstance(report, dict) else None,
            "num_total": report.get("num_total_records") if isinstance(report, dict) else None,
            "split": report.get("split") if isinstance(report, dict) else None,
            "metrics": {
                "mean_z": fmt_metric(safe_float(metrics.get("mean_z"))),
                "median_z": fmt_metric(safe_float(metrics.get("median_z"))),
                "min_z": fmt_metric(safe_float(metrics.get("min_z"))),
                "max_z": fmt_metric(safe_float(metrics.get("max_z"))),
                "threshold_4_rate": fmt_metric(safe_float(metrics.get("threshold_4_rate"))),
            },
            "erosion": {
                "risk_score": fmt_metric(safe_float(erosion.get("risk_score"))),
                "risk_level": str(erosion.get("risk_level", "not_measured")),
                "source": str(erosion.get("source", "missing")),
            },
        }

    def read_text_result(self, path: str) -> str:
        if not path:
            return ""
        if self.should_run_remote():
            command = f"if [ -f {shlex.quote(path)} ]; then cat {shlex.quote(path)}; fi"
            try:
                result = self.run_remote_config_command(command)
            except Exception as exc:
                self.append_log(f"[dashboard] failed to read remote result {path}: {exc}")
                return ""
            if result.exit_code != 0:
                return ""
            return result.stdout
        local_path = Path(path)
        if not local_path.is_file():
            return ""
        return local_path.read_text(encoding="utf-8", errors="replace")

    def read_json_result(self, path: str) -> dict[str, Any]:
        text = self.read_text_result(path)
        if not text.strip():
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def read_tsv_result(self, path: str) -> dict[str, str]:
        text = self.read_text_result(path)
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return {}
        keys = lines[0].split("\t")
        values = lines[1].split("\t")
        return {key: values[idx] if idx < len(values) else "" for idx, key in enumerate(keys)}

    def sim_dashboard_payload(self) -> dict[str, Any]:
        """Reveal fabricated metrics as the matching simulated stages finish."""
        done = self.sim_sync_completed()
        origin = sim_origin_result() if "origin_eval" in done else {}
        teacher = sim_teacher_result() if "teacher_eval" in done else {}
        student = sim_student_result() if "student_eval" in done else {}
        reason = dict(SIM_REASON_SUMMARY) if "reason_judge" in done else {}
        watermark = calculate_watermark_erosion(0.82, 0.34, 0.08) if "risk_report" in done else calculate_watermark_erosion(None, None)
        return self.format_dashboard(origin, teacher, student, reason, watermark)

    def dashboard_payload(self) -> dict[str, Any]:
        if self.debug:
            return self.sim_dashboard_payload()
        try:
            with self.lock:
                result_dir = str(self.console_vars.get("result_dir", ""))
            origin = self.read_json_result(_path_join(result_dir, "origin_scienceqa_control_suite_full_fast.json"))
            teacher = self.read_json_result(_path_join(result_dir, "scienceqa_control_suite_teacher_full.json"))
            student = self.read_json_result(_path_join(result_dir, "scienceqa_control_suite_full_fast.json"))
            report = self.read_json_result(_path_join(result_dir, "mm_eval_suite_report_full_fast.json"))
            reason = self.read_tsv_result(_path_join(self.resolve_reason_out_dir(), "summary.tsv"))
            if not student and report:
                student = {"metrics": report.get("control_summary", {})} if isinstance(report, dict) else {}
            return self.format_dashboard(origin, teacher, student, reason, self.watermark_erosion_payload())
        except Exception as exc:
            self.append_log(f"[dashboard] refresh failed: {exc}")
            return self.format_dashboard({}, {}, {}, {}, calculate_watermark_erosion(None, None))

    def format_dashboard(
        self,
        origin: dict[str, Any],
        teacher: dict[str, Any],
        student: dict[str, Any],
        reason: dict[str, str],
        watermark: dict[str, Any],
    ) -> dict[str, Any]:
        clr = calculate_capability_leakage(origin, teacher, student, reason)
        teacher_summary = extract_control_summary(teacher)
        student_summary = extract_control_summary(student)
        origin_acc = extract_baseline_accuracy(origin)
        teacher_acc = extract_baseline_accuracy(teacher)
        student_acc = extract_baseline_accuracy(student)
        risk = safe_float(clr.get("risk_score"))
        reason_delta = safe_float(reason.get("delta_reason_score")) if isinstance(reason, dict) else None
        controls = []
        vr_rows = {
            row["control"]: row
            for row in clr.get("dimensions", {}).get("VR", {}).get("controls", [])
            if isinstance(row, dict) and row.get("control")
        }
        for name in sorted(set(teacher_summary) | set(student_summary)):
            teacher_score = safe_float(teacher_summary.get(name, {}).get("accuracy")) if isinstance(teacher_summary.get(name), dict) else None
            student_score = safe_float(student_summary.get(name, {}).get("accuracy")) if isinstance(student_summary.get(name), dict) else None
            vr_row = vr_rows.get(name, {})
            controls.append(
                {
                    "control": name,
                    "teacher": fmt_metric(teacher_score),
                    "student": fmt_metric(student_score),
                    "teacher_drop": fmt_metric(safe_float(vr_row.get("teacher_normalized_drop"))),
                    "student_drop": fmt_metric(safe_float(vr_row.get("student_normalized_drop"))),
                    "similarity": fmt_metric(safe_float(vr_row.get("drop_similarity"))),
                }
            )
        cot_dimensions = []
        cot_payload = clr.get("dimensions", {}).get("CoT", {})
        for name, weight in COT_WEIGHTS.items():
            row = cot_payload.get("dimensions", {}).get(name, {}) if isinstance(cot_payload, dict) else {}
            normalized = safe_float(row.get("normalized")) if isinstance(row, dict) else None
            cot_dimensions.append(
                {
                    "dimension": name,
                    "raw": fmt_metric(safe_float(row.get("raw")) if isinstance(row, dict) else None),
                    "normalized": fmt_metric(normalized),
                    "weight": fmt_metric(weight),
                    "contribution": fmt_metric(normalized * weight if normalized is not None else None),
                }
            )
        dimensions = clr.get("dimensions", {})
        acc_score = safe_float(dimensions.get("ACC", {}).get("score"))
        vr_score = safe_float(dimensions.get("VR", {}).get("score"))
        cot_score = safe_float(dimensions.get("CoT", {}).get("score"))
        wer_score = safe_float(watermark.get("risk_score"))
        return {
            "schema_version": 2,
            "capability_leakage": clr,
            "watermark_erosion": watermark,
            "metrics": {
                "defense": fmt_metric(1.0 - risk if risk is not None else None),
                "risk": fmt_metric(risk),
                "clr": fmt_metric(risk),
                "risk_level": str(clr.get("risk_level", "not_measured")),
                "confidence": str(clr.get("confidence", "low")),
                "coverage": fmt_metric(safe_float(clr.get("coverage_weight"))),
                "origin_acc": fmt_metric(origin_acc),
                "teacher_acc": fmt_metric(teacher_acc),
                "student_acc": fmt_metric(student_acc),
                "acc": fmt_metric(acc_score),
                "vr": fmt_metric(vr_score),
                "cot": fmt_metric(cot_score),
                "wer": fmt_metric(wer_score),
                "wer_level": str(watermark.get("risk_level", "not_measured")),
                "reason_delta": fmt_metric(reason_delta, signed=True),
            },
            "controls": controls,
            "cot_dimensions": cot_dimensions,
            "evidence": {
                "missing_dimensions": list(clr.get("missing_dimensions", [])),
                "acc_source": str(dimensions.get("ACC", {}).get("source", "missing")),
                "vr_source": str(dimensions.get("VR", {}).get("source", "missing")),
                "cot_source": str(dimensions.get("CoT", {}).get("source", "missing")),
                "wer_source": str(watermark.get("source", "missing")),
            },
            "reason": {
                "n": str(reason.get("n", "-")) if isinstance(reason, dict) else "-",
                "stage1_reason": fmt_metric(safe_float(reason.get("stage2_reason_score")) if isinstance(reason, dict) else None),
                "stage2_reason": fmt_metric(safe_float(reason.get("stage3_reason_score")) if isinstance(reason, dict) else None),
                "delta": fmt_metric(reason_delta, signed=True),
                "stage2_win": fmt_metric(safe_float(reason.get("stage2_win_rate")) if isinstance(reason, dict) else None),
            },
        }

    def watermark_erosion_payload(self) -> dict[str, Any]:
        with self.lock:
            victim = self.console_vars.get("wm_base_before_score")
            extracted = self.console_vars.get("wm_extracted_score")
            clean = self.console_vars.get("wm_test_score")
        return calculate_watermark_erosion(victim, extracted, clean)

    def save_risk_report(self) -> str:
        dashboard = self.dashboard_payload()
        report = {
            "schema_version": dashboard.get("schema_version", 2),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "capability_leakage": dashboard.get("capability_leakage", {}),
            "watermark_erosion": dashboard.get("watermark_erosion", {}),
        }
        with self.lock:
            result_dir = str(self.console_vars.get("result_dir", ""))
        path = _path_join(result_dir, "capability_leakage_risk_report.json")
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        if self.should_run_remote():
            command = f"mkdir -p {shlex.quote(result_dir)} && printf %s {shlex.quote(payload)} > {shlex.quote(path)}"
            result = self.run_remote_config_command(command)
            if result.exit_code != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "failed to save risk report")
        else:
            local_path = Path(path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(payload, encoding="utf-8")
        self.append_log(f"[risk] saved standard report: {path}")
        return path

    def dashboard_control_summary(self, payload: object) -> dict[str, Any]:
        return extract_control_summary(payload)

    def dashboard_baseline_accuracy(self, payload: object) -> float | None:
        return extract_baseline_accuracy(payload)


def create_app(debug: bool = False) -> FastAPI:
    app = FastAPI(title="MLLM Risk Console")
    console = WebConsole(debug=debug)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML.replace("__DEBUG_ENABLED__", "true" if debug else "false")

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return console.snapshot()

    @app.post("/api/config")
    def update_config(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        return console.update_config(payload)

    @app.post("/api/clone")
    def clone(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        return console.clone_repository(payload)

    @app.post("/api/remote/load-config")
    def load_remote_config(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        console.update_config(payload)
        return console.load_remote_config()

    @app.post("/api/remote/save-config")
    def save_remote_config(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        console.update_config(payload)
        return console.save_remote_config()

    @app.post("/api/tasks/{step_id}")
    def start_task(step_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        console.update_config(payload)
        definitions = console.task_definitions()
        if step_id not in definitions:
            return JSONResponse({"ok": False, "message": f"unknown task: {step_id}"}, status_code=404)
        return JSONResponse(console.start_tasks([step_id]))

    @app.post("/api/pipeline/full")
    def start_full_pipeline(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        console.update_config(payload)
        return console.start_tasks([step_id for step_id, _, _ in PIPELINE_STEPS])

    @app.get("/api/tasks/status")
    def task_status() -> dict[str, Any]:
        return console.pipeline_status()

    @app.get("/api/logs")
    def logs() -> dict[str, Any]:
        status = console.pipeline_status()
        return {"logs": status.get("logs", []), "local_log_path": status.get("local_log_path", "")}

    @app.get("/api/dashboard")
    def dashboard() -> dict[str, Any]:
        return console.dashboard_payload()

    @app.get("/api/watermark")
    def watermark() -> dict[str, Any]:
        return console.watermark_payload()

    @app.get("/api/history")
    def history_list() -> dict[str, Any]:
        return {"ok": True, "items": console.history_list(), "dir": str(HISTORY_DIR)}

    @app.get("/api/history/{history_id}")
    def history_get(history_id: str) -> dict[str, Any]:
        return console.history_get(history_id)

    @app.post("/api/history")
    def history_save(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        console.update_config(payload)
        return console.save_history(label=str(payload.get("label", "")).strip())

    @app.delete("/api/history/{history_id}")
    def history_delete(history_id: str) -> dict[str, Any]:
        return console.history_delete(history_id)

    @app.post("/api/assistant/chat")
    def assistant_chat(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        return console.assistant_chat(payload)

    return app


def launch_web_app(debug: bool = False, host: str = "127.0.0.1", port: int = 8011, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(debug=debug), host=host, port=port, log_level="info")


INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MLLM能力泄漏风险检测平台</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect width='24' height='24' rx='6' fill='%232563eb'/%3E%3Ccircle cx='12' cy='12' r='7' stroke='rgba(255,255,255,.6)' stroke-width='1.4' fill='none'/%3E%3Ccircle cx='12' cy='12' r='3.5' stroke='rgba(255,255,255,.45)' stroke-width='1.2' fill='none'/%3E%3Ccircle cx='12' cy='12' r='1.4' fill='white'/%3E%3Cpath d='M12 12 L12 5 A7 7 0 0 1 18.2 9.4 Z' fill='rgba(255,255,255,.4)'/%3E%3C/svg%3E">
  <style>
    :root {
      color-scheme: light;
      --bg: #eaf1fb;
      --bg-2: #f4f8ff;
      --surface: #ffffff;
      --surface-2: #f6f9ff;
      --surface-3: #eef4fe;
      --line: #dbe6f6;
      --line-2: #c2d5f0;
      --text: #0e1f38;
      --muted: #56688a;
      --dim: #8496b4;
      --accent: #2563eb;
      --accent-2: #1d4ed8;
      --accent-3: #3b82f6;
      --accent-soft: #dbeafe;
      --accent-glow: rgba(37,99,235,.28);
      --ok: #15a34a;
      --ok-soft: #dcfce7;
      --warn: #d97706;
      --warn-soft: #fef3c7;
      --bad: #dc2626;
      --bad-soft: #fee2e2;
      --blue: #2563eb;
      --input: #f7fafe;
      --ring: 0 0 0 3px rgba(37,99,235,.16);
      --shadow-sm: 0 1px 2px rgba(15,31,56,.06);
      --shadow: 0 10px 30px -14px rgba(23,54,110,.35), 0 2px 8px -3px rgba(23,54,110,.14);
      --shadow-lg: 0 26px 60px -26px rgba(23,54,110,.45);
      --font: "IBM Plex Sans", "Microsoft YaHei UI", "Source Han Sans SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Segoe UI", sans-serif;
      --display: "Sora", "IBM Plex Sans", "Microsoft YaHei UI", "Source Han Sans SC", sans-serif;
      --mono: "JetBrains Mono", "Cascadia Mono", "Consolas", "Source Han Mono SC", "Noto Sans Mono CJK SC", monospace;
    }
    * { box-sizing: border-box; }
    ::selection { background: var(--accent-soft); color: var(--accent-2); }
    body {
      margin: 0;
      min-width: 1230px;
      color: var(--text);
      font-family: var(--font);
      letter-spacing: .1px;
      background:
        radial-gradient(920px 620px at 88% -8%, rgba(59,130,246,.14), transparent 60%),
        radial-gradient(760px 560px at 4% 108%, rgba(37,99,235,.12), transparent 58%),
        linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 100%);
      background-attachment: fixed;
    }
    /* drifting atmosphere blobs behind everything */
    body::before, body::after {
      content: "";
      position: fixed;
      z-index: 0;
      border-radius: 50%;
      filter: blur(52px);
      opacity: .55;
      pointer-events: none;
    }
    body::before {
      width: 460px; height: 460px; top: -140px; right: 8%;
      background: radial-gradient(circle at 30% 30%, rgba(96,165,250,.55), rgba(37,99,235,0) 70%);
      animation: drift1 20s ease-in-out infinite;
    }
    body::after {
      width: 420px; height: 420px; bottom: -160px; left: 4%;
      background: radial-gradient(circle at 60% 40%, rgba(147,197,253,.5), rgba(37,99,235,0) 70%);
      animation: drift2 26s ease-in-out infinite;
    }
    @keyframes drift1 { 0%,100% { transform: translate(0,0) scale(1);} 50% { transform: translate(-38px,34px) scale(1.08);} }
    @keyframes drift2 { 0%,100% { transform: translate(0,0) scale(1);} 50% { transform: translate(30px,-30px) scale(1.1);} }
    button, input, textarea { font: inherit; }

    /* hairline light along the very top; flows while the pipeline is busy */
    .toplight {
      position: fixed; top: 0; left: 0; right: 0; height: 2px; z-index: 60; pointer-events: none;
      background: linear-gradient(90deg, transparent 4%, rgba(59,130,246,.5) 34%, rgba(29,78,216,.65) 50%, rgba(59,130,246,.5) 66%, transparent 96%);
      background-size: 200% 100%;
      opacity: .5;
    }
    body.is-busy .toplight { opacity: 1; animation: topflow 2.6s linear infinite; }
    @keyframes topflow { from { background-position: 200% 0; } to { background-position: -200% 0; } }

    .app {
      --sidebar-width: 372px;
      --sidebar-duration: .46s;
      --sidebar-ease: cubic-bezier(.22,.8,.28,1);
      position: relative; z-index: 1; display: grid;
      grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
      height: 100vh; overflow: hidden;
      transition: grid-template-columns var(--sidebar-duration) var(--sidebar-ease);
    }
    .app.sidebar-collapsed { --sidebar-width: 28px; }
    /* faint blueprint grid behind the content, fading toward the edges */
    .app::before {
      content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
      background:
        repeating-linear-gradient(0deg, rgba(37,99,235,.05) 0 1px, transparent 1px 56px),
        repeating-linear-gradient(90deg, rgba(37,99,235,.05) 0 1px, transparent 1px 56px);
      -webkit-mask-image: radial-gradient(1300px 850px at 62% 28%, rgba(0,0,0,.85), transparent 78%);
      mask-image: radial-gradient(1300px 850px at 62% 28%, rgba(0,0,0,.85), transparent 78%);
    }
    .sidebar {
      min-width: 0;
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(246,249,255,.86));
      backdrop-filter: blur(8px);
      overflow-x: hidden;
      overflow-y: auto;
      padding: 20px 18px 30px;
      box-shadow: 1px 0 0 rgba(255,255,255,.6), var(--shadow-sm);
      transition: padding var(--sidebar-duration) var(--sidebar-ease), box-shadow .3s ease;
    }
    .sidebar > .brand, .sidebar > .section, .sidebar > form {
      width: 336px;
      opacity: 1;
      transform: translateX(0);
      visibility: visible;
      transition: opacity .3s ease .08s, transform var(--sidebar-duration) var(--sidebar-ease);
    }
    .app.sidebar-collapsed .sidebar { padding-left: 0; padding-right: 0; overflow-y: hidden; box-shadow: 1px 0 0 rgba(255,255,255,.7); }
    .app.sidebar-collapsed .sidebar > .brand,
    .app.sidebar-collapsed .sidebar > .section,
    .app.sidebar-collapsed .sidebar > form {
      opacity: 0;
      transform: translateX(-72px);
      visibility: hidden;
      pointer-events: none;
      transition: opacity .28s ease, transform var(--sidebar-duration) var(--sidebar-ease), visibility 0s var(--sidebar-duration);
    }
    .main { position: relative; min-width: 0; overflow: auto; padding: 24px 26px 30px; }

    /* ---- scrollbars ---- */
    .sidebar::-webkit-scrollbar, .main::-webkit-scrollbar, .logbox::-webkit-scrollbar, .tabs::-webkit-scrollbar { width: 10px; height: 10px; }
    .sidebar::-webkit-scrollbar-thumb, .main::-webkit-scrollbar-thumb, .logbox::-webkit-scrollbar-thumb, .tabs::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 20px; border: 2px solid transparent; background-clip: padding-box; }
    .sidebar::-webkit-scrollbar-thumb:hover, .main::-webkit-scrollbar-thumb:hover { background: var(--accent-3); background-clip: padding-box; }

    .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
    .brand .mark {
      flex: none; width: 42px; height: 42px; border-radius: 12px;
      display: grid; place-items: center;
      background: linear-gradient(150deg, var(--accent-3), var(--accent-2));
      box-shadow: 0 8px 20px -8px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,.4);
      position: relative; overflow: hidden;
    }
    .brand .mark svg { width: 26px; height: 26px; display: block; }
    .brand .mark .sweep { transform-origin: 12px 12px; animation: radar 3.4s linear infinite; }
    @keyframes radar { to { transform: rotate(360deg); } }
    .brand .btitle { min-width: 0; }
    h1 { margin: 0; font-family: var(--display); font-size: 19px; line-height: 1.22; font-weight: 800; letter-spacing: .2px; }
    .brand .btitle small { display: block; margin-top: 3px; color: var(--dim); font-size: 10.5px; font-weight: 600; letter-spacing: 1.6px; text-transform: uppercase; }

    .badge {
      display: inline-flex; align-items: center; gap: 7px;
      border: 1px solid var(--line-2);
      background: linear-gradient(180deg, #fff, var(--surface-3));
      color: var(--accent-2);
      padding: 6px 11px; border-radius: 999px; font-size: 11px; font-weight: 700;
      letter-spacing: .6px; white-space: nowrap; box-shadow: var(--shadow-sm);
    }
    .badge::before {
      content: ""; width: 7px; height: 7px; border-radius: 50%;
      background: var(--ok); box-shadow: 0 0 0 0 rgba(21,163,74,.5);
      animation: livePulse 1.8s ease-out infinite;
    }
    @keyframes livePulse { 0% { box-shadow: 0 0 0 0 rgba(21,163,74,.5);} 70% { box-shadow: 0 0 0 7px rgba(21,163,74,0);} 100% { box-shadow: 0 0 0 0 rgba(21,163,74,0);} }

    .section { border-top: 1px solid var(--line); padding-top: 18px; margin-top: 18px; }
    .section:first-child { border-top: 0; margin-top: 0; padding-top: 0; }
    .section-title {
      color: var(--text); font-family: var(--display); font-size: 12px; font-weight: 700;
      margin: 0 0 13px; letter-spacing: .5px; text-transform: uppercase;
      display: flex; align-items: center; gap: 8px;
    }
    .section-title::before { content: ""; width: 4px; height: 14px; border-radius: 3px; background: linear-gradient(var(--accent-3), var(--accent-2)); box-shadow: 0 0 8px var(--accent-glow); }

    label { display: block; color: var(--muted); font-size: 11px; font-weight: 600; margin-bottom: 6px; letter-spacing: .2px; }
    input, textarea, select {
      width: 100%;
      background: var(--input);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 10px;
      outline: none;
      padding: 10px 11px;
      min-height: 40px;
      font-family: var(--mono);
      font-size: 12px;
      transition: border-color .18s ease, box-shadow .18s ease, background .18s ease;
    }
    input::placeholder { color: var(--dim); }
    input:hover, textarea:hover, select:hover { border-color: var(--line-2); }
    input:focus, textarea:focus, select:focus { border-color: var(--accent); background: #fff; box-shadow: var(--ring); }
    .reuse-field { margin-bottom: 11px; animation: fadeIn .4s ease both; }
    .reuse-head { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 7px; }
    .reuse-head > label:first-child { margin: 0; }
    .reuse-check { display: inline-flex; align-items: center; gap: 7px; flex: none; color: var(--text); font-size: 12px; font-weight: 650; cursor: pointer; }
    input.reuse-toggle { width: 18px; min-height: 18px; height: 18px; margin: 0; padding: 0; accent-color: var(--accent-2); box-shadow: none; cursor: pointer; }
    input.reuse-toggle:focus { box-shadow: 0 0 0 3px var(--accent-soft); }
    .field { margin-bottom: 11px; animation: fadeIn .4s ease both; }
    select { cursor: pointer; appearance: none; font-family: var(--font); font-size: 12.5px; font-weight: 600;
      padding-right: 30px;
      background-image: linear-gradient(45deg, transparent 50%, var(--muted) 50%), linear-gradient(135deg, var(--muted) 50%, transparent 50%);
      background-position: calc(100% - 16px) 17px, calc(100% - 11px) 17px;
      background-size: 5px 5px, 5px 5px; background-repeat: no-repeat; }
    .hist-hint { margin: 9px 0 0; color: var(--dim); font-size: 10.5px; line-height: 1.5; }

    /* viewing an archived run: amber accents so it can't be mistaken for live data */
    body.history-view #historySection { border-radius: 12px; padding: 14px 13px 13px; margin-top: 0;
      border: 1px solid rgba(217,119,6,.34); background: linear-gradient(180deg, rgba(254,243,199,.75), rgba(254,243,199,.32)); }
    body.history-view #historySection .section-title::before { background: linear-gradient(var(--warn), #b45309); box-shadow: 0 0 8px rgba(217,119,6,.4); }
    body.history-view .badge { color: #92400e; border-color: rgba(217,119,6,.4); background: linear-gradient(180deg,#fffbeb,#fef3c7); }
    body.history-view .badge::before { background: var(--warn); animation: none; box-shadow: none; }
    body.history-view .hero::after { animation: none; }

    .btnrow { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
    .btn {
      position: relative; overflow: hidden;
      border: 1px solid var(--line-2);
      background: linear-gradient(180deg, #fff, var(--surface-3));
      color: var(--text);
      min-height: 42px; padding: 10px 14px; border-radius: 11px;
      cursor: pointer; font-size: 13px; font-weight: 700; letter-spacing: .2px;
      box-shadow: var(--shadow-sm);
      transition: transform .15s ease, box-shadow .2s ease, border-color .2s ease, background .2s ease;
    }
    .btn:hover { border-color: var(--accent-3); background: #fff; transform: translateY(-1px); box-shadow: var(--shadow); }
    .btn:active { transform: translateY(0); }
    /* ripple sheen sweeping across on hover */
    .btn::after {
      content: ""; position: absolute; inset: 0;
      background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.6) 50%, transparent 70%);
      transform: translateX(-120%); transition: transform .55s ease;
    }
    .btn:hover::after { transform: translateX(120%); }
    .btn.primary {
      background: linear-gradient(135deg, var(--accent-3), var(--accent-2));
      color: #fff; border-color: transparent;
      box-shadow: 0 12px 26px -12px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,.35);
    }
    .btn.primary:hover { background: linear-gradient(135deg, #4b8bf7, var(--accent)); box-shadow: 0 16px 32px -12px var(--accent-glow); }
    .btn:disabled { opacity: .6; cursor: progress; transform: none; }
    .btn:disabled::after { animation: btnLoad 1.1s linear infinite; transform: none; }
    @keyframes btnLoad { 0% { transform: translateX(-120%);} 100% { transform: translateX(120%);} }

    .topbar { display: grid; grid-template-columns: minmax(0, 1fr) 430px; gap: 18px; align-items: stretch; margin-bottom: 18px; }
    .hero {
      position: relative; overflow: hidden;
      border: 1px solid var(--line);
      background:
        radial-gradient(520px 200px at 100% 0%, rgba(59,130,246,.1), transparent 70%),
        linear-gradient(180deg, rgba(255,255,255,.84), rgba(246,249,255,.74));
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      padding: 20px 22px; border-radius: 18px; box-shadow: var(--shadow);
      animation: cardRise .55s cubic-bezier(.2,.7,.3,1) both;
    }
    /* animated telemetry scanline across the hero */
    .hero::after {
      content: ""; position: absolute; top: 0; bottom: 0; width: 40%;
      background: linear-gradient(90deg, transparent, rgba(59,130,246,.08), transparent);
      animation: scan 5.5s ease-in-out infinite; pointer-events: none;
    }
    @keyframes scan { 0% { left: -40%; } 100% { left: 100%; } }
    .hero h1 { font-size: 23px; }
    .sidebar-toggle {
      position: absolute; z-index: 30;
      top: 50%; left: calc(var(--sidebar-width) + 3px);
      width: 16px; height: 74px; padding: 0;
      display: grid; place-items: center;
      border: 1px solid rgba(148,177,218,.55); border-radius: 0 8px 8px 0;
      background: rgba(255,255,255,.58);
      -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
      color: var(--accent-2); box-shadow: 2px 0 9px rgba(23,54,110,.1);
      opacity: .6;
      overflow: hidden;
      cursor: pointer;
      transform: translateY(-50%);
      transition: left var(--sidebar-duration) var(--sidebar-ease), opacity .18s ease, color .18s ease, border-color .18s ease, background .18s ease, box-shadow .22s ease, transform .18s ease;
    }
    .sidebar-toggle:hover { color: var(--accent); border-color: rgba(59,130,246,.72); background: rgba(255,255,255,.86); box-shadow: 3px 0 12px rgba(23,54,110,.16); opacity: .95; }
    .sidebar-toggle:active { transform: translateY(-50%) scale(.94); }
    .sidebar-toggle:focus-visible { outline: 2px solid var(--accent-3); outline-offset: 3px; }
    .sidebar-toggle svg { width: 14px; height: 30px; display: block; filter: drop-shadow(0 1px 1px rgba(29,78,216,.18)); }
    .sidebar-toggle .toggle-chevron {
      transform-box: fill-box;
      transform-origin: center;
      transition: transform var(--sidebar-duration) var(--sidebar-ease);
    }
    .app.sidebar-collapsed .sidebar-toggle .toggle-chevron { transform: rotate(180deg); }
    .hero p { margin: 10px 0 0; color: var(--muted); font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 9px; }
    .hero p::before { content: ""; flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
    .actions {
      display: grid; grid-template-columns: 1fr 150px; gap: 12px; align-content: center;
      border: 1px solid var(--line); border-radius: 18px; padding: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,.84), rgba(246,249,255,.74));
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      box-shadow: var(--shadow);
      animation: cardRise .55s cubic-bezier(.2,.7,.3,1) .06s both;
    }

    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 13px; margin-bottom: 20px; counter-reset: metric; }
    .metric {
      position: relative; overflow: hidden;
      border: 1px solid var(--line); border-radius: 16px;
      background: linear-gradient(180deg, #ffffff, var(--surface-2));
      min-height: 96px; padding: 15px 16px; box-shadow: var(--shadow-sm);
      transition: transform .18s ease, box-shadow .2s ease, border-color .2s ease;
      animation: cardRise .5s cubic-bezier(.2,.7,.3,1) both;
      counter-increment: metric;
    }
    .metric:nth-child(1){animation-delay:.05s} .metric:nth-child(2){animation-delay:.1s}
    .metric:nth-child(3){animation-delay:.15s} .metric:nth-child(4){animation-delay:.2s}
    .metric:hover {
      transform: translateY(-3px); box-shadow: var(--shadow); border-color: transparent;
      background:
        linear-gradient(180deg, #ffffff, var(--surface-2)) padding-box,
        linear-gradient(150deg, var(--accent-3), var(--line-2) 45%, var(--accent-2)) border-box;
    }
    .metric::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: linear-gradient(var(--accent-3), var(--accent-2)); }
    .metric::after { content: ""; position: absolute; right: -30px; top: -30px; width: 90px; height: 90px; border-radius: 50%; background: radial-gradient(circle, rgba(59,130,246,.12), transparent 70%); }
    .metric span { display: block; color: var(--muted); font-size: 11.5px; font-weight: 600; margin-bottom: 12px; letter-spacing: .3px; }
    /* HUD-style card index in the top-right corner */
    .metric span::after {
      content: counter(metric, decimal-leading-zero);
      position: absolute; top: 13px; right: 15px;
      font-family: var(--mono); font-size: 10px; font-weight: 500; letter-spacing: 1px;
      color: var(--dim); opacity: .75;
    }
    .metric strong { font-family: var(--display); font-size: 27px; line-height: 1; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; }
    .metric.flash strong { animation: metricFlash .9s ease; }
    @keyframes metricFlash { 0% { color: var(--accent); transform: scale(1.12); } 100% { color: var(--text); transform: scale(1); } }
    .risk-summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:13px; margin:14px 0; }
    .risk-summary .metric { min-height:88px; }
    .risk-meta { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .risk-pill { display:inline-flex; align-items:center; min-height:30px; padding:6px 10px; border:1px solid var(--line); border-radius:8px; background:var(--surface-2); color:var(--muted); font-size:11px; font-family:var(--mono); }
    .risk-pill strong { color:var(--text); margin-left:6px; }
    .risk-level-critical { color:#b91c1c; border-color:rgba(185,28,28,.28); background:#fef2f2; }
    .risk-level-high { color:#c2410c; border-color:rgba(194,65,12,.28); background:#fff7ed; }
    .risk-level-medium { color:#a16207; border-color:rgba(161,98,7,.28); background:#fefce8; }
    .risk-level-low { color:#15803d; border-color:rgba(21,128,61,.28); background:#f0fdf4; }
    .risk-note { margin:12px 0 0; color:var(--muted); font-size:12px; line-height:1.6; }

    /* watermark z-mean hero card */
    .wm-hero { display: flex; align-items: center; justify-content: space-between; gap: 22px; overflow: hidden; position: relative; }
    .wm-hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 200px; height: 200px; border-radius: 50%; background: radial-gradient(circle, rgba(59,130,246,.14), transparent 70%); }
    .wm-hero-main { display: flex; flex-direction: column; gap: 6px; z-index: 1; }
    .wm-cap { color: var(--muted); font-size: 12.5px; font-weight: 600; letter-spacing: .3px; }
    .wm-hero-main strong { font-family: var(--display); font-size: 52px; line-height: 1; font-weight: 800; color: var(--accent-2); font-variant-numeric: tabular-nums; }
    .wm-hero-main strong.flash { animation: metricFlash .9s ease; }
    .wm-sub { color: var(--dim); font-size: 12px; font-family: var(--mono); }
    .wm-gauge { position: relative; width: 128px; height: 128px; border-radius: 50%; flex: none; z-index: 1;
      background: conic-gradient(var(--accent-3), var(--accent-2) var(--deg,0deg), var(--surface-3) 0deg);
      box-shadow: 0 14px 30px -14px var(--accent-glow);
      transition: --deg .8s cubic-bezier(.3,.8,.3,1); display: grid; place-items: center; }
    .wm-gauge::before { content: ""; position: absolute; inset: 12px; border-radius: 50%; background: linear-gradient(180deg,#fff,var(--surface-2)); box-shadow: inset 0 1px 4px rgba(23,54,110,.12); }
    /* tick mark at the z=4 detection threshold (half of the 0..8 scale = 180deg = bottom) */
    .wm-gauge::after { content: ""; position: absolute; left: 50%; bottom: 0; width: 2px; height: 11px; transform: translateX(-50%); border-radius: 2px; background: var(--warn); opacity: .85; }
    .wm-gauge span { position: relative; z-index: 1; font-family: var(--mono); font-size: 12px; color: var(--muted); }
    .wm-metrics { grid-template-columns: repeat(4, minmax(0,1fr)); }
    @property --deg { syntax: "<angle>"; inherits: false; initial-value: 0deg; }

    .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--line); margin-bottom: 18px; overflow: auto; padding-bottom: 0; }
    .tab {
      position: relative;
      border: 1px solid transparent; border-bottom: 0;
      background: transparent; color: var(--muted);
      padding: 12px 16px; cursor: pointer; font-weight: 600; font-size: 13px;
      white-space: nowrap; border-radius: 12px 12px 0 0;
      transition: color .18s ease, background .18s ease;
    }
    .tab:hover { color: var(--accent-2); background: rgba(37,99,235,.05); }
    .tab::after { content: ""; position: absolute; left: 12px; right: 12px; bottom: -1px; height: 3px; border-radius: 3px 3px 0 0; background: linear-gradient(90deg, var(--accent-3), var(--accent-2)); transform: scaleX(0); transform-origin: center; transition: transform .25s cubic-bezier(.2,.7,.3,1); }
    .tab.active { color: var(--accent-2); background: linear-gradient(180deg, #fff, var(--surface-2)); border-color: var(--line); box-shadow: 0 -3px 10px -8px rgba(23,54,110,.3); }
    .tab.active::after { transform: scaleX(1); box-shadow: 0 2px 12px var(--accent-glow); }
    .btn:focus-visible, .tab:focus-visible { outline: 2px solid var(--accent-3); outline-offset: 2px; }

    .pane { display: none; }
    .pane.active { display: block; animation: paneIn .38s cubic-bezier(.2,.7,.3,1) both; }
    @keyframes paneIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .panel {
      border: 1px solid var(--line); border-radius: 16px;
      background: linear-gradient(180deg, #ffffff, var(--surface-2));
      padding: 16px 17px; margin-bottom: 15px; box-shadow: var(--shadow-sm);
    }
    .panel-title { font-family: var(--display); font-weight: 700; margin: 0 0 14px; font-size: 15px; letter-spacing: .2px; display: flex; align-items: center; gap: 9px; }
    .panel-title::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }

    .runline { display: grid; grid-template-columns: minmax(0, 1fr) 200px; gap: 16px; align-items: center; }
    .prog-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 9px; gap: 10px; }
    .prog-head .panel-title { margin: 0; }
    .prog-pct { font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--accent-2); font-variant-numeric: tabular-nums; }
    .progress {
      position: relative; height: 12px; border: 1px solid var(--line);
      background: var(--surface-3); border-radius: 999px; overflow: hidden;
      box-shadow: inset 0 1px 3px rgba(23,54,110,.1);
    }
    .progress > div {
      position: relative; height: 100%; width: 0; border-radius: 999px;
      background: linear-gradient(90deg, var(--accent-3), var(--accent-2));
      transition: width .5s cubic-bezier(.3,.8,.3,1);
    }
    /* moving stripes + shimmer while a task is running */
    .progress.is-running > div {
      background-image:
        linear-gradient(90deg, var(--accent-3), var(--accent-2)),
        repeating-linear-gradient(45deg, rgba(255,255,255,.28) 0 12px, transparent 12px 24px);
      background-blend-mode: overlay;
      animation: stripes 1s linear infinite;
    }
    .progress.is-running > div::after {
      content: ""; position: absolute; inset: 0;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,.55), transparent);
      transform: translateX(-100%); animation: sweep 1.6s ease-in-out infinite;
    }
    @keyframes stripes { from { background-position: 0 0, 0 0; } to { background-position: 0 0, 48px 0; } }
    @keyframes sweep { 0% { transform: translateX(-100%);} 60%,100% { transform: translateX(180%);} }

    table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; background: var(--surface); box-shadow: var(--shadow-sm); }
    th, td { border-bottom: 1px solid var(--line); padding: 11px 13px; text-align: left; font-size: 13px; }
    th { color: var(--muted); background: var(--surface-3); font-family: var(--display); font-weight: 600; font-size: 11.5px; letter-spacing: .5px; text-transform: uppercase; }
    td { color: var(--text); font-family: var(--mono); }
    tbody tr { transition: background .15s ease; }
    tbody tr:hover { background: var(--surface-2); }
    tr:last-child td { border-bottom: 0; }
    /* highlight the stage currently executing */
    tr.row-running td { background: rgba(37,99,235,.055); }
    tr.row-running td:first-child { box-shadow: inset 3px 0 0 var(--accent-3); }

    /* status pills */
    .pill { display: inline-flex; align-items: center; gap: 7px; padding: 4px 11px; border-radius: 999px; font-family: var(--font); font-size: 12px; font-weight: 600; border: 1px solid transparent; }
    .pill i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex: none; }
    .pill.status-success { color: var(--ok); background: var(--ok-soft); border-color: rgba(21,163,74,.25); }
    .pill.status-running { color: var(--blue); background: var(--accent-soft); border-color: rgba(37,99,235,.25); }
    .pill.status-running i { animation: livePulse 1.4s ease-out infinite; }
    .pill.status-failed { color: var(--bad); background: var(--bad-soft); border-color: rgba(220,38,38,.25); }
    .pill.status-pending { color: var(--muted); background: var(--surface-3); border-color: var(--line); }
    /* legacy text colors (kept for any direct status-* usage) */
    .status-success { color: var(--ok); }
    .status-running { color: var(--blue); }
    .status-failed { color: var(--bad); }
    .status-pending { color: var(--muted); }

    /* per-row mini progress */
    .mini { display: flex; align-items: center; gap: 9px; }
    .mini-track { flex: 1; height: 7px; border-radius: 999px; background: var(--surface-3); overflow: hidden; min-width: 60px; box-shadow: inset 0 1px 2px rgba(23,54,110,.1); }
    .mini-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent-3), var(--accent-2)); transition: width .5s cubic-bezier(.3,.8,.3,1); }
    .mini-fill.run { background-image: linear-gradient(90deg, var(--accent-3), var(--accent-2)), repeating-linear-gradient(45deg, rgba(255,255,255,.3) 0 8px, transparent 8px 16px); animation: stripes 1s linear infinite; }
    .mini-label { font-family: var(--mono); font-size: 11.5px; color: var(--muted); white-space: nowrap; }

    .logbox {
      position: relative;
      height: 440px; overflow: auto; white-space: pre-wrap;
      background:
        repeating-linear-gradient(0deg, rgba(148,197,255,.03) 0 1px, transparent 1px 3px),
        linear-gradient(180deg, rgba(37,99,235,.05), transparent 90px),
        #0d1b30;
      border: 1px solid #16345f; border-radius: 14px;
      padding: 0; font-family: var(--mono); font-size: 12px; line-height: 1.6;
      color: #cfe0f7; box-shadow: inset 0 2px 14px rgba(0,0,0,.35);
    }
    .logbox::-webkit-scrollbar-thumb { background: #2c4f80; }
    /* semantic log-line tinting (classes assigned in renderPipeline) */
    .log-line { display: block; }
    .log-line:empty::before { content: "\00a0"; }
    .log-line.log-err { color: #ff9285; }
    .log-line.log-warn { color: #ffd479; }
    .log-line.log-ok { color: #7ee2a8; }
    .log-line.log-prog { color: #8ec1ff; }
    /* wrapper so the jump-to-bottom chip can float over the scroller */
    .log-wrap { position: relative; }
    .log-jump {
      position: absolute; right: 14px; bottom: 14px; z-index: 3;
      border: 1px solid rgba(120,170,240,.35); border-radius: 999px;
      background: rgba(16,32,58,.92); color: #9fc0ee;
      font-family: var(--mono); font-size: 11px; padding: 6px 12px; cursor: pointer;
      opacity: 0; pointer-events: none; transform: translateY(6px);
      transition: opacity .2s ease, transform .2s ease, border-color .2s ease, color .2s ease;
    }
    .log-jump.show { opacity: 1; pointer-events: auto; transform: translateY(0); }
    .log-jump:hover { color: #fff; border-color: var(--accent-3); }
    /* panel header row (outside the dark box) */
    .log-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
    /* terminal title bar pinned to the top of the dark box */
    .log-titlebar {
      position: sticky; top: 0; z-index: 2;
      display: flex; align-items: center; gap: 9px;
      padding: 10px 15px;
      background: linear-gradient(180deg, #10203a, #0c1a30);
      border-bottom: 1px solid rgba(120,170,240,.14);
      border-radius: 13px 13px 0 0;
    }
    .log-titlebar .log-path { font-family: var(--mono); font-size: 11.5px; color: #9fb6d8; letter-spacing: .3px; }
    .log-titlebar .log-tag { margin-left: auto; display: inline-flex; align-items: center; gap: 7px; }
    /* macOS-style traffic lights */
    .log-dots { display: inline-flex; gap: 6px; margin-right: 3px; flex: none; }
    .log-dots i { width: 10px; height: 10px; border-radius: 50%; box-shadow: inset 0 -1px 1px rgba(0,0,0,.25); }
    .log-dots i:nth-child(1) { background: #ff5f57; }
    .log-dots i:nth-child(2) { background: #febc2e; }
    .log-dots i:nth-child(3) { background: #28c840; }
    .log-live {
      width: 8px; height: 8px; border-radius: 50%; flex: none;
      background: #28c840; box-shadow: 0 0 0 0 rgba(40,200,64,.5);
      animation: livePulse 1.8s ease-out infinite;
    }
    .log-body { padding: 12px 16px 16px; }
    .log-tag { font-family: var(--mono); font-size: 11px; color: var(--dim); letter-spacing: .5px; }

    .muted { color: var(--muted); }
    .debug-only { display: none; }
    body.debug .debug-only { display: block; }

    .assistant-orb { position: fixed; z-index: 1200; right: 22px; bottom: 24px; width: 58px; height: 58px; border: 0; border-radius: 50%; display: grid; place-items: center; background: linear-gradient(145deg, var(--accent-3), var(--accent-2)); color: #fff; font-family: var(--display); font-size: 16px; font-weight: 800; letter-spacing: .5px; box-shadow: 0 10px 24px rgba(37,99,235,.28), 0 0 0 5px rgba(37,99,235,.09); cursor: grab; user-select: none; touch-action: none; transition: left .28s ease, top .28s ease, right .28s ease, bottom .28s ease, transform .18s ease, box-shadow .18s ease; }
    .assistant-orb:hover { transform: translateY(-2px) scale(1.04); box-shadow: 0 14px 30px rgba(37,99,235,.34), 0 0 0 6px rgba(37,99,235,.11); }
    .assistant-orb.dragging { cursor: grabbing; transition: none; transform: scale(1.04); }
    .assistant-orb.assistant-hidden { opacity: 0; pointer-events: none; transform: scale(.72); }
    .assistant-orb span, .assistant-orb svg { pointer-events: none; }
    .assistant-orb svg { width: 29px; height: 29px; display: block; transform: translate(0, 2px); }
    .assistant-panel { position: fixed; z-index: 1190; right: 22px; bottom: 94px; width: min(410px, calc(100vw - 32px)); height: min(610px, calc(100vh - 120px)); display: flex; flex-direction: column; overflow: hidden; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.97); box-shadow: 0 18px 50px rgba(23,54,110,.22); opacity: 0; pointer-events: none; transform: translateY(12px) scale(.98); transition: opacity .18s ease, transform .18s ease; }
    .assistant-panel.open { opacity: 1; pointer-events: auto; transform: translateY(0) scale(1); }
    .assistant-head { display: flex; align-items: center; gap: 10px; padding: 14px 15px; border-bottom: 1px solid var(--line); background: linear-gradient(135deg, #f8fbff, #eef5ff); }
    .assistant-head strong { flex: 1; font-family: var(--display); font-size: 14px; }
    .assistant-head small { color: var(--muted); font-size: 11px; }
    .assistant-icon { width: 30px; height: 30px; display: grid; place-items: center; border-radius: 10px; background: var(--accent-soft); color: var(--accent-2); font-weight: 800; }
    .assistant-icon-btn { width: 30px; height: 30px; padding: 0; border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--muted); cursor: pointer; }
    .assistant-icon-btn:hover { color: var(--accent-2); border-color: var(--accent-3); }
    .assistant-messages { flex: 1; min-height: 0; overflow-y: auto; padding: 14px; background: #f8fbff; }
    .assistant-msg { max-width: 88%; margin: 0 0 11px; padding: 10px 12px; border-radius: 11px; font-size: 12.5px; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
    .assistant-msg.assistant { margin-right: auto; border: 1px solid var(--line); background: #fff; color: var(--text); white-space: normal; }
    .assistant-msg.user { margin-left: auto; background: var(--accent-2); color: #fff; }
    .assistant-msg.assistant p { margin: 0 0 7px; }
    .assistant-msg.assistant p:last-child { margin-bottom: 0; }
    .assistant-msg.assistant ul { margin: 5px 0 7px 18px; padding: 0; }
    .assistant-msg.assistant li { margin: 2px 0; }
    .assistant-msg.assistant code { padding: 1px 4px; border-radius: 4px; background: #edf3ff; color: #174ea6; font-family: var(--mono); font-size: 11.5px; }
    .assistant-msg.assistant pre { margin: 7px 0 2px; padding: 9px 10px; overflow-x: auto; border-radius: 7px; background: #0d1b30; color: #dbeafe; font-family: var(--mono); font-size: 11px; line-height: 1.5; white-space: pre; }
    .assistant-msg.assistant pre code { padding: 0; background: transparent; color: inherit; }
    .assistant-settings { display: none; padding: 10px 14px; border-bottom: 1px solid var(--line); background: #fff; }
    .assistant-settings.open { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .assistant-settings .field { margin: 0; }
    .assistant-settings .field:last-child { grid-column: 1 / -1; }
    .assistant-settings label { font-size: 10px; margin-bottom: 4px; }
    .assistant-settings input { min-height: 32px; padding: 7px 8px; font-size: 11px; }
    .assistant-compose { display: flex; align-items: flex-end; gap: 8px; padding: 11px; border-top: 1px solid var(--line); background: #fff; }
    .assistant-compose textarea { min-height: 42px; max-height: 110px; resize: vertical; padding: 9px 10px; font-family: var(--font); font-size: 12.5px; }
    .assistant-send { flex: none; min-height: 42px; padding: 9px 12px; }
    .assistant-send:disabled { opacity: .55; cursor: wait; }

    @keyframes cardRise { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

    @media (prefers-reduced-motion: reduce) {
      * { animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
    }
    @media (max-width: 1260px) {
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .risk-summary { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .topbar { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      html, body { width:100%; max-width:100%; overflow:hidden; }
      .app { width:100vw; max-width:100vw; grid-template-columns:minmax(0,1fr); grid-template-rows:minmax(260px,42vh) minmax(0,1fr); transition: grid-template-rows var(--sidebar-duration) var(--sidebar-ease); }
      .app.sidebar-collapsed { grid-template-rows:74px minmax(0,1fr); }
      .sidebar { width:100%; max-width:100vw; border-right:0; border-bottom:1px solid var(--line); padding:16px; }
      .app.sidebar-collapsed .sidebar { padding:15px 16px; }
      .sidebar > .section, .sidebar > form { min-width:calc(100vw - 32px); }
      .main { width:100%; max-width:100vw; padding:16px 12px 24px; }
      .actions { grid-template-columns:1fr; }
      .actions .brand { grid-column:auto !important; }
      .runline { grid-template-columns:1fr; }
      .metrics, .risk-summary, .grid2, .grid3 { grid-template-columns:1fr; }
      .reuse-head { align-items:flex-start; }
      .reuse-check { white-space:normal; justify-content:flex-end; text-align:right; }
      .pane { width:100%; max-width:100%; overflow-x:auto; }
      .pane table { min-width:680px; }
      .hero h1 { font-size:20px; }
      .sidebar-toggle { top:42vh; left:50%; width:74px; height:16px; border-radius:0 0 8px 8px; transform:translate(-50%, 3px); }
      .sidebar-toggle:active { transform:translate(-50%, 3px) scale(.94); }
      .sidebar-toggle .toggle-chevron { transform:rotate(90deg); }
      .app.sidebar-collapsed .sidebar-toggle { top:74px; }
      .app.sidebar-collapsed .sidebar-toggle .toggle-chevron { transform:rotate(270deg); }
      .metric strong { font-size:24px; }
    }
    @media (max-width: 560px) { .assistant-panel { right: 12px; bottom: 84px; width: calc(100vw - 24px); height: min(600px, calc(100vh - 105px)); } .assistant-orb { right: 14px; bottom: 14px; } }
  </style>
</head>
<body>
  <div class="toplight" aria-hidden="true"></div>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="9" stroke="rgba(255,255,255,.55)" stroke-width="1.2"/>
            <circle cx="12" cy="12" r="5" stroke="rgba(255,255,255,.4)" stroke-width="1.1"/>
            <circle cx="12" cy="12" r="1.6" fill="#fff"/>
            <path class="sweep" d="M12 12 L12 3 A9 9 0 0 1 20 9 Z" fill="rgba(255,255,255,.35)"/>
          </svg>
        </div>
        <div class="btitle">
          <h1>MLLM能力泄漏<br>风险检测平台</h1>
          <small>Capability Leakage Radar</small>
        </div>
      </div>
      <div class="section" id="historySection">
        <p class="section-title">历史测评数据</p>
        <div class="field">
          <label>选择归档记录</label>
          <select id="historySelect"><option value="">当前数据（实时）</option></select>
        </div>
        <div class="btnrow" style="grid-template-columns:1fr 1fr;">
          <button class="btn" type="button" id="historyBackBtn">返回默认数据</button>
          <button class="btn" type="button" id="historySaveBtn">归档当前结果</button>
        </div>
        <button class="btn" type="button" id="historyDeleteBtn" style="width:100%;margin-top:10px;">删除选中记录</button>
        <p class="hist-hint" id="historyHint">跑完 Pipeline 会自动归档一份结果</p>
      </div>
      <form id="configForm">
        <div class="section">
          <p class="section-title">服务器连接</p>
          <div class="field"><label>SSH 连接命令</label><input data-root="connection_command" placeholder="ssh root@example.com -p 22"></div>
          <div class="field"><label>项目父目录</label><input data-root="project_path" placeholder="/root/workspace"></div>
          <div class="field"><label>SSH 密码/密钥口令</label><input data-root="password" type="password" placeholder=""></div>
          <div class="btnrow">
            <button class="btn" type="button" id="cloneBtn">Clone/进入</button>
            <button class="btn" type="button" id="loadRemoteBtn">读取远程配置</button>
          </div>
          <button class="btn" type="button" id="saveRemoteBtn" style="width:100%;margin-top:10px;">保存到服务器</button>
        </div>
        <div class="section" id="globalFields"></div>
        <div class="section" id="teacherFields"></div>
        <div class="section debug-only" id="debugFields"></div>
        <div class="section" id="runFields"></div>
      </form>
    </aside>
    <button class="sidebar-toggle" id="sidebarToggle" type="button" aria-label="收起左侧栏" aria-expanded="true" title="收起左侧栏">
      <svg viewBox="0 0 20 24" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path class="toggle-chevron" d="M14.5 4.5 6.5 12l8 7.5"></path>
      </svg>
    </button>
    <main class="main">
      <div class="topbar">
        <div class="hero">
          <h1>MLLM能力泄漏风险检测平台</h1>
          <p id="statusText">准备就绪</p>
        </div>
        <div class="actions">
          <div class="brand" style="grid-column:1 / -1; margin:0 0 2px;">
            <span id="modeBadge" class="badge">LIVE</span>
          </div>
          <button class="btn primary" id="runPipelineBtn">一键跑完整 Pipeline</button>
          <button class="btn" id="refreshDashboardBtn">刷新大盘</button>
        </div>
      </div>
      <div class="metrics">
        <div class="metric"><span>防蒸馏能力</span><strong id="metricDefense">-</strong></div>
        <div class="metric"><span>能力泄露风险 CLR</span><strong id="metricRisk">-</strong></div>
        <div class="metric"><span>风险等级</span><strong id="metricRiskLevel">-</strong></div>
        <div class="metric"><span>证据覆盖率</span><strong id="metricCoverage">-</strong></div>
      </div>
      <nav class="tabs" id="tabs">
        <button class="tab active" data-pane="statusPane">任务状态和日志信息</button>
        <button class="tab" data-pane="dataPane">1. 数据与教师基线</button>
        <button class="tab" data-pane="trainPane">2. Stage1/Stage2 蒸馏</button>
        <button class="tab" data-pane="evalPane">3. 完整评测</button>
        <button class="tab" data-pane="riskPane">4. 风险大盘</button>
        <button class="tab" data-pane="watermarkPane">水印检测</button>
      </nav>
      <section id="statusPane" class="pane active">
        <div class="panel">
          <div class="runline">
            <div>
              <div class="prog-head">
                <p class="panel-title" id="pipelineSummary">暂无任务</p>
                <span class="prog-pct" id="pipelinePercent">0%</span>
              </div>
              <div class="progress"><div id="pipelineProgress"></div></div>
            </div>
            <button class="btn" id="refreshStatusBtn">刷新任务状态</button>
          </div>
        </div>
        <table id="taskTable"><thead><tr><th>阶段</th><th>状态</th><th>进度</th><th>脚本</th></tr></thead><tbody></tbody></table>
        <div class="panel" style="margin-top:15px;">
          <div class="log-head">
            <p class="panel-title" style="margin:0;">系统后台日志</p>
            <span class="log-tag">stdout · live</span>
          </div>
          <div class="log-wrap">
            <div class="logbox">
              <div class="log-titlebar"><span class="log-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="log-path">remote://pipeline.log</span><span class="log-tag"><span class="log-live"></span>stdout · live</span></div>
              <div class="log-body"><div id="logBox">No real tasks yet.</div></div>
            </div>
            <button class="log-jump" id="logJumpBtn" type="button">↓ 回到底部</button>
          </div>
        </div>
      </section>
      <section id="watermarkPane" class="pane">
        <div class="panel">
          <p class="panel-title">水印失效风险检测 · z-mean</p>
          <div class="grid2">
            <div class="field"><label>学生四字段回答 JSON 路径</label><input data-console="watermark_input_json" placeholder="/path/to/stage3_..._parallel.json"></div>
            <div class="field"><label>抽样条数（0 = 按 20% 抽样）</label><input data-console="watermark_sample_size" placeholder="0"></div>
          </div>
          <div class="btnrow" style="grid-template-columns:1fr 1fr; max-width:520px;">
            <button class="btn primary" data-task="watermark_detect">启动 z-mean 检测</button>
            <button class="btn" id="refreshWatermarkBtn">刷新检测结果</button>
          </div>
        </div>
        <div class="wm-hero panel">
          <div class="wm-hero-main">
            <span class="wm-cap">z-mean 平均水印分数</span>
            <strong id="wmMeanZ">-</strong>
            <span class="wm-sub" id="wmScope">尚未运行检测</span>
          </div>
          <div class="wm-gauge" id="wmGauge"><span></span></div>
        </div>
        <div class="metrics wm-metrics">
          <div class="metric"><span>中位数 median z</span><strong id="wmMedianZ">-</strong></div>
          <div class="metric"><span>最小值 min z</span><strong id="wmMinZ">-</strong></div>
          <div class="metric"><span>最大值 max z</span><strong id="wmMaxZ">-</strong></div>
          <div class="metric"><span>水印检出率 (z&gt;4)</span><strong id="wmRate">-</strong></div>
        </div>
        <div class="grid3" id="watermarkFields"></div>
      </section>
      <section id="dataPane" class="pane">
        <div class="panel">
          <p class="panel-title">1A. 教师四字段标注采集</p>
          <div class="reuse-field">
            <div class="reuse-head">
              <label>教师标注 JSON 路径</label>
              <label class="reuse-check"><input class="reuse-toggle" type="checkbox" data-console="reuse_teacher_annotation">复用已有教师标注</label>
            </div>
            <input data-console="teacher_annotation_path" placeholder="/path/to/teacher_annotations.json">
          </div>
          <div class="grid2" id="teacherCollectFields"></div><button class="btn primary" data-task="teacher_collect">开始采集教师数据</button>
        </div>
        <div class="panel">
          <p class="panel-title">1B. 教师风险基线评测</p>
          <div class="reuse-field">
            <div class="reuse-head">
              <label>教师风险基线 JSON 路径</label>
              <label class="reuse-check"><input class="reuse-toggle" type="checkbox" data-console="reuse_teacher_baseline">复用已有教师风险基线</label>
            </div>
            <input data-console="teacher_baseline_path" placeholder="/path/to/scienceqa_control_suite_teacher_full.json">
          </div>
          <div class="grid2" id="teacherEvalFields"></div><button class="btn primary" data-task="teacher_eval">启动教师完整评测</button>
        </div>
      </section>
      <section id="trainPane" class="pane">
        <div class="panel"><p class="panel-title">路径配置</p><div class="grid2" id="trainPathFields"></div></div>
        <div class="panel"><p class="panel-title">Stage1 训练</p><div class="grid3" id="stage1Fields"></div><button class="btn primary" data-task="stage1_train">启动 Stage1 蒸馏</button></div>
        <div class="panel"><p class="panel-title">Stage2 训练</p><div class="grid3" id="stage2Fields"></div><button class="btn primary" data-task="stage2_train">启动 Stage2 蒸馏</button></div>
      </section>
      <section id="evalPane" class="pane">
        <div class="panel"><p class="panel-title">3A. 原始学生能力基线</p><button class="btn primary" data-task="origin_eval">启动 Origin 基线评测</button></div>
        <div class="panel"><p class="panel-title">3B. Stage2 学生完整风险评测</p><div class="grid2" id="studentEvalFields"></div><button class="btn primary" data-task="student_eval">启动学生完整评测</button></div>
        <div class="panel"><p class="panel-title">3C. 思维链评估</p><div class="grid2" id="reasonFields"></div><button class="btn primary" data-task="reason_judge">启动思维链评估</button></div>
      </section>
      <section id="riskPane" class="pane">
        <div class="panel">
          <div class="runline"><div><p class="panel-title">能力泄露风险 CLR</p><p class="risk-note" id="riskEvidenceNote">等待评测结果</p></div><button class="btn" id="refreshRiskBtn">刷新结果摘要</button></div>
          <div class="risk-summary">
            <div class="metric"><span>正确率迁移 ACC · 50%</span><strong id="metricAcc">-</strong></div>
            <div class="metric"><span>视觉依赖 VR · 30%</span><strong id="metricVr">-</strong></div>
            <div class="metric"><span>思维链 CoT · 20%</span><strong id="metricCot">-</strong></div>
            <div class="metric"><span>水印削弱 WER</span><strong id="metricWer">-</strong></div>
          </div>
          <div class="risk-meta">
            <span class="risk-pill">Origin Acc<strong id="riskOriginAcc">-</strong></span>
            <span class="risk-pill">Victim Acc<strong id="riskTeacherAcc">-</strong></span>
            <span class="risk-pill">Stage2 Acc<strong id="riskStudentAcc">-</strong></span>
            <span class="risk-pill">Confidence<strong id="riskConfidence">-</strong></span>
            <span class="risk-pill">WER Level<strong id="riskWerLevel">-</strong></span>
          </div>
        </div>
        <p class="panel-title">视觉扰动证据</p>
        <table id="controlTable"><thead><tr><th>control</th><th>teacher acc</th><th>student acc</th><th>teacher drop</th><th>student drop</th><th>similarity</th></tr></thead><tbody></tbody></table>
        <div style="height:14px"></div>
        <p class="panel-title">思维链五维评分</p>
        <table id="cotTable"><thead><tr><th>dimension</th><th>raw (1-5)</th><th>normalized</th><th>weight</th><th>contribution</th></tr></thead><tbody></tbody></table>
        <div style="height:14px"></div>
        <p class="panel-title">思维链对比诊断</p>
        <table id="reasonTable"><thead><tr><th>Judged N</th><th>Stage1 Reason</th><th>Stage2 Reason</th><th>Delta</th><th>Stage2 Win</th></tr></thead><tbody></tbody></table>
      </section>
    </main>
    <button class="assistant-orb" id="assistantOrb" type="button" aria-label="打开 AI 助手" title="AI 助手">
      <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <path d="M5 8.5A4.5 4.5 0 0 1 9.5 4h10A4.5 4.5 0 0 1 24 8.5v6A4.5 4.5 0 0 1 19.5 19h-5.8l-5.7 4 .9-4.2A4.5 4.5 0 0 1 5 14.5v-6Z" fill="white"/>
        <circle cx="10.5" cy="12" r="1.25" fill="#2563eb"/><circle cx="15" cy="12" r="1.25" fill="#2563eb"/><circle cx="19.5" cy="12" r="1.25" fill="#2563eb"/>
        <rect x="17.5" y="20" width="10.5" height="6.5" rx="3.25" fill="white"/>
        <text x="22.75" y="24.65" text-anchor="middle" fill="#2563eb" font-family="Arial, sans-serif" font-size="4.8" font-weight="800">AI</text>
        <path d="M27 3v4M25 5h4" stroke="white" stroke-width="1.7" stroke-linecap="round"/>
      </svg>
    </button>
    <section class="assistant-panel" id="assistantPanel" aria-label="AI 助手">
      <div class="assistant-head">
        <span class="assistant-icon">AI</span>
        <strong>平台 AI 助手</strong>
        <small id="assistantModelLabel">未配置</small>
        <button class="assistant-icon-btn" id="assistantSettingsBtn" type="button" title="助手设置">⚙</button>
        <button class="assistant-icon-btn" id="assistantCloseBtn" type="button" title="关闭">×</button>
      </div>
      <div class="assistant-settings" id="assistantSettings">
        <div class="field"><label>Base URL</label><input data-console="assistant_api_base" placeholder="https://api.openai.com/v1"></div>
        <div class="field"><label>模型</label><input data-console="assistant_model" placeholder="gpt-4o-mini"></div>
        <div class="field"><label>API Key</label><input type="password" data-console="assistant_api_key" placeholder="sk-..."></div>
      </div>
      <div class="assistant-messages" id="assistantMessages"><div class="assistant-msg assistant">你好，我可以帮你解答平台功能、参数含义、运行流程和常见报错。请直接描述你的问题。</div></div>
      <form class="assistant-compose" id="assistantForm">
        <textarea id="assistantInput" rows="1" placeholder="询问平台用法或参数..."></textarea>
        <button class="btn primary assistant-send" id="assistantSendBtn" type="submit">发送</button>
      </form>
    </section>
  </div>
  <script>
    const DEBUG = __DEBUG_ENABLED__;
    if (DEBUG) document.body.classList.add("debug");
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));
    let state = { console_vars: {}, form_vars: {} };
    let pollTimer = null;
    let wasRunning = false;
    let historyId = null;          // non-null => viewing an archived run
    let historyItems = [];
    let historyDir = "";
    let suppressSave = false;      // true while programmatically filling inputs
    const assistantHistory = [];
    const metricCache = {};
    const REDUCED_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const SIDEBAR_STORAGE_KEY = "mllm-console-sidebar-collapsed";
    const SIDEBAR_SCROLL_STORAGE_KEY = "mllm-console-sidebar-scroll-top";
    function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
    // cosmetic per-line tint for the log terminal
    function logLineClass(line) {
      if (/error|failed|exception|traceback|fatal|错误|失败/i.test(line)) return " log-err";
      if (/warn|警告/i.test(line)) return " log-warn";
      if (/success|completed|finished|\bdone\b|完成|成功/i.test(line)) return " log-ok";
      if (/\d{1,3}%\s*\||\|\s*\d+\s*\/\s*\d+|\bepoch\s*\d|\bperiod\s*\d/i.test(line)) return " log-prog";
      return "";
    }
    // tween a numeric metric from its previous value; falls back to a plain set
    // for non-numeric values (e.g. "-") or when the user prefers reduced motion
    function tweenMetric(el, fromText, toText) {
      if (el.__tween) cancelAnimationFrame(el.__tween);
      const m = String(toText).match(/^(-?\d+(?:\.\d+)?)(.*)$/);
      const from = parseFloat(fromText);
      const to = m ? parseFloat(m[1]) : NaN;
      if (REDUCED_MOTION || !m || isNaN(from) || isNaN(to) || from === to) { el.textContent = toText; return; }
      const suffix = m[2] || "";
      const decimals = (m[1].split(".")[1] || "").length;
      const start = performance.now(), dur = 600;
      const step = (now) => {
        const t = Math.min(1, (now - start) / dur);
        if (t >= 1) { el.textContent = toText; el.__tween = null; return; }
        const eased = 1 - Math.pow(1 - t, 3);
        el.textContent = (from + (to - from) * eased).toFixed(decimals) + suffix;
        el.__tween = requestAnimationFrame(step);
      };
      el.__tween = requestAnimationFrame(step);
    }
    let lastLogSig = null;
    const consoleGroups = {
      globalFields: ["root_dir","reason_judge_dir","vla_mark_dir","python_bin","model_path","dataset_name","dataset_path","cuda_devices"],
      teacherFields: ["teacher_api_key","teacher_api_base","victim_model"],
      debugFields: ["sim_duration"],
      runFields: ["auto_refresh"],
      watermarkFields: ["watermark_python_bin","watermark_model_name","watermark_torch_dtype","watermark_device","watermark_sample_fraction","wm_base_before_score","wm_extracted_score","wm_test_score"]
    };
    const formGroups = {
      teacherCollectFields: ["SCIENCEQA_SPLIT","TRAIN_NUM","MAX_SAMPLES","SCIENCEQA_SEED","TEACHER_LANG","TEACHER_ENABLE_THINKING","COLLECT_TEACHER_DATA","STRICT_TEACHER_DISTILL","NUM_WORKERS"],
      teacherEvalFields: ["MAX_NEW_TOKENS","MAX_CONCURRENCY","SCIENCEQA_CONTROL_SPLIT","SCIENCEQA_CONTROL_MAX_SAMPLES","SCIENCEQA_CONTROLS","TEACHER_RESULT_DIR"],
      trainPathFields: ["STAGE1_CKPT_PATH","STAGE2_FINAL_ADAPTER_PATH"],
      stage1Fields: ["STAGE1_EPOCHS","STAGE1_BATCH_SIZE","STAGE1_GRAD_ACCUM","STAGE1_LR","STAGE1_MAX_LENGTH","USE_4BIT","FREEZE_VISION_TOWER","LORA_RANK","LORA_ALPHA","STAGE1_FIELD_WEIGHT_REASONING","STAGE1_FIELD_WEIGHT_ANSWER"],
      stage2Fields: ["STAGE2_EPOCHS","PERIOD_NUM","STAGE2_GRAD_ACCUM","STAGE2_LR","TAU1","STAGE2_MAX_LENGTH","PHASE_A_BATCH_SIZE","PHASE_B_BATCH_SIZE","STAGE2_EVAL_EVERY_PERIOD","STAGE2_EVAL_TRAIN_NUM","STAGE2_EVAL_MAX_SAMPLES","EVAL_MAX_SAMPLES","FREEZE_VISION_TOWER","LORA_RANK","LORA_ALPHA","USE_4BIT","STAGE2_WRONG_IMAGE_ENABLE","STAGE2_PAIR_USE_ANSWER_CORRECTNESS"],
      studentEvalFields: ["ADAPTER_PATH","VQ_CODEBOOK_PATH","RESULT_DIR","EVAL_MAX_NEW_TOKENS"],
      reasonFields: ["STAGE2","STAGE3","TEACHER","DATASET","OUT_DIR","JUDGE_MODEL","JUDGE_API_BASE","JUDGE_API_KEY","SAMPLE_NUM","JUDGE_DATASET_NAME","SPLIT","REQUIRE_VALID_FORMAT"]
    };
    const fieldLabels = {
      wm_base_before_score: "Watermarked victim score (WM_V)",
      wm_extracted_score: "Extracted student score (WM_E)",
      wm_test_score: "Clean baseline score (WM_C, optional)"
    };
    function fieldHtml(key, scope) {
      const type = key.includes("API_KEY") || key === "teacher_api_key" || key === "judge_api_key" ? "password" : "text";
      return `<div class="field"><label>${fieldLabels[key] || key}</label><input type="${type}" data-${scope}="${key}"></div>`;
    }
    function buildForms() {
      for (const [id, keys] of Object.entries(consoleGroups)) {
        const node = document.getElementById(id);
        if (!node) continue;
        const title = id === "globalFields" ? "<p class='section-title'>全局路径</p>" : id === "teacherFields" ? "<p class='section-title'>教师模型连接</p>" : id === "debugFields" ? "<p class='section-title'>仿真控制</p>" : id === "runFields" ? "<p class='section-title'>运行控制</p>" : "";
        node.innerHTML = title + keys.map(k => fieldHtml(k, "console")).join("");
      }
      for (const [id, keys] of Object.entries(formGroups)) {
        const node = document.getElementById(id);
        if (!node) continue;
        node.innerHTML = keys.map(k => fieldHtml(k, "form")).join("");
      }
    }
    function collectPayload() {
      const payload = { connection_command: $('[data-root="connection_command"]').value, project_path: $('[data-root="project_path"]').value, password: $('[data-root="password"]').value, console_vars: {}, form_vars: {} };
      $$("[data-console]").forEach(el => payload.console_vars[el.dataset.console] = el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value);
      $$("[data-form]").forEach(el => payload.form_vars[el.dataset.form] = el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value);
      return payload;
    }
    function setInputValue(el, value) {
      if (el.type === "checkbox") el.checked = ["1", "true", "yes", "on"].includes(String(value ?? "").toLowerCase());
      else el.value = value ?? "";
    }
    async function post(url, payload = collectPayload()) {
      const res = await fetch(url, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload) });
      return await res.json();
    }
    async function saveConfig() {
      if (suppressSave || historyId) return;   // never let an archive overwrite live config
      const data = await post("/api/config");
      if (data.config) state = data.config;
    }
    async function saveAssistantConfig() {
      const data = await post("/api/config", collectPayload());
      if (data.config) {
        state = data.config;
        const model = data.config.console_vars?.assistant_model || "未配置";
        $("#assistantModelLabel")?.replaceChildren(document.createTextNode(model));
      }
      return data;
    }
    function applyConfig(data) {
      state = data;
      $("#modeBadge").textContent = data.debug ? "DEBUG" : "LIVE";
      $('[data-root="connection_command"]').value = data.connection_command || "";
      $('[data-root="project_path"]').value = data.project_path || "";
      $$("[data-console]").forEach(el => setInputValue(el, data.console_vars?.[el.dataset.console]));
      $$("[data-form]").forEach(el => setInputValue(el, data.form_vars?.[el.dataset.form]));
      const assistantModel = data.console_vars?.assistant_model || "未配置";
      $("#assistantModelLabel")?.replaceChildren(document.createTextNode(assistantModel));
      renderPipeline(data.pipeline || {});
    }
    function appendAssistantMessage(role, content) {
      const box = $("#assistantMessages");
      const message = document.createElement("div");
      message.className = `assistant-msg ${role}`;
      message[role === "assistant" ? "innerHTML" : "textContent"] = role === "assistant" ? renderAssistantMarkdown(content) : content;
      box.appendChild(message);
      box.scrollTop = box.scrollHeight;
    }
    function renderAssistantMarkdown(source) {
      let text = escapeHtml(String(source || ""));
      const codeBlocks = [];
      text = text.replace(/```(?:[a-zA-Z0-9_+-]+)?\n?([\s\S]*?)```/g, (_, code) => {
        const token = `@@ASSISTANT_CODE_${codeBlocks.length}@@`;
        codeBlocks.push(`<pre><code>${code.trimEnd()}</code></pre>`);
        return token;
      });
      text = text.replace(/^###\s+(.+)$/gm, "<strong>$1</strong>");
      text = text.replace(/^##\s+(.+)$/gm, "<strong>$1</strong>");
      text = text.replace(/^#\s+(.+)$/gm, "<strong>$1</strong>");
      text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      text = text.replace(/`([^`\n]+)`/g, "<code>$1</code>");
      text = text.replace(/^\s*[-*]\s+(.+)$/gm, "<li>$1</li>");
      text = text.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");
      text = text.replace(/\n/g, "<br>");
      codeBlocks.forEach((block, index) => { text = text.replace(`@@ASSISTANT_CODE_${index}@@`, block); });
      return text;
    }
    async function sendAssistantMessage() {
      const input = $("#assistantInput");
      const send = $("#assistantSendBtn");
      const content = input.value.trim();
      if (!content || send.disabled) return;
      assistantHistory.push({ role: "user", content });
      appendAssistantMessage("user", content);
      input.value = "";
      send.disabled = true;
      try {
        await saveAssistantConfig();
        const response = await fetch("/api/assistant/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: assistantHistory }) });
        const data = await response.json();
        if (!data.ok) throw new Error(data.message || "助手请求失败");
        assistantHistory.push({ role: "assistant", content: data.message });
        appendAssistantMessage("assistant", data.message);
      } catch (error) {
        appendAssistantMessage("assistant", `请求失败：${error.message || error}`);
      } finally {
        send.disabled = false;
        input.focus();
      }
    }
    function wireAssistant() {
      const orb = $("#assistantOrb");
      const panel = $("#assistantPanel");
      let drag = null;
      const setAssistantOpen = open => {
        panel.classList.toggle("open", open);
        orb.classList.toggle("assistant-hidden", open);
        orb.setAttribute("aria-hidden", String(open));
      };
      orb.addEventListener("pointerdown", event => {
        const rect = orb.getBoundingClientRect();
        drag = { id: event.pointerId, startX: event.clientX, startY: event.clientY, moved: false, offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top };
        orb.setPointerCapture(event.pointerId);
        orb.classList.add("dragging");
      });
      orb.addEventListener("pointermove", event => {
        if (!drag || drag.id !== event.pointerId) return;
        const dx = event.clientX - drag.startX;
        const dy = event.clientY - drag.startY;
        if (Math.hypot(dx, dy) > 5) drag.moved = true;
        if (!drag.moved) return;
        const left = Math.max(8, Math.min(window.innerWidth - orb.offsetWidth - 8, event.clientX - drag.offsetX));
        const top = Math.max(8, Math.min(window.innerHeight - orb.offsetHeight - 8, event.clientY - drag.offsetY));
        orb.style.left = left + "px";
        orb.style.top = top + "px";
        orb.style.right = "auto";
        orb.style.bottom = "auto";
      });
      orb.addEventListener("pointerup", event => {
        if (!drag || drag.id !== event.pointerId) return;
        const moved = drag.moved;
        orb.releasePointerCapture(event.pointerId);
        orb.classList.remove("dragging");
        if (moved) {
          orb.style.left = Math.max(8, window.innerWidth - orb.offsetWidth - 18) + "px";
          orb.style.top = Math.max(8, Math.min(window.innerHeight - orb.offsetHeight - 8, orb.getBoundingClientRect().top)) + "px";
        } else {
          const open = !panel.classList.contains("open");
          setAssistantOpen(open);
          if (open) $("#assistantInput").focus();
        }
        drag = null;
      });
      $("#assistantCloseBtn").addEventListener("click", () => setAssistantOpen(false));
      $("#assistantSettingsBtn").addEventListener("click", () => $("#assistantSettings").classList.toggle("open"));
      $("#assistantForm").addEventListener("submit", event => { event.preventDefault(); sendAssistantMessage(); });
      $("#assistantInput").addEventListener("keydown", event => {
        if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendAssistantMessage(); }
      });
      $("#assistantSettings").addEventListener("change", saveAssistantConfig);
      $("#assistantSettings").addEventListener("input", () => {
        clearTimeout(window.__assistantSaveTimer);
        window.__assistantSaveTimer = setTimeout(saveAssistantConfig, 500);
      });
    }
    function parseProgress(v) {
      if (v === null || v === undefined) return null;
      const s = String(v).trim();
      let m = s.match(/^(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/);
      if (m && Number(m[2]) > 0) return Math.max(0, Math.min(100, (Number(m[1]) / Number(m[2])) * 100));
      m = s.match(/^(\d+(?:\.\d+)?)\s*%$/);
      if (m) return Math.max(0, Math.min(100, Number(m[1])));
      m = s.match(/^(\d+(?:\.\d+)?)$/);
      if (m) { const n = Number(m[1]); return n <= 1 ? n * 100 : Math.min(100, n); }
      return null;
    }
    // Smooth per-stage progress (client-side only; backend reports 0/100 for real tasks).
    // Training stages run many epochs/periods where each tqdm bar RESETS every epoch, so we
    // derive overall progress from epoch/period COUNTERS in the log instead of a single bar
    // (which is why the old "max of all bars" locked at 99%). Single-pass stages (collection /
    // eval / judge) track the latest tqdm value, which naturally climbs 0 -> 100 once.
    const stageRun = {};
    function clampPct(v) { return Math.max(0, Math.min(100, v)); }
    // Setup/noise progress bars that appear BEFORE (or alongside) the real work loop and would
    // otherwise hijack the bar: model weight loading, dataset download/caching, HF map/filter, etc.
    // The real work loop (e.g. "detect vq_lord result watermark: X/N", "ScienceQA Eval: X/N")
    // is what we want to track, so any tqdm line matching these is ignored for progress.
    const SETUP_BAR_RE = /(loading checkpoint shards|loading weights|checkpoint shards|downloading|fetching|resolving data|generating (?:train|test|validation|split)|casting|tokeniz|^\s*map:|\bmap\s*:|\bfilter\s*:|loading (?:dataset|cached|builder)|extracting|preprocess|构建|加载(?:模型|检查点|权重|缓存|数据集))/i;
    function isSetupBar(line) { return SETUP_BAR_RE.test(line || ""); }
    function ensureState(row, logs) {
      let st = stageRun[row.id];
      if (!st) st = stageRun[row.id] = { startTs: Date.now(), startLogLen: logs.length, pct: 0 };
      if (st.startLogLen > logs.length) st.startLogLen = 0;   // logs were trimmed server-side
      return st;
    }
    function timeCreep(st, cap, tau) {
      const elapsed = (Date.now() - st.startTs) / 1000;
      return cap * (1 - Math.exp(-elapsed / (tau || 45)));
    }
    function parseBarPct(line) {                        // one tqdm/log line -> 0..100 or null
      if (!line) return null;
      let m = line.match(/(\d{1,3})%\s*\|/);            // "35%|###"
      if (m) return clampPct(+m[1]);
      m = line.match(/\|\s*(\d+)\s*\/\s*(\d+)/);         // "| 12/34"
      if (m && +m[2] > 0) return clampPct(+m[1] / +m[2] * 100);
      m = line.match(/\b(\d+)\s*\/\s*(\d+)\b/);          // bare "12/34"
      if (m && +m[2] > 0) return clampPct(+m[1] / +m[2] * 100);
      m = line.match(/\b(\d{1,3}(?:\.\d+)?)\s*%/);       // generic "35%"
      if (m) { const v = +m[1]; if (v >= 0 && v <= 100) return v; }
      return null;
    }
    function latestBarPct(logs, fromIdx) {              // latest REAL work-loop bar, skipping setup noise
      for (let i = logs.length - 1; i >= (fromIdx || 0); i--) {
        if (isSetupBar(logs[i])) continue;
        const v = parseBarPct(logs[i]);
        if (v !== null) return v;
      }
      return null;
    }
    function sawSetupBar(logs, fromIdx) {               // model still loading -> show warm-up, not the 100% shard bar
      for (let i = logs.length - 1; i >= (fromIdx || 0); i--) {
        if (isSetupBar(logs[i]) && parseBarPct(logs[i]) !== null) return true;
      }
      return false;
    }
    function lastNum(logs, re) {
      for (let i = logs.length - 1; i >= 0; i--) { const m = logs[i].match(re); if (m) return +m[1]; }
      return null;
    }
    function maxNum(logs, re, fromIdx) {
      let best = null;
      for (let i = (fromIdx || 0); i < logs.length; i++) { const m = logs[i].match(re); if (m) best = Math.max(best ?? -Infinity, +m[1]); }
      return best;
    }
    function lastPair(logs, re) {
      for (let i = logs.length - 1; i >= 0; i--) { const m = logs[i].match(re); if (m) return [+m[1], +m[2]]; }
      return null;
    }
    function innerFracForDesc(logs, descRe, fromIdx) {   // inner tqdm fraction of the most recent matching line
      for (let i = logs.length - 1; i >= (fromIdx || 0); i--) {
        if (!descRe.test(logs[i])) continue;
        const v = parseBarPct(logs[i]);
        return v === null ? null : v / 100;
      }
      return null;
    }
    const TRAIN_STAGES = {
      stage1_train: {
        descRe: /Stage1 Epoch\s*(\d+)/i,
        resolve(logs, fromIdx) {
          const total = lastNum(logs, /EPOCHS:\s*(\d+)/i);       // script echo "EPOCHS: 100"
          const cur = maxNum(logs, /Stage1 Epoch\s*(\d+)/i, fromIdx);
          return (total && cur) ? { cur, total, label: `Epoch ${cur}/${total}` } : null;
        }
      },
      stage2_train: {
        descRe: /Stage2 S\d+ P\d+/i,
        resolve(logs) {
          const pr = lastPair(logs, /period\s*(\d+)\s*\/\s*(\d+)/i);   // "进入 period 3/10"
          return pr ? { cur: pr[0], total: pr[1], label: `Period ${pr[0]}/${pr[1]}` } : null;
        }
      }
    };
    function stageDisplay(row, logs) {
      if (row.status === "success") { delete stageRun[row.id]; return { pct: 100, label: "100%", running: false }; }
      if (row.status !== "running") {
        const raw = parseProgress(row.progress);
        if (row.status === "failed") { const p = stageRun[row.id]?.pct ?? raw; delete stageRun[row.id]; return { pct: p, label: p != null ? `${Math.round(p)}%` : (row.progress || "-"), running: false }; }
        delete stageRun[row.id];
        return { pct: raw, label: (row.progress ?? "") === "" ? "-" : row.progress, running: false };
      }
      // running --------------------------------------------------------------
      const raw = parseProgress(row.progress);
      if (raw !== null && raw > 0 && raw < 100) return { pct: raw, label: row.progress, running: true };  // debug sim is already smooth
      const st = ensureState(row, logs);
      const train = TRAIN_STAGES[row.id];
      if (train) {
        const info = train.resolve(logs, st.startLogLen);
        if (info) {
          const inner = innerFracForDesc(logs, train.descRe, st.startLogLen);
          const frac = inner !== null
            ? ((info.cur - 1) + inner) / info.total
            : ((info.cur - 1) + timeCreep(st, 0.9)) / info.total;   // within-epoch crawl if no inner bar this tick
          st.pct = clampPct(Math.min(99, frac * 100));
          return { pct: st.pct, label: info.label, running: true };
        }
        st.pct = Math.max(st.pct, timeCreep(st, 8));   // counters not printed yet -> gentle warm-up
        return { pct: st.pct, label: "启动中", running: true };
      }
      // single-pass stage: latest REAL work-loop bar (setup/loading bars are skipped),
      // else warm-up while the model is still loading, else a generic time estimate.
      const bar = latestBarPct(logs, st.startLogLen);
      if (bar !== null) { st.pct = Math.max(st.pct, Math.min(99, bar)); return { pct: st.pct, label: `${Math.round(st.pct)}%`, running: true }; }
      if (sawSetupBar(logs, st.startLogLen)) { st.pct = Math.max(st.pct, timeCreep(st, 12)); return { pct: st.pct, label: "加载模型…", running: true }; }
      st.pct = Math.max(st.pct, timeCreep(st, 90));
      return { pct: st.pct, label: `~${Math.round(st.pct)}%`, running: true };
    }
    function progressCell(row, logs) {
      const d = stageDisplay(row, logs);
      if (d.pct === null || d.pct === undefined) return `<span class="mini-label">${d.label}</span>`;
      const run = d.running ? " run" : "";
      return `<div class="mini"><div class="mini-track"><div class="mini-fill${run}" style="width:${d.pct.toFixed(0)}%"></div></div><span class="mini-label">${d.label}</span></div>`;
    }
    function renderPipeline(p) {
      // an in-flight poll must not repaint over an archive we just loaded
      if (historyId && !p.frozen) return;
      $("#pipelineSummary").textContent = p.summary || "暂无任务";
      $("#statusText").textContent = p.summary || "准备就绪";
      const rows = p.rows || [];
      const logs = p.logs || [];
      const cells = rows.map(row => progressCell(row, logs));
      // top bar: completed stages + fractional progress of the running stage
      const total = rows.length || 1;
      const done = rows.filter(r => r.status === "success").length;
      let frac = done / total;
      const runningRow = rows.find(r => r.status === "running");
      if (runningRow && stageRun[runningRow.id]) frac = Math.max(frac, (done + stageRun[runningRow.id].pct / 100) / total);
      frac = Math.max(frac, p.progress || 0);
      const pct = Math.round(frac * 100);
      $("#pipelineProgress").style.width = `${pct}%`;
      const pctEl = $("#pipelinePercent");
      if (pctEl) pctEl.textContent = `${pct}%`;
      const running = rows.some(r => r.status === "running");
      document.body.classList.toggle("is-busy", running);
      const wrap = $("#pipelineProgress").parentElement;
      if (wrap) wrap.classList.toggle("is-running", running);
      const tbody = $("#taskTable tbody");
      tbody.innerHTML = rows.map((row, i) => `<tr${row.status === "running" ? ' class="row-running"' : ""}><td>${row.stage}</td><td><span class="pill status-${row.status}"><i></i>${row.status}</span></td><td>${cells[i]}</td><td>${row.script}</td></tr>`).join("");
      const logBox = $("#logBox");
      const scroller = logBox.closest(".logbox") || logBox;
      const lines = logs.length ? logs : ["No logs yet."];
      const sig = lines.join("\n");
      if (sig !== lastLogSig) {
        lastLogSig = sig;
        const first = scroller.scrollTop;
        const atBottom = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 12;
        logBox.innerHTML = lines.map(l => `<span class="log-line${logLineClass(l)}">${escapeHtml(l)}</span>`).join("");
        if (atBottom) scroller.scrollTop = scroller.scrollHeight;
        else scroller.scrollTop = first;
      }
      if (p.frozen) return;        // archived snapshot: no live-transition bookkeeping
      // a run just finished -> pull fresh numbers so the cards stop showing "-"
      if (wasRunning && !running) { refreshDashboard(); refreshWatermark(); loadHistoryList(); }
      wasRunning = running;
    }
    async function refreshStatus() {
      const res = await fetch("/api/tasks/status");
      renderPipeline(await res.json());
    }
    function setMetric(id, value) {
      const el = $(id);
      const v = value || "-";
      const prev = metricCache[id];
      if (prev !== undefined && prev !== v && v !== "-") {
        const card = el.closest(".metric");
        if (card) { card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash"); }
      }
      metricCache[id] = v;
      tweenMetric(el, prev, v);
    }
    async function refreshDashboard() {
      if (historyId) return;                 // viewing an archive: keep it on screen
      renderDashboard(await (await fetch("/api/dashboard")).json());
    }
    function renderDashboard(data) {
      const m = data.metrics || {};
      const levelLabels = {critical:"极高", high:"高", medium:"中", low:"低", not_measured:"未测量"};
      setMetric("#metricDefense", m.defense);
      setMetric("#metricRisk", m.clr || m.risk);
      setMetric("#metricRiskLevel", levelLabels[m.risk_level] || m.risk_level);
      setMetric("#metricCoverage", m.coverage);
      setMetric("#metricAcc", m.acc);
      setMetric("#metricVr", m.vr);
      setMetric("#metricCot", m.cot);
      setMetric("#metricWer", m.wer);
      $("#riskOriginAcc").textContent = m.origin_acc || "-";
      $("#riskTeacherAcc").textContent = m.teacher_acc || "-";
      $("#riskStudentAcc").textContent = m.student_acc || "-";
      $("#riskConfidence").textContent = m.confidence || "-";
      $("#riskWerLevel").textContent = levelLabels[m.wer_level] || m.wer_level || "未测量";
      const levelEl = $("#metricRiskLevel");
      levelEl.closest(".metric").className = `metric risk-level-${m.risk_level || "not_measured"}`;
      const evidence = data.evidence || {};
      const missing = (evidence.missing_dimensions || []).join(", ") || "none";
      $("#riskEvidenceNote").textContent = `ACC: ${evidence.acc_source || "missing"} · VR: ${evidence.vr_source || "missing"} · CoT: ${evidence.cot_source || "missing"} · missing: ${missing}`;
      $("#controlTable tbody").innerHTML = (data.controls || []).map(r => `<tr><td>${escapeHtml(r.control)}</td><td>${escapeHtml(r.teacher)}</td><td>${escapeHtml(r.student)}</td><td>${escapeHtml(r.teacher_drop || "-")}</td><td>${escapeHtml(r.student_drop || "-")}</td><td>${escapeHtml(r.similarity || "-")}</td></tr>`).join("") || `<tr><td>暂无结果</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>`;
      $("#cotTable tbody").innerHTML = (data.cot_dimensions || []).map(r => `<tr><td>${escapeHtml(r.dimension)}</td><td>${escapeHtml(r.raw)}</td><td>${escapeHtml(r.normalized)}</td><td>${escapeHtml(r.weight)}</td><td>${escapeHtml(r.contribution)}</td></tr>`).join("") || `<tr><td>暂无结果</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>`;
      const reason = data.reason || {};
      $("#reasonTable tbody").innerHTML = `<tr><td>${escapeHtml(reason.n || "-")}</td><td>${escapeHtml(reason.stage1_reason || "-")}</td><td>${escapeHtml(reason.stage2_reason || "-")}</td><td>${escapeHtml(reason.delta || "-")}</td><td>${escapeHtml(reason.stage2_win || "-")}</td></tr>`;
    }
    let wmMeanCache = null;
    async function refreshWatermark() {
      if (historyId) return;                 // viewing an archive: keep it on screen
      let data;
      try { data = await (await fetch("/api/watermark")).json(); } catch (e) { return; }
      renderWatermark(data);
    }
    function renderWatermark(data) {
      const m = data.metrics || {};
      const meanEl = $("#wmMeanZ");
      if (meanEl) {
        const prevMean = wmMeanCache;
        if (prevMean !== null && prevMean !== m.mean_z && (m.mean_z || "-") !== "-") {
          meanEl.classList.remove("flash"); void meanEl.offsetWidth; meanEl.classList.add("flash");
        }
        wmMeanCache = m.mean_z;
        tweenMetric(meanEl, prevMean, m.mean_z || "-");
      }
      $("#wmMedianZ").textContent = m.median_z || "-";
      $("#wmMinZ").textContent = m.min_z || "-";
      $("#wmMaxZ").textContent = m.max_z || "-";
      $("#wmRate").textContent = m.threshold_4_rate || "-";
      // gauge: map z-mean onto 0..8 (detection threshold is 4), fill the ring
      const gauge = $("#wmGauge");
      const meanZ = parseFloat(m.mean_z);
      if (gauge) {
        if (!isNaN(meanZ)) {
          const frac = Math.max(0, Math.min(1, meanZ / 8));
          gauge.style.setProperty("--deg", `${(frac * 360).toFixed(1)}deg`);
          gauge.style.background = `conic-gradient(${meanZ > 4 ? "#34d399, var(--ok)" : "var(--accent-3), var(--accent-2)"} var(--deg,0deg), var(--surface-3) 0deg)`;
          gauge.querySelector("span").textContent = meanZ > 4 ? "有水印" : "偏弱";
        } else {
          gauge.style.setProperty("--deg", "0deg");
          gauge.querySelector("span").textContent = "z / 8";
        }
      }
      const scope = $("#wmScope");
      if (scope) scope.textContent = data.ok ? `已评分 ${data.num_scored ?? "?"} / ${data.num_total ?? "?"} 条 · split=${data.split ?? "?"}` : "尚未运行检测";
      // results are in -> stop the 3s catch-up poll started by the detect button
      if (data.ok && window.__wmPoll) { clearInterval(window.__wmPoll); window.__wmPoll = null; }
    }
    function startPolling() {
      if (pollTimer) clearInterval(pollTimer);
      if (historyId) return;                 // frozen while viewing an archive
      pollTimer = setInterval(refreshStatus, 1000);
    }
    function stopPolling() {
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      if (window.__wmPoll) { clearInterval(window.__wmPoll); window.__wmPoll = null; }
    }
    // ---- history archive -----------------------------------------------------
    function historyOptionLabel(it) {
      const risk = (it.risk && it.risk !== "-") ? ` · risk ${it.risk}` : "";
      return `${it.label}${risk}`;
    }
    function renderHistoryOptions(items, keep) {
      historyItems = items || [];
      const sel = $("#historySelect");
      sel.innerHTML = `<option value="">当前数据（实时）</option>` +
        historyItems.map(it => `<option value="${it.id}">${escapeHtml(historyOptionLabel(it))}</option>`).join("");
      sel.value = keep && historyItems.some(it => it.id === keep) ? keep : "";
    }
    async function loadHistoryList(keep) {
      try {
        const data = await (await fetch("/api/history")).json();
        renderHistoryOptions(data.items, keep ?? historyId);
        historyDir = data.dir || "";
      } catch (e) { /* keep whatever is on screen */ }
    }
    function setHistoryHint(text) { const el = $("#historyHint"); if (el) el.textContent = text; }
    // Running a task while an archive is on screen would execute against live config,
    // not the archived params -> lock the action buttons until we exit history mode.
    function setActionsLocked(locked) {
      $$("[data-task], #runPipelineBtn, #cloneBtn, #loadRemoteBtn, #saveRemoteBtn").forEach(b => { b.disabled = locked; });
    }
    async function enterHistory(id) {
      const rec = await (await fetch(`/api/history/${encodeURIComponent(id)}`)).json();
      if (!rec.ok) { setHistoryHint(rec.message || "读取历史记录失败"); $("#historySelect").value = historyId || ""; return; }
      historyId = id;
      stopPolling();
      document.body.classList.add("history-view");
      setActionsLocked(true);
      suppressSave = true;                   // filling inputs must not overwrite live config
      $$("[data-console]").forEach(el => setInputValue(el, rec.console_vars?.[el.dataset.console]));
      $$("[data-form]").forEach(el => setInputValue(el, rec.form_vars?.[el.dataset.form]));
      suppressSave = false;
      renderDashboard(rec.dashboard || {});
      renderWatermark(rec.watermark || {});
      const p = rec.pipeline || {};
      renderPipeline({ summary: p.summary, progress: p.progress, rows: p.rows, logs: p.logs, frozen: true });
      $("#modeBadge").textContent = rec.debug ? "ARCHIVE · SIM" : "ARCHIVE";
      $("#statusText").textContent = `历史回看：${rec.label || id}`;
      setHistoryHint(`已载入 ${rec.created_at || id} 的归档，参数与分数均为当时快照；运行按钮已锁定`);
    }
    async function exitHistory() {
      historyId = null;
      document.body.classList.remove("history-view");
      setActionsLocked(false);
      $("#historySelect").value = "";
      suppressSave = true;
      applyConfig(await (await fetch("/api/config")).json());
      suppressSave = false;
      await refreshDashboard();
      await refreshWatermark();
      startPolling();
      setHistoryHint(historyDir ? `已回到实时数据 · 归档目录 ${historyDir}` : "已回到实时数据");
    }
    function wireEvents() {
      const appShell = $(".app");
      const sidebar = $(".sidebar");
      const sidebarToggle = $("#sidebarToggle");
      let sidebarScrollTop = 0;
      try { sidebarScrollTop = Math.max(0, Number(localStorage.getItem(SIDEBAR_SCROLL_STORAGE_KEY)) || 0); } catch (error) { /* use top */ }
      const setSidebarCollapsed = (collapsed, initializing = false) => {
        if (collapsed && !initializing) {
          sidebarScrollTop = sidebar.scrollTop;
          try { localStorage.setItem(SIDEBAR_SCROLL_STORAGE_KEY, String(sidebarScrollTop)); } catch (error) { /* storage may be disabled */ }
        }
        appShell.classList.toggle("sidebar-collapsed", collapsed);
        const toggleLabel = collapsed ? "展开左侧栏" : "收起左侧栏";
        sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
        sidebarToggle.setAttribute("aria-label", toggleLabel);
        sidebarToggle.title = toggleLabel;
        try { localStorage.setItem(SIDEBAR_STORAGE_KEY, collapsed ? "1" : "0"); } catch (error) { /* storage may be disabled */ }
        requestAnimationFrame(() => { sidebar.scrollTop = sidebarScrollTop; });
      };
      let sidebarCollapsed = false;
      try { sidebarCollapsed = localStorage.getItem(SIDEBAR_STORAGE_KEY) === "1"; } catch (error) { /* use expanded default */ }
      sidebar.scrollTop = sidebarScrollTop;
      setSidebarCollapsed(sidebarCollapsed, true);
      sidebarToggle.addEventListener("click", () => setSidebarCollapsed(!appShell.classList.contains("sidebar-collapsed")));
      $("#tabs").addEventListener("click", (event) => {
        const btn = event.target.closest(".tab");
        if (!btn) return;
        $$(".tab").forEach(x => x.classList.toggle("active", x === btn));
        $$(".pane").forEach(x => x.classList.toggle("active", x.id === btn.dataset.pane));
      });
      $("#configForm").addEventListener("change", saveConfig);
      $("#configForm").addEventListener("input", (event) => {
        const el = event.target;
        if (el?.dataset?.form) {
          $$(`[data-form="${el.dataset.form}"]`).forEach(peer => {
            if (peer !== el) peer.value = el.value;
          });
        }
        clearTimeout(window.__saveTimer);
        window.__saveTimer = setTimeout(saveConfig, 500);
      });
      $("#cloneBtn").addEventListener("click", async () => {
        $("#cloneBtn").disabled = true;
        const data = await post("/api/clone");
        if (data.config) applyConfig(data.config);
        $("#statusText").textContent = data.message || "完成";
        $("#cloneBtn").disabled = false;
      });
      $("#loadRemoteBtn").addEventListener("click", async () => {
        const data = await post("/api/remote/load-config");
        if (data.config) applyConfig(data.config);
        $("#statusText").textContent = data.message || (data.ok ? "已读取远程配置" : "读取远程配置失败");
      });
      $("#saveRemoteBtn").addEventListener("click", async () => {
        const data = await post("/api/remote/save-config");
        $("#statusText").textContent = data.message || (data.ok ? "已保存到服务器" : "保存失败");
      });
      $("#runPipelineBtn").addEventListener("click", async () => {
        const data = await post("/api/pipeline/full");
        if (data.pipeline) renderPipeline(data.pipeline);
        startPolling();
      });
      $$("#refreshStatusBtn, #refreshDashboardBtn, #refreshRiskBtn").forEach(btn => btn.addEventListener("click", () => { refreshStatus(); refreshDashboard(); }));
      const wmBtn = $("#refreshWatermarkBtn");
      if (wmBtn) wmBtn.addEventListener("click", refreshWatermark);
      const logScroller = $(".logbox");
      const logJump = $("#logJumpBtn");
      if (logScroller && logJump) {
        logScroller.addEventListener("scroll", () => {
          const atBottom = logScroller.scrollTop + logScroller.clientHeight >= logScroller.scrollHeight - 24;
          logJump.classList.toggle("show", !atBottom);
        });
        logJump.addEventListener("click", () => { logScroller.scrollTop = logScroller.scrollHeight; });
      }
      $$("[data-task]").forEach(btn => btn.addEventListener("click", async () => {
        const data = await post(`/api/tasks/${btn.dataset.task}`);
        if (data.pipeline) renderPipeline(data.pipeline);
        startPolling();
        if (btn.dataset.task === "watermark_detect") window.__wmPoll = setInterval(refreshWatermark, 3000);
      }));
      $("#historySelect").addEventListener("change", (event) => {
        const id = event.target.value;
        if (id) enterHistory(id); else exitHistory();
      });
      $("#historyBackBtn").addEventListener("click", exitHistory);
      $("#historySaveBtn").addEventListener("click", async () => {
        if (historyId) { setHistoryHint("请先返回默认数据，再归档当前结果"); return; }
        const btn = $("#historySaveBtn");
        btn.disabled = true;
        const data = await post("/api/history", { ...collectPayload(), label: "" });
        btn.disabled = false;
        setHistoryHint(data.message || (data.ok ? "已归档" : "归档失败"));
        if (data.items) renderHistoryOptions(data.items, null); else await loadHistoryList(null);
      });
      $("#historyDeleteBtn").addEventListener("click", async () => {
        const sel = $("#historySelect");
        const id = sel.value || historyId;
        if (!id) { setHistoryHint("请先在上方选择一条归档记录"); return; }
        const item = historyItems.find(it => it.id === id);
        if (!confirm(`删除归档记录「${item ? item.label : id}」？此操作不可撤销。`)) return;
        const res = await fetch(`/api/history/${encodeURIComponent(id)}`, { method: "DELETE" });
        const data = await res.json();
        if (historyId === id) await exitHistory();
        if (data.items) renderHistoryOptions(data.items, null); else await loadHistoryList(null);
        setHistoryHint(data.message || "已删除");
      });
    }
    async function boot() {
      buildForms();
      wireEvents();
      wireAssistant();
      const res = await fetch("/api/config");
      applyConfig(await res.json());
      await refreshDashboard();
      await refreshWatermark();
      await loadHistoryList(null);
      if (historyDir) setHistoryHint(`跑完 Pipeline 会自动归档 · 目录 ${historyDir}`);
      startPolling();
    }
    boot();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    launch_web_app(debug=args.debug, host=args.host, port=args.port, open_browser=not args.no_browser)
