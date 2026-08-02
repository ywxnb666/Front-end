from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import APP_DIR, AppConfig, load_config, save_config
from app.models import CloneRequest
from app.services.git_service import REPO_DIR_NAME, RemoteGitCloneService
from app.services.ssh_client import SSHClientManager, parse_connection_command


PIPELINE_STEPS: list[tuple[str, str, str]] = [
    ("teacher_collect", "教师 API 数据采集", "scripts2/teacher_model_data_collect.sh"),
    ("teacher_eval", "教师完整风险基线", "scripts2/run_full_eval_pipeline_teacher.sh"),
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
        self._refresh_derived_paths()
        self.form_vars = default_form_vars(self.console_vars)
        self.form_vars.update({str(k): str(v) for k, v in self.config.form_vars.items() if str(k) not in OBSOLETE_FORM_KEYS})
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
            self._sync_console_to_form_paths()
            self.save_app_config_locked()
            return {"ok": True, "config": self.snapshot_locked()}

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
        save_config(
            AppConfig(
                connection_command=self.connection_command,
                project_path=self.project_path,
                console_vars=dict(self.console_vars),
                form_vars={k: v for k, v in self.form_vars.items() if k not in OBSOLETE_FORM_KEYS},
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
        judge_key = str(console.get("judge_api_key", "") or teacher_key).strip()
        judge_base = normalize_openai_base_url(str(console.get("judge_api_base", "") or teacher_base))
        env = {
            "PYTHONUNBUFFERED": "1",
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
        protected = {
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
        root_dir = str(console.get("root_dir", "")).strip()
        if script == "dashboard aggregation":
            return None, root_dir, label
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
            env = self.task_env(step_id)
            command, cwd, label = self.resolve_task_command(step_id)
            self.set_real_step_state(step_id, "running", "0%")
            if command is None:
                self.append_log(f"[real] {label}: no external command, marked success")
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
        local_env.update(env)
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
            return self.pipeline_status_locked()

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
            rows.append({"id": step_id, "stage": label, "status": status, "progress": f"{progress * 100:.0f}%", "script": script})
        running = total_progress < 1.0
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
        report = self.read_json_result(output_path)
        metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
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

    def dashboard_payload(self) -> dict[str, Any]:
        try:
            with self.lock:
                result_dir = str(self.console_vars.get("result_dir", ""))
            teacher = self.read_json_result(_path_join(result_dir, "scienceqa_control_suite_teacher_full.json"))
            student = self.read_json_result(_path_join(result_dir, "scienceqa_control_suite_full_fast.json"))
            report = self.read_json_result(_path_join(result_dir, "mm_eval_suite_report_full_fast.json"))
            reason = self.read_tsv_result(_path_join(self.resolve_reason_out_dir(), "summary.tsv"))
            if not student and report:
                student = {"metrics": report.get("control_summary", {})} if isinstance(report, dict) else {}
            return self.format_dashboard(teacher, student, reason)
        except Exception as exc:
            self.append_log(f"[dashboard] refresh failed: {exc}")
            return self.format_dashboard({}, {}, {})

    def format_dashboard(self, teacher: dict[str, Any], student: dict[str, Any], reason: dict[str, str]) -> dict[str, Any]:
        teacher_summary = self.dashboard_control_summary(teacher)
        student_summary = self.dashboard_control_summary(student)
        teacher_acc = self.dashboard_baseline_accuracy(teacher)
        student_acc = self.dashboard_baseline_accuracy(student)
        retention = student_acc / teacher_acc if student_acc is not None and teacher_acc else None
        risk = retention if retention is not None else student_acc
        reason_delta = safe_float(reason.get("delta_reason_score")) if isinstance(reason, dict) else None
        controls = []
        for name in sorted(set(teacher_summary) | set(student_summary)):
            teacher_score = safe_float(teacher_summary.get(name, {}).get("accuracy")) if isinstance(teacher_summary.get(name), dict) else None
            student_score = safe_float(student_summary.get(name, {}).get("accuracy")) if isinstance(student_summary.get(name), dict) else None
            delta = student_score - teacher_score if student_score is not None and teacher_score is not None else None
            controls.append({"control": name, "teacher": fmt_metric(teacher_score), "student": fmt_metric(student_score), "delta": fmt_metric(delta, signed=True)})
        return {
            "metrics": {
                "defense": fmt_metric(1.0 - risk if risk is not None else None),
                "risk": fmt_metric(risk),
                "teacher_acc": fmt_metric(teacher_acc),
                "student_acc": fmt_metric(student_acc),
                "retention": fmt_metric(retention),
                "reason_delta": fmt_metric(reason_delta, signed=True),
            },
            "controls": controls,
            "reason": {
                "n": str(reason.get("n", "-")) if isinstance(reason, dict) else "-",
                "stage1_reason": fmt_metric(safe_float(reason.get("stage2_reason_score")) if isinstance(reason, dict) else None),
                "stage2_reason": fmt_metric(safe_float(reason.get("stage3_reason_score")) if isinstance(reason, dict) else None),
                "delta": fmt_metric(reason_delta, signed=True),
                "stage2_win": fmt_metric(safe_float(reason.get("stage2_win_rate")) if isinstance(reason, dict) else None),
            },
        }

    def dashboard_control_summary(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        metrics = payload.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("control_summary"), dict):
            return metrics["control_summary"]
        control_summary = payload.get("control_summary")
        if isinstance(control_summary, dict):
            return control_summary
        return {}

    def dashboard_baseline_accuracy(self, payload: object) -> float | None:
        if not isinstance(payload, dict):
            return None
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            value = safe_float(metrics.get("baseline_accuracy"))
            if value is not None:
                return value
            summary = metrics.get("control_summary")
            if isinstance(summary, dict) and isinstance(summary.get("baseline"), dict):
                return safe_float(summary["baseline"].get("accuracy"))
        summary = payload.get("control_summary")
        if isinstance(summary, dict) and isinstance(summary.get("baseline"), dict):
            return safe_float(summary["baseline"].get("accuracy"))
        return None


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

    .app { position: relative; z-index: 1; display: grid; grid-template-columns: 372px minmax(0, 1fr); height: 100vh; overflow: hidden; }
    .sidebar {
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(246,249,255,.86));
      backdrop-filter: blur(8px);
      overflow: auto;
      padding: 20px 18px 30px;
      box-shadow: 1px 0 0 rgba(255,255,255,.6), var(--shadow-sm);
    }
    .main { position: relative; overflow: auto; padding: 24px 26px 30px; }

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
    input, textarea {
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
    input:hover, textarea:hover { border-color: var(--line-2); }
    input:focus, textarea:focus { border-color: var(--accent); background: #fff; box-shadow: var(--ring); }
    .field { margin-bottom: 11px; animation: fadeIn .4s ease both; }

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
        linear-gradient(180deg, #ffffff, var(--surface-2));
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
    .hero p { margin: 10px 0 0; color: var(--muted); font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 9px; }
    .hero p::before { content: ""; flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
    .actions {
      display: grid; grid-template-columns: 1fr 150px; gap: 12px; align-content: center;
      border: 1px solid var(--line); border-radius: 18px; padding: 16px;
      background: linear-gradient(180deg, #ffffff, var(--surface-2)); box-shadow: var(--shadow);
      animation: cardRise .55s cubic-bezier(.2,.7,.3,1) .06s both;
    }

    .metrics { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 13px; margin-bottom: 20px; }
    .metric {
      position: relative; overflow: hidden;
      border: 1px solid var(--line); border-radius: 16px;
      background: linear-gradient(180deg, #ffffff, var(--surface-2));
      min-height: 96px; padding: 15px 16px; box-shadow: var(--shadow-sm);
      transition: transform .18s ease, box-shadow .2s ease, border-color .2s ease;
      animation: cardRise .5s cubic-bezier(.2,.7,.3,1) both;
    }
    .metric:nth-child(1){animation-delay:.05s} .metric:nth-child(2){animation-delay:.1s}
    .metric:nth-child(3){animation-delay:.15s} .metric:nth-child(4){animation-delay:.2s}
    .metric:nth-child(5){animation-delay:.25s}
    .metric:hover { transform: translateY(-3px); box-shadow: var(--shadow); border-color: var(--line-2); }
    .metric::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: linear-gradient(var(--accent-3), var(--accent-2)); }
    .metric::after { content: ""; position: absolute; right: -30px; top: -30px; width: 90px; height: 90px; border-radius: 50%; background: radial-gradient(circle, rgba(59,130,246,.12), transparent 70%); }
    .metric span { display: block; color: var(--muted); font-size: 11.5px; font-weight: 600; margin-bottom: 12px; letter-spacing: .3px; }
    .metric strong { font-family: var(--display); font-size: 27px; line-height: 1; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; }
    .metric.flash strong { animation: metricFlash .9s ease; }
    @keyframes metricFlash { 0% { color: var(--accent); transform: scale(1.12); } 100% { color: var(--text); transform: scale(1); } }

    /* watermark z-mean hero card */
    .wm-hero { display: flex; align-items: center; justify-content: space-between; gap: 22px; overflow: hidden; position: relative; }
    .wm-hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 200px; height: 200px; border-radius: 50%; background: radial-gradient(circle, rgba(59,130,246,.14), transparent 70%); }
    .wm-hero-main { display: flex; flex-direction: column; gap: 6px; z-index: 1; }
    .wm-cap { color: var(--muted); font-size: 12.5px; font-weight: 600; letter-spacing: .3px; }
    .wm-hero-main strong { font-family: var(--display); font-size: 52px; line-height: 1; font-weight: 800; color: var(--accent-2); font-variant-numeric: tabular-nums; }
    .wm-hero-main strong.flash { animation: metricFlash .9s ease; }
    .wm-sub { color: var(--dim); font-size: 12px; font-family: var(--mono); }
    .wm-gauge { position: relative; width: 128px; height: 128px; border-radius: 50%; flex: none; z-index: 1;
      background: conic-gradient(var(--accent-3) var(--deg,0deg), var(--surface-3) 0deg);
      transition: --deg .8s cubic-bezier(.3,.8,.3,1); display: grid; place-items: center; }
    .wm-gauge::before { content: ""; position: absolute; inset: 12px; border-radius: 50%; background: linear-gradient(180deg,#fff,var(--surface-2)); box-shadow: inset 0 1px 4px rgba(23,54,110,.12); }
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
    .tab.active::after { transform: scaleX(1); }

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
        linear-gradient(180deg, rgba(37,99,235,.05), transparent 90px),
        #0d1b30;
      border: 1px solid #16345f; border-radius: 14px;
      padding: 0; font-family: var(--mono); font-size: 12px; line-height: 1.6;
      color: #cfe0f7; box-shadow: inset 0 2px 14px rgba(0,0,0,.35);
    }
    .logbox::-webkit-scrollbar-thumb { background: #2c4f80; }
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
    .log-titlebar .log-tag { margin-left: auto; }
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

    @keyframes cardRise { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

    @media (prefers-reduced-motion: reduce) {
      * { animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
    }
    @media (max-width: 1260px) {
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .topbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
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
        <div class="metric"><span>窃取风险</span><strong id="metricRisk">-</strong></div>
        <div class="metric"><span>Teacher Acc</span><strong id="metricTeacher">-</strong></div>
        <div class="metric"><span>Student Acc</span><strong id="metricStudent">-</strong></div>
        <div class="metric"><span>Acc Retention</span><strong id="metricRetention">-</strong></div>
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
          <div class="logbox">
            <div class="log-titlebar"><span class="log-live"></span><span class="log-path">remote://pipeline.log</span><span class="log-tag">stdout · live</span></div>
            <div class="log-body"><div id="logBox">No real tasks yet.</div></div>
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
        <div class="panel"><p class="panel-title">1A. 教师四字段标注采集</p><div class="grid2" id="teacherCollectFields"></div><button class="btn primary" data-task="teacher_collect">开始采集教师数据</button></div>
        <div class="panel"><p class="panel-title">1B. 教师风险基线评测</p><div class="grid2" id="teacherEvalFields"></div><button class="btn primary" data-task="teacher_eval">启动教师完整评测</button></div>
      </section>
      <section id="trainPane" class="pane">
        <div class="panel"><p class="panel-title">路径配置</p><div class="grid2" id="trainPathFields"></div></div>
        <div class="panel"><p class="panel-title">Stage1 训练</p><div class="grid3" id="stage1Fields"></div><button class="btn primary" data-task="stage1_train">启动 Stage1 蒸馏</button></div>
        <div class="panel"><p class="panel-title">Stage2 训练</p><div class="grid3" id="stage2Fields"></div><button class="btn primary" data-task="stage2_train">启动 Stage2 蒸馏</button></div>
      </section>
      <section id="evalPane" class="pane">
        <div class="panel"><p class="panel-title">3A. 学生完整风险评测</p><div class="grid2" id="studentEvalFields"></div><button class="btn primary" data-task="student_eval">启动学生完整评测</button></div>
        <div class="panel"><p class="panel-title">3B. 思维链评估</p><div class="grid2" id="reasonFields"></div><button class="btn primary" data-task="reason_judge">启动思维链评估</button></div>
      </section>
      <section id="riskPane" class="pane">
        <div class="panel"><button class="btn" id="refreshRiskBtn">刷新结果摘要</button></div>
        <table id="controlTable"><thead><tr><th>control</th><th>teacher</th><th>student</th><th>delta</th></tr></thead><tbody></tbody></table>
        <div style="height:14px"></div>
        <table id="reasonTable"><thead><tr><th>Judged N</th><th>Stage1 Reason</th><th>Stage2 Reason</th><th>Delta</th><th>Stage2 Win</th></tr></thead><tbody></tbody></table>
      </section>
    </main>
  </div>
  <script>
    const DEBUG = __DEBUG_ENABLED__;
    if (DEBUG) document.body.classList.add("debug");
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));
    let state = { console_vars: {}, form_vars: {} };
    let pollTimer = null;
    const metricCache = {};
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
    function fieldHtml(key, scope) {
      const type = key.includes("API_KEY") || key === "teacher_api_key" || key === "judge_api_key" ? "password" : "text";
      return `<div class="field"><label>${key}</label><input type="${type}" data-${scope}="${key}"></div>`;
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
      $$("[data-console]").forEach(el => payload.console_vars[el.dataset.console] = el.value);
      $$("[data-form]").forEach(el => payload.form_vars[el.dataset.form] = el.value);
      return payload;
    }
    async function post(url, payload = collectPayload()) {
      const res = await fetch(url, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload) });
      return await res.json();
    }
    async function saveConfig() {
      const data = await post("/api/config");
      if (data.config) state = data.config;
    }
    function applyConfig(data) {
      state = data;
      $("#modeBadge").textContent = data.debug ? "DEBUG" : "LIVE";
      $('[data-root="connection_command"]').value = data.connection_command || "";
      $('[data-root="project_path"]').value = data.project_path || "";
      $$("[data-console]").forEach(el => el.value = data.console_vars?.[el.dataset.console] ?? "");
      $$("[data-form]").forEach(el => el.value = data.form_vars?.[el.dataset.form] ?? "");
      renderPipeline(data.pipeline || {});
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
      const wrap = $("#pipelineProgress").parentElement;
      if (wrap) wrap.classList.toggle("is-running", running);
      const tbody = $("#taskTable tbody");
      tbody.innerHTML = rows.map((row, i) => `<tr><td>${row.stage}</td><td><span class="pill status-${row.status}"><i></i>${row.status}</span></td><td>${cells[i]}</td><td>${row.script}</td></tr>`).join("");
      const logBox = $("#logBox");
      const scroller = logBox.closest(".logbox") || logBox;
      const first = scroller.scrollTop;
      const atBottom = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 12;
      logBox.textContent = (logs.length ? logs : ["No logs yet."]).join("\n");
      if (atBottom) scroller.scrollTop = scroller.scrollHeight;
      else scroller.scrollTop = first;
    }
    async function refreshStatus() {
      const res = await fetch("/api/tasks/status");
      renderPipeline(await res.json());
    }
    function setMetric(id, value) {
      const el = $(id);
      const v = value || "-";
      if (metricCache[id] !== undefined && metricCache[id] !== v && v !== "-") {
        const card = el.closest(".metric");
        if (card) { card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash"); }
      }
      metricCache[id] = v;
      el.textContent = v;
    }
    async function refreshDashboard() {
      const res = await fetch("/api/dashboard");
      const data = await res.json();
      const m = data.metrics || {};
      setMetric("#metricDefense", m.defense);
      setMetric("#metricRisk", m.risk);
      setMetric("#metricTeacher", m.teacher_acc);
      setMetric("#metricStudent", m.student_acc);
      setMetric("#metricRetention", m.retention);
      $("#controlTable tbody").innerHTML = (data.controls || []).map(r => `<tr><td>${r.control}</td><td>${r.teacher}</td><td>${r.student}</td><td>${r.delta}</td></tr>`).join("") || `<tr><td>暂无结果</td><td>-</td><td>-</td><td>-</td></tr>`;
      const reason = data.reason || {};
      $("#reasonTable tbody").innerHTML = `<tr><td>${reason.n || "-"}</td><td>${reason.stage1_reason || "-"}</td><td>${reason.stage2_reason || "-"}</td><td>${reason.delta || "-"}</td><td>${reason.stage2_win || "-"}</td></tr>`;
    }
    let wmMeanCache = null;
    async function refreshWatermark() {
      let data;
      try { data = await (await fetch("/api/watermark")).json(); } catch (e) { return; }
      const m = data.metrics || {};
      const meanEl = $("#wmMeanZ");
      if (meanEl) {
        if (wmMeanCache !== null && wmMeanCache !== m.mean_z && (m.mean_z || "-") !== "-") {
          meanEl.classList.remove("flash"); void meanEl.offsetWidth; meanEl.classList.add("flash");
        }
        wmMeanCache = m.mean_z;
        meanEl.textContent = m.mean_z || "-";
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
          gauge.style.background = `conic-gradient(${meanZ > 4 ? "var(--ok)" : "var(--accent-3)"} var(--deg,0deg), var(--surface-3) 0deg)`;
          gauge.querySelector("span").textContent = meanZ > 4 ? "有水印" : "偏弱";
        } else {
          gauge.style.setProperty("--deg", "0deg");
          gauge.querySelector("span").textContent = "z / 8";
        }
      }
      const scope = $("#wmScope");
      if (scope) scope.textContent = data.ok ? `已评分 ${data.num_scored ?? "?"} / ${data.num_total ?? "?"} 条 · split=${data.split ?? "?"}` : "尚未运行检测";
    }
    function startPolling() {
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(refreshStatus, 1000);
    }
    function wireEvents() {
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
      $$("[data-task]").forEach(btn => btn.addEventListener("click", async () => {
        const data = await post(`/api/tasks/${btn.dataset.task}`);
        if (data.pipeline) renderPipeline(data.pipeline);
        startPolling();
        if (btn.dataset.task === "watermark_detect") window.__wmPoll = setInterval(refreshWatermark, 3000);
      }));
    }
    async function boot() {
      buildForms();
      wireEvents();
      const res = await fetch("/api/config");
      applyConfig(await res.json());
      await refreshDashboard();
      await refreshWatermark();
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
