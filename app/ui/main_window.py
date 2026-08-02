from __future__ import annotations

import os
import json
import posixpath
import shlex
import subprocess
import sys
import threading
import tkinter as tk
import time
from tkinter import font as tkfont, messagebox, ttk
from pathlib import Path, PureWindowsPath
from typing import Callable

from app.config import APP_DIR, AppConfig, load_config, save_config
from app.models import CloneRequest
from app.services.git_service import REPO_DIR_NAME, RemoteGitCloneService
from app.services.ssh_client import SSHClientManager, parse_connection_command


BG = "#0b0f16"
SURFACE = "#141b26"
SURFACE_ALT = "#1c2635"
SURFACE_HI = "#222d3f"
TEXT = "#f2f6fc"
MUTED = "#9aa7bd"
ACCENT = "#2dd4bf"
ACCENT_ACTIVE = "#14b8a6"
ACCENT_SOFT = "#0f2c2b"
BORDER = "#26344a"
BORDER_STRONG = "#33465f"
SUCCESS = "#0f2926"
INPUT_BG = "#0a0f17"
INPUT_FOCUS_BG = "#0d141e"
BUTTON_DARK = "#243349"
BUTTON_DARK_ACTIVE = "#2f4160"
SELECTED_BG = "#173b3a"
SUBTLE = "#6b7890"
STATUS_SUCCESS = "#34d399"
STATUS_RUNNING = "#38bdf8"
STATUS_PAUSED = "#fbbf24"
STATUS_PENDING = "#94a3b8"
STATUS_FAILED = "#fb7185"

UI_FONT_CANDIDATES = (
    "Segoe UI",
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
    "AR PL UMing CN",
    "Ubuntu",
    "DejaVu Sans",
)
MONO_FONT_CANDIDATES = (
    "Consolas",
    "Cascadia Mono",
    "Source Han Mono SC",
    "Noto Sans Mono CJK SC",
    "Noto Sans Mono CJK JP",
    "WenQuanYi Micro Hei",
    "DejaVu Sans Mono",
    "Courier New",
)

PIPELINE_STEPS = [
    ("teacher_collect", "教师 API 数据采集", "scripts2/teacher_model_data_collect.sh"),
    ("teacher_eval", "教师完整风险基线", "scripts2/run_full_eval_pipeline_teacher.sh"),
    ("stage1_train", "Stage1 学生蒸馏", "scripts2/run_stage1.sh"),
    ("stage2_train", "Stage2 学生蒸馏", "scripts2/run_stage2.sh"),
    ("student_eval", "学生完整风险评估", "scripts2/run_full_eval_pipeline_fast.sh"),
    ("reason_judge", "思维链评估", "reason_judge/run_judge.sh"),
    ("risk_report", "风险报告聚合", "dashboard aggregation"),
]


class MainWindow:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.root = tk.Tk()
        self._configure_tk_scaling()
        self.root.title("Remote Clone Tool")
        self.root.geometry("1480x900")
        self.root.minsize(1280, 820)
        self.root.configure(bg=BG)
        self.ui_font_family = self._choose_font(UI_FONT_CANDIDATES)
        self.mono_font_family = self._choose_font(MONO_FONT_CANDIDATES)
        self._configure_default_fonts()

        self.clone_service = RemoteGitCloneService()
        self.config = load_config()

        self.connection_command_var = tk.StringVar(value=self.config.connection_command)
        self.project_var = tk.StringVar(value=self.config.project_path)
        self.password_var = tk.StringVar()
        self.ssh_password = ""
        self.status_var = tk.StringVar(value="准备就绪")

        self.console_vars: dict[str, tk.Variable] = {}
        self.form_vars: dict[str, tk.StringVar] = {}
        self.clone_button: tk.Button | None = None
        self.run_pipeline_button: tk.Button | None = None
        self.main_frame: tk.Frame | None = None
        self.console_frame: tk.Frame | None = None
        self.console_repo_path: tk.Label | None = None
        self.task_tree: ttk.Treeview | None = None
        self.task_progress: ttk.Progressbar | None = None
        self.task_log_box: tk.Text | None = None
        self.task_summary_var = tk.StringVar(value="暂无仿真任务" if self.debug else "暂无真实任务")
        self.dashboard_vars = {
            "defense": tk.StringVar(value="-"),
            "risk": tk.StringVar(value="-"),
            "teacher_acc": tk.StringVar(value="-"),
            "student_acc": tk.StringVar(value="-"),
            "retention": tk.StringVar(value="-"),
            "reason_delta": tk.StringVar(value="-"),
        }
        self.dashboard_control_tree: ttk.Treeview | None = None
        self.dashboard_reason_tree: ttk.Treeview | None = None
        self.pipeline_task: dict[str, float] | None = None
        self.pipeline_refresh_after_id: str | None = None
        self.real_pipeline_running = False
        self.real_pipeline_rows: list[dict[str, str]] | None = None
        self.real_pipeline_logs: list[str] = []
        self.local_real_log_path: Path | None = None
        self.real_pipeline_progress = 0.0
        self.config_save_after_id: str | None = None
        self.startup_repo_path = self._startup_repo_path()
        self.connection_command_var.trace_add("write", lambda *_: self._schedule_config_save())
        self.project_var.trace_add("write", lambda *_: self._schedule_config_save())

        self._init_console_vars()
        self._configure_styles()
        self._build_layout()
        if self.startup_repo_path is not None:
            saved_root_dir = str(self.config.console_vars.get("root_dir", "")).strip()
            self._show_console_screen(saved_root_dir or str(self.startup_repo_path))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _choose_font(self, candidates: tuple[str, ...]) -> str:
        if sys.platform.startswith("linux"):
            for family in candidates:
                if self._fontconfig_matches(family) and self._tk_accepts_font(family):
                    return family
        available = {family.lower(): family for family in tkfont.families(self.root)}
        for family in candidates:
            matched = available.get(family.lower())
            if matched:
                return matched
        for family in candidates:
            requested = family.lower()
            for available_key, available_family in available.items():
                if requested in available_key or available_key in requested:
                    return available_family
        return tkfont.nametofont("TkDefaultFont").actual("family")

    def _fontconfig_matches(self, family: str) -> bool:
        try:
            result = subprocess.run(
                ["fc-match", "-f", "%{family}", family],
                capture_output=True,
                text=True,
                timeout=0.3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        matched = result.stdout.split(",", 1)[0].strip().lower()
        requested = family.lower()
        return bool(matched) and (requested in matched or matched in requested)

    def _tk_accepts_font(self, family: str) -> bool:
        actual = tkfont.Font(root=self.root, family=family, size=10).actual("family")
        actual = str(actual).lower()
        requested = family.lower()
        return requested in actual or actual in requested

    def _font(self, size: int, weight: str = "normal", mono: bool = False) -> tuple[str, int, str]:
        family = self.mono_font_family if mono else self.ui_font_family
        return (family, size, weight)

    def _button_feedback(
        self,
        button: tk.Button,
        normal_bg: str,
        hover_bg: str,
        normal_fg: str = TEXT,
        hover_fg: str | None = None,
    ) -> None:
        hover_fg = hover_fg or normal_fg

        def on_enter(_: tk.Event) -> None:
            if str(button.cget("state")) == "normal":
                button.configure(bg=hover_bg, fg=hover_fg)

        def on_leave(_: tk.Event) -> None:
            if str(button.cget("state")) == "normal":
                button.configure(bg=normal_bg, fg=normal_fg)

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def _focus_highlight(self, entry: tk.Entry, resting_bg: str = INPUT_BG) -> tk.Entry:
        def on_focus_in(_: tk.Event) -> None:
            entry.configure(bg=INPUT_FOCUS_BG, highlightbackground=ACCENT)

        def on_focus_out(_: tk.Event) -> None:
            entry.configure(bg=resting_bg, highlightbackground=BORDER)

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)
        return entry

    def _configure_default_fonts(self) -> None:
        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(font_name).configure(family=self.ui_font_family)
        tkfont.nametofont("TkFixedFont").configure(family=self.mono_font_family)

    def _configure_tk_scaling(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes

            dpi = ctypes.windll.user32.GetDpiForWindow(self.root.winfo_id())
            self.root.tk.call("tk", "scaling", dpi / 72)
        except Exception:
            pass

    def _startup_base_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parents[2]

    def _startup_repo_path(self) -> Path | None:
        base_dir = self._startup_base_dir()
        candidates = [
            base_dir,
            base_dir / REPO_DIR_NAME,
            base_dir.parent / REPO_DIR_NAME,
        ]
        for candidate in candidates:
            if self._looks_like_repo(candidate):
                return candidate
        return None

    def _looks_like_repo(self, path: Path) -> bool:
        return path.is_dir() and (
            ((path / ".git").is_dir() and path.name == REPO_DIR_NAME)
            or (path / "vq_lord3").is_dir()
            or (path / "fastapi_vqlord").is_dir()
        )

    def _init_console_vars(self) -> None:
        repo_root = str(self.startup_repo_path) if self.startup_repo_path is not None else self._repo_path_from_project_path(self.project_var.get())
        defaults = {
            "server_ip": "127.0.0.1",
            "server_port": "8011",
            "python_bin": self._default_python_bin(repo_root),
            "model_path": self._default_model_path(repo_root),
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
        }
        defaults.update(self._console_path_defaults(repo_root))
        for key, value in defaults.items():
            if isinstance(value, bool):
                self.console_vars[key] = tk.BooleanVar(value=value)
            elif isinstance(value, int):
                self.console_vars[key] = tk.IntVar(value=value)
            else:
                self.console_vars[key] = tk.StringVar(value=value)
        self._apply_saved_console_vars()
        self.console_vars["result_dir"].trace_add("write", lambda *_: self._refresh_derived_console_paths())
        self.console_vars["stage2_adapter"].trace_add("write", lambda *_: self._refresh_derived_console_paths())
        self.console_vars["reason_judge_dir"].trace_add("write", lambda *_: self._refresh_reason_judge_paths())
        self.console_vars["dataset_path"].trace_add("write", lambda *_: self._refresh_reason_judge_paths())
        for variable in self.console_vars.values():
            variable.trace_add("write", lambda *_: self._schedule_config_save())
        self._refresh_derived_console_paths()
        self._refresh_reason_judge_paths()

    def _apply_saved_console_vars(self) -> None:
        for key, value in self.config.console_vars.items():
            variable = self.console_vars.get(key)
            if variable is None:
                continue
            try:
                if isinstance(variable, tk.BooleanVar):
                    variable.set(self._as_bool(value))
                elif isinstance(variable, tk.IntVar):
                    variable.set(int(value))
                else:
                    variable.set(str(value))
            except (tk.TclError, TypeError, ValueError):
                continue

    def _as_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _repo_path_from_project_path(self, project_path: str) -> str:
        project_path = project_path.strip()
        if not project_path:
            return REPO_DIR_NAME
        return self._remote_join(project_path, REPO_DIR_NAME)

    def _remote_join(self, *parts: str) -> str:
        cleaned = [part.strip() for part in parts if part and part.strip()]
        if not cleaned:
            return ""
        return posixpath.normpath(posixpath.join(*cleaned))

    def _remote_parent(self, path: str) -> str:
        normalized = posixpath.normpath(path.strip().rstrip("/") or ".")
        parent = posixpath.dirname(normalized)
        return parent or "."

    def _is_windows_path(self, path: str) -> bool:
        stripped = path.strip()
        return len(stripped) >= 2 and stripped[1] == ":" or "\\" in stripped

    def _path_join(self, *parts: str) -> str:
        cleaned = [part.strip() for part in parts if part and part.strip()]
        if not cleaned:
            return ""
        if self._is_windows_path(cleaned[0]):
            return str(PureWindowsPath(cleaned[0], *cleaned[1:]))
        return self._remote_join(*cleaned)

    def _path_parent(self, path: str) -> str:
        if self._is_windows_path(path):
            parent = PureWindowsPath(path.strip().rstrip("\\/")).parent
            return str(parent) if str(parent) != "." else "."
        return self._remote_parent(path)

    def _local_path_exists(self, path: str) -> bool:
        if not self._is_windows_path(path):
            return False
        return Path(path).exists()

    def _default_vla_mark_dir(self, repo_root: str) -> str:
        candidates = [
            self._path_join(repo_root, "VLA-mark"),
            self._path_join(self._path_parent(repo_root), "VLA-mark"),
        ]
        for candidate in candidates:
            if self._local_path_exists(candidate):
                return candidate
        return candidates[0]

    def _default_python_bin(self, repo_root: str) -> str:
        configured = os.environ.get("PYTHON_BIN") or os.environ.get("PYTHON")
        if configured:
            return configured

        if self._is_windows_path(repo_root):
            for candidate in (
                self._path_join(repo_root, ".venv", "Scripts", "python.exe"),
                self._path_join(repo_root, "venv", "Scripts", "python.exe"),
            ):
                if self._local_path_exists(candidate):
                    return candidate
            if not getattr(sys, "frozen", False):
                return sys.executable
            return "python"

        return "python3"

    def _default_project_root(self, repo_root: str) -> str:
        return self._path_parent(repo_root)

    def _default_model_path(self, repo_root: str) -> str:
        configured = os.environ.get("MODEL_PATH")
        if configured:
            return configured
        return self._path_join(self._default_project_root(repo_root), "models")

    def _default_dataset_path(self, repo_root: str) -> str:
        configured = os.environ.get("DATASET_PATH")
        if configured:
            return configured
        return self._path_join(self._default_project_root(repo_root), "datasets")

    def _console_path_defaults(self, repo_root: str) -> dict[str, str]:
        repo_root = posixpath.normpath(repo_root.strip().rstrip("/") or REPO_DIR_NAME)
        result_dir = self._path_join(repo_root, "vq_lord_test_results")
        stage2_adapter = self._path_join(repo_root, "vq_lord_ckpts", "stage2", "stage2_lord_final")
        reason_judge_dir = self._path_join(repo_root, "reason_judge")
        reason_stage2_json = self._path_join(result_dir, "stage2_test_generate_readable.json")
        reason_stage3_json = self._path_join(result_dir, "stage3_test_generate_vq1_bucketed_parallel.json")
        return {
            "root_dir": repo_root,
            "python_bin": self._default_python_bin(repo_root),
            "model_path": self._default_model_path(repo_root),
            "reason_judge_dir": reason_judge_dir,
            "vla_mark_dir": self._default_vla_mark_dir(repo_root),
            "dataset_path": self._default_dataset_path(repo_root),
            "stage1_ckpt": self._path_join(repo_root, "vq_lord_ckpts", "stage1", "stage1_vision_epoch1"),
            "stage2_adapter": stage2_adapter,
            "result_dir": result_dir,
            "teacher_result_dir": self._path_join(result_dir, "teacher_compare"),
            "stage2_codebook": self._path_join(stage2_adapter, "vq_codebook.pt"),
            "reason_stage2_json": reason_stage2_json,
            "reason_stage3_json": reason_stage3_json,
            "reason_teacher_json": "",
            "reason_dataset_path": self._default_dataset_path(repo_root),
            "reason_out_dir": "outputs/judge_latest",
        }

    def _apply_repo_path_defaults(self, repo_path: str) -> None:
        for key, value in self._console_path_defaults(repo_path).items():
            variable = self.console_vars.get(key)
            if variable is not None:
                variable.set(value)

    def _refresh_derived_console_paths(self) -> None:
        result_dir = self.console_vars["result_dir"].get()
        stage2_adapter = self.console_vars["stage2_adapter"].get()
        self.console_vars["teacher_result_dir"].set(self._path_join(str(result_dir), "teacher_compare"))
        self.console_vars["stage2_codebook"].set(self._path_join(str(stage2_adapter), "vq_codebook.pt"))

    def _refresh_reason_judge_paths(self) -> None:
        result_dir = str(self.console_vars["result_dir"].get())
        if not str(self.console_vars["reason_stage2_json"].get()).strip():
            self.console_vars["reason_stage2_json"].set(self._path_join(result_dir, "stage2_test_generate_readable.json"))
        if not str(self.console_vars["reason_stage3_json"].get()).strip():
            self.console_vars["reason_stage3_json"].set(self._path_join(result_dir, "stage3_test_generate_vq1_bucketed_parallel.json"))
        dataset_path = self.console_vars.get("dataset_path")
        if dataset_path is not None and not str(self.console_vars["reason_dataset_path"].get()).strip():
            self.console_vars["reason_dataset_path"].set(str(dataset_path.get()))
        if not str(self.console_vars["reason_out_dir"].get()).strip():
            self.console_vars["reason_out_dir"].set("outputs/judge_latest")

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
        style.configure(
            "Dark.TNotebook.Tab",
            background=SURFACE_ALT,
            foreground=MUTED,
            bordercolor=BORDER,
            lightcolor=SURFACE_ALT,
            darkcolor=SURFACE_ALT,
            borderwidth=0,
            padding=(20, 11),
            font=self._font(10, "bold"),
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", SURFACE), ("active", SURFACE_HI)],
            foreground=[("selected", ACCENT), ("active", TEXT)],
            lightcolor=[("selected", SURFACE)],
            expand=[("selected", (0, 0, 0, 0))],
        )
        style.configure(
            "Dark.Treeview",
            background=INPUT_BG,
            fieldbackground=INPUT_BG,
            foreground=TEXT,
            bordercolor=BORDER,
            borderwidth=0,
            rowheight=30,
            font=self._font(10),
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=SURFACE_ALT,
            foreground=MUTED,
            relief="flat",
            borderwidth=0,
            bordercolor=BORDER,
            padding=(8, 6),
            font=self._font(10, "bold"),
        )
        style.map(
            "Dark.Treeview.Heading",
            background=[("active", SURFACE_HI), ("pressed", SURFACE_HI)],
            foreground=[("active", TEXT), ("pressed", TEXT)],
            relief=[("active", "flat"), ("pressed", "flat")],
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", SELECTED_BG)],
            foreground=[("selected", TEXT)],
        )
        style.configure(
            "Dark.Horizontal.TProgressbar",
            background=ACCENT,
            troughcolor=INPUT_BG,
            bordercolor=BORDER,
            lightcolor=ACCENT,
            darkcolor=ACCENT_ACTIVE,
            thickness=10,
        )
        style.configure(
            "Vertical.TScrollbar",
            background=BUTTON_DARK,
            troughcolor=BG,
            bordercolor=BG,
            arrowcolor=MUTED,
            relief="flat",
            borderwidth=0,
            lightcolor=BUTTON_DARK,
            darkcolor=BUTTON_DARK,
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", BUTTON_DARK_ACTIVE)],
            arrowcolor=[("active", TEXT)],
        )

    def _build_layout(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        container = tk.Frame(self.root, bg=BG)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        self.main_frame = tk.Frame(container, bg=BG)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self._build_clone_screen(self.main_frame)

        self.console_frame = tk.Frame(container, bg=BG)
        self.console_frame.grid(row=0, column=0, sticky="nsew")
        self._build_console_screen(self.console_frame)

        self.main_frame.tkraise()

    def _build_clone_screen(self, parent: tk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        outer = tk.Frame(parent, bg=BG, padx=40, pady=32)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.grid_columnconfigure(0, weight=1)

        hero = tk.Frame(outer, bg=BG)
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 26))

        tk.Label(hero, text="🛰  Remote Project Console", bg=BG, fg=TEXT, font=self._font(26, "bold"), anchor="w").pack(anchor="w")
        tk.Label(hero, text="连接远程服务器并准备项目仓库", bg=BG, fg=MUTED, font=self._font(12), anchor="w").pack(anchor="w", pady=(8, 0))

        info_band = tk.Frame(outer, bg=SUCCESS, highlightbackground=BORDER, highlightthickness=1)
        info_band.grid(row=1, column=0, sticky="ew", pady=(0, 24))
        tk.Frame(info_band, bg=ACCENT, width=3).pack(side="left", fill="y")
        tk.Label(
            info_band,
            text="ⓘ  使用服务器连接命令连接远程环境，并在目标项目目录下 clone 指定仓库。",
            bg=SUCCESS,
            fg="#c6fff6",
            font=self._font(12, "bold"),
            anchor="w",
            justify="left",
            padx=18,
            pady=16,
        ).pack(side="left", anchor="w")

        form = tk.Frame(outer, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, padx=24, pady=24)
        form.grid(row=2, column=0, sticky="ew")
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)

        self._build_input(form, 0, 0, "服务器连接命令", self.connection_command_var, "ssh root@192.168.1.10 或 ssh ubuntu@example.com -p 22", wide=True)
        self._build_input(form, 1, 0, "项目地址", self.project_var, "/home/user/workspace")
        self._build_input(form, 1, 1, "密码", self.password_var, "运行时使用，不会本地保存", show="*")

        footer = tk.Frame(outer, bg=BG)
        footer.grid(row=3, column=0, sticky="ew", pady=(20, 0))
        footer.grid_columnconfigure(0, weight=1)

        tk.Label(footer, textvariable=self.status_var, bg=BG, fg=MUTED, font=self._font(11), anchor="w").grid(row=0, column=0, sticky="w")
        self.clone_button = tk.Button(
            footer,
            text="⬇  开始远程 Clone",
            command=self._start_clone,
            bg=ACCENT,
            fg="#041311",
            activebackground=ACCENT_ACTIVE,
            activeforeground="#041311",
            relief="flat",
            bd=0,
            padx=22,
            pady=12,
            font=self._font(11, "bold"),
            cursor="hand2",
        )
        self._button_feedback(self.clone_button, ACCENT, ACCENT_ACTIVE, "#041311", "#041311")
        self.clone_button.grid(row=0, column=1, sticky="e")

    def _build_console_screen(self, parent: tk.Frame) -> None:
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        sidebar = tk.Frame(parent, bg=SURFACE, width=230, highlightbackground=BORDER, highlightthickness=1)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        main = tk.Frame(parent, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(3, weight=3)

        self._build_sidebar(sidebar)
        self._build_console_main(main)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        canvas = tk.Canvas(parent, bg=SURFACE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=SURFACE)

        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(content_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        self._bind_mousewheel(canvas)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._sidebar_section(content, "🌐  全局路径")
        self._sidebar_input(content, "ROOT_DIR", self.console_vars["root_dir"])
        self._sidebar_input(content, "REASON_JUDGE_DIR", self.console_vars["reason_judge_dir"])
        self._sidebar_input(content, "VLA_MARK_DIR", self.console_vars["vla_mark_dir"])
        self._sidebar_input(content, "PYTHON_BIN", self.console_vars["python_bin"])
        self._sidebar_input(content, "MODEL_PATH (学生模型)", self.console_vars["model_path"])
        self._sidebar_input(content, "DATASET_NAME", self.console_vars["dataset_name"])
        self._sidebar_input(content, "DATASET_PATH", self.console_vars["dataset_path"])
        self._sidebar_input(content, "CUDA_VISIBLE_DEVICES", self.console_vars["cuda_devices"])

        self._sidebar_section(content, "🔑  教师模型连接")
        self._sidebar_input(content, "TEACHER_API_KEY", self.console_vars["teacher_api_key"], show="*")
        self._sidebar_input(content, "TEACHER_API_BASE", self.console_vars["teacher_api_base"])
        self._sidebar_input(content, "VICTIM_MODEL", self.console_vars["victim_model"])

        if self.debug:
            self._sidebar_section(content, "🎛  仿真控制")
            self._sidebar_scale(content, "单任务仿真秒数", self.console_vars["sim_duration"], 3, 60)
        self._sidebar_section(content, "⚙  运行控制")
        self._sidebar_check(content, "任务运行时自动刷新", self.console_vars["auto_refresh"])

        back = tk.Button(
            content,
            text="←  返回连接界面",
            command=self._show_main_screen,
            bg=BUTTON_DARK,
            fg=TEXT,
            activebackground=BUTTON_DARK_ACTIVE,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=16,
            pady=10,
            font=self._font(10, "bold"),
            cursor="hand2",
        )
        self._button_feedback(back, BUTTON_DARK, BUTTON_DARK_ACTIVE)
        back.pack(fill="x", padx=12, pady=(18, 24))

    def _build_console_main(self, parent: tk.Frame) -> None:
        header = tk.Frame(parent, bg=BG, padx=24, pady=14)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title_row = tk.Frame(header, bg=BG)
        title_row.grid(row=0, column=0, sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)
        tk.Label(title_row, text="MLLM 能力泄漏风险检测平台", bg=BG, fg=TEXT, font=self._font(22, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        mode_text = "● DEBUG · 仿真" if self.debug else "● LIVE · scripts2"
        mode_bg = BUTTON_DARK if self.debug else SUCCESS
        mode_fg = STATUS_RUNNING if self.debug else STATUS_SUCCESS
        mode_badge = tk.Label(title_row, text=mode_text, bg=mode_bg, fg=mode_fg, font=self._font(10, "bold"), padx=14, pady=6, highlightbackground=BORDER, highlightthickness=1)
        mode_badge.grid(row=0, column=1, sticky="e", padx=(16, 0))
        tk.Label(header, text="多阶段蒸馏、评测与风险监控控制台", bg=BG, fg=MUTED, font=self._font(11, "bold"), anchor="w").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.console_repo_path = tk.Label(header, text="", bg=BG, fg=ACCENT, font=self._font(10, mono=True), anchor="w")
        self.console_repo_path.grid(row=2, column=0, sticky="w", pady=(6, 0))

        metrics = tk.Frame(parent, bg=BG, padx=24)
        metrics.grid(row=1, column=0, sticky="ew")
        for idx in range(5):
            metrics.grid_columnconfigure(idx, weight=1)
        metric_data = [
            ("🛡  防蒸馏能力", self.dashboard_vars["defense"]),
            ("⚠  窃取风险", self.dashboard_vars["risk"]),
            ("🎯  Teacher Acc", self.dashboard_vars["teacher_acc"]),
            ("🎯  Student Acc", self.dashboard_vars["student_acc"]),
            ("📉  Acc Retention", self.dashboard_vars["retention"]),
        ]
        for idx, (label, value) in enumerate(metric_data):
            self._metric_card(metrics, idx, label, value)

        actions = tk.Frame(parent, bg=BG, padx=24, pady=10)
        actions.grid(row=2, column=0, sticky="ew")
        actions.grid_columnconfigure(0, weight=3)
        actions.grid_columnconfigure(1, weight=1)
        pipeline_text = "一键跑完整 Pipeline（仿真）" if self.debug else "一键跑完整 Pipeline"
        self.run_pipeline_button = self._action_button(actions, pipeline_text, 0, self._toggle_pipeline)
        self._action_button(actions, "↻  刷新大盘", 1, self._refresh_dashboard_async)

        body = tk.Frame(parent, bg=BG, padx=24, pady=10)
        body.grid(row=3, column=0, sticky="nsew", pady=(0, 12))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        notebook = ttk.Notebook(body, style="Dark.TNotebook")
        notebook.grid(row=0, column=0, sticky="nsew")

        tab_info = tk.Frame(notebook, bg=BG)
        tab_watermark = tk.Frame(notebook, bg=BG)
        tab_data = tk.Frame(notebook, bg=BG)
        tab_train = tk.Frame(notebook, bg=BG)
        tab_eval = tk.Frame(notebook, bg=BG)
        tab_risk = tk.Frame(notebook, bg=BG)

        notebook.add(tab_info, text="  📊  任务状态和日志信息  ")
        notebook.add(tab_watermark, text="  💧  水印检测  ")
        notebook.add(tab_data, text="  🗂  1. 数据与教师基线  ")
        notebook.add(tab_train, text="  🧪  2. Stage1/Stage2 蒸馏  ")
        notebook.add(tab_eval, text="  📈  3. 完整评测  ")
        notebook.add(tab_risk, text="  🛡  4. 风险大盘  ")

        self._build_tab_info(tab_info)
        self._build_tab_watermark(tab_watermark)
        self._build_tab_data(tab_data)
        self._build_tab_train(tab_train)
        self._build_tab_eval(tab_eval)
        self._build_tab_risk(tab_risk)

    def _build_tab_info(self, parent: tk.Frame) -> None:
        container = self._scrollable_tab(parent)
        self._section_title(container, "任务运行状态")
        self._wide_button(container, "↻  刷新任务状态", self._refresh_pipeline_view)
        self._build_task_status_panel(container)

        self._divider(container)
        self._section_title(container, "系统后台日志")
        self._wide_button(container, "↻  刷新最新日志", self._refresh_pipeline_view)
        self.task_log_box = self._code_box(container, "No simulated tasks yet." if self.debug else "No real tasks yet.")

    def _build_tab_watermark(self, parent: tk.Frame) -> None:
        container = self._scrollable_tab(parent)
        row = tk.Frame(container, bg=BG)
        row.pack(fill="x", pady=(0, 18))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)

        left = tk.Frame(row, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = tk.Frame(row, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._panel_input(left, "基底模型 score/probability", self.console_vars["wm_base_before_score"])
        self._panel_input(left, "提取模型 score/probability", self.console_vars["wm_extracted_score"])
        self._panel_input(right, "测试模型 score/probability", self.console_vars["wm_test_score"])

        self._wide_button(container, "▶  水印失效风险评估", lambda: self._start_single_task("watermark_detect"))

    def _build_tab_data(self, parent: tk.Frame) -> None:
        container = self._scrollable_tab(parent)
        self._section_title(container, "1A. 教师四字段标注采集")
        self._two_col_form(
            container,
            [
                ("SCIENCEQA_SPLIT", "train"),
                ("TRAIN_NUM (0=全量)", "0"),
                ("MAX_SAMPLES (0=全量)", "0"),
                ("SCIENCEQA_SEED", "20240306"),
            ],
            [
                ("TEACHER_LANG", "en"),
                ("TEACHER_ENABLE_THINKING", "False"),
                ("COLLECT_TEACHER_DATA", "True"),
                ("STRICT_TEACHER_DISTILL", "True"),
                ("NUM_WORKERS", "4"),
            ],
        )
        self._wide_button(container, "▶  开始采集教师数据", lambda: self._start_single_task("teacher_collect"))

        self._divider(container)
        self._section_title(container, "1B. 教师风险基线评测")
        self._two_col_form(
            container,
            [
                ("MAX_NEW_TOKENS", "64"),
                ("MAX_CONCURRENCY", "4"),
                ("SCIENCEQA_CONTROL_SPLIT", "test"),
                ("SCIENCEQA_CONTROL_MAX_SAMPLES", "0"),
            ],
            [
                ("SCIENCEQA_CONTROLS", "baseline,text_only_blank,hint_ablation,option_shuffle,random_image_swap,image_blur,image_downsample"),
                ("TEACHER_RESULT_DIR", self.console_vars["teacher_result_dir"]),
            ],
        )
        self._wide_button(container, "▶  启动教师完整评测", lambda: self._start_single_task("teacher_eval"))

    def _build_tab_train(self, parent: tk.Frame) -> None:
        container = self._scrollable_tab(parent)
        self._section_title(container, "路径配置")
        self._two_col_form(
            container,
            [("STAGE1_CKPT_PATH", self.console_vars["stage1_ckpt"])],
            [("STAGE2_FINAL_ADAPTER_PATH", self.console_vars["stage2_adapter"])],
        )

        self._divider(container)
        self._section_title(container, "Stage1 训练")
        self._three_col_form(
            container,
            [("STAGE1_EPOCHS", "3"), ("STAGE1_BATCH_SIZE", "1"), ("STAGE1_GRAD_ACCUM", "2")],
            [("STAGE1_LR", "3e-5"), ("STAGE1_MAX_LENGTH", "1536"), ("USE_4BIT", "False")],
            [("FREEZE_VISION_TOWER", "True"), ("LORA_RANK", "16"), ("LORA_ALPHA", "32")],
        )
        self._two_col_form(
            container,
            [("STAGE1_FIELD_WEIGHT_REASONING", "2.0")],
            [("STAGE1_FIELD_WEIGHT_ANSWER", "12.0")],
        )
        self._wide_button(container, "▶  启动 Stage1 蒸馏", lambda: self._start_single_task("stage1_train"))

        self._divider(container)
        self._section_title(container, "Stage2 训练")
        self._three_col_form(
            container,
            [("STAGE2_EPOCHS", "1"), ("PERIOD_NUM", "1"), ("STAGE2_GRAD_ACCUM", "2")],
            [("STAGE2_LR", "2e-5"), ("TAU1", "0.02"), ("STAGE2_MAX_LENGTH", "1024")],
            [("PHASE_A_BATCH_SIZE", "1"), ("PHASE_B_BATCH_SIZE", "1"), ("STAGE2_EVAL_EVERY_PERIOD", "1")],
        )
        self._three_col_form(
            container,
            [("STAGE2_EVAL_TRAIN_NUM", "200"), ("STAGE2_EVAL_MAX_SAMPLES", "200"), ("EVAL_MAX_SAMPLES", "200")],
            [("FREEZE_VISION_TOWER", "True"), ("LORA_RANK", "16"), ("LORA_ALPHA", "32")],
            [("USE_4BIT", "False"), ("STAGE2_WRONG_IMAGE_ENABLE", "True"), ("STAGE2_PAIR_USE_ANSWER_CORRECTNESS", "False")],
        )
        self._wide_button(container, "▶  启动 Stage2 蒸馏", lambda: self._start_single_task("stage2_train"))

    def _build_tab_eval(self, parent: tk.Frame) -> None:
        container = self._scrollable_tab(parent)
        self._section_title(container, "3A. 学生完整风险评测")
        self._two_col_form(
            container,
            [
                ("ADAPTER_PATH", self.console_vars["stage2_adapter"]),
                ("VQ_CODEBOOK_PATH", self.console_vars["stage2_codebook"]),
            ],
            [
                ("RESULT_DIR", self.console_vars["result_dir"]),
                ("EVAL_MAX_NEW_TOKENS", "512"),
            ],
        )
        self._wide_button(container, "▶  启动学生完整评测", lambda: self._start_single_task("student_eval"))

        self._divider(container)
        self._section_title(container, "3B. 思维链评估")
        self._two_col_form(
            container,
            [
                ("STAGE2 / Stage1 readable eval JSON", self.console_vars["reason_stage2_json"]),
                ("STAGE3 / Stage2 readable eval JSON", self.console_vars["reason_stage3_json"]),
                ("TEACHER JSON (可选)", self.console_vars["reason_teacher_json"]),
                ("DATASET", self.console_vars["reason_dataset_path"]),
                ("OUT_DIR", self.console_vars["reason_out_dir"]),
            ],
            [
                ("JUDGE_MODEL", self.console_vars["judge_model"]),
                ("JUDGE_API_BASE", self.console_vars["judge_api_base"]),
                ("JUDGE_API_KEY", self.console_vars["judge_api_key"]),
                ("SAMPLE_NUM", self.console_vars["judge_sample_num"]),
                ("JUDGE_DATASET_NAME", "scienceqa"),
                ("SPLIT", "test"),
                ("REQUIRE_VALID_FORMAT", "True"),
            ],
        )
        self._wide_button(container, "▶  启动思维链评估", lambda: self._start_single_task("reason_judge"))

    def _build_tab_risk(self, parent: tk.Frame) -> None:
        container = self._scrollable_tab(parent)
        self._wide_button(container, "↻  刷新结果摘要", self._refresh_dashboard_async)
        metrics = tk.Frame(container, bg=BG)
        metrics.pack(fill="x", pady=(6, 18))
        for idx in range(4):
            metrics.grid_columnconfigure(idx, weight=1)
        for idx, (label, value) in enumerate(
            [
                ("🎯  Teacher Acc", self.dashboard_vars["teacher_acc"]),
                ("🎯  Student Acc", self.dashboard_vars["student_acc"]),
                ("📉  Acc Retention", self.dashboard_vars["retention"]),
                ("🧠  Reason Delta", self.dashboard_vars["reason_delta"]),
            ]
        ):
            self._metric_card(metrics, idx, label, value)

        self._divider(container)
        self._section_title(container, "Teacher / Student Controls")
        self.dashboard_control_tree = self._placeholder_table(
            container,
            ["control", "teacher", "student", "delta"],
            [["暂无结果", "-", "-", "-"]],
        )

        self._divider(container)
        self._section_title(container, "Reason Judge")
        self.dashboard_reason_tree = self._placeholder_table(
            container,
            ["Judged N", "Stage1 Reason", "Stage2 Reason", "Delta", "Stage2 Win"],
            [["-", "-", "-", "-", "-"]],
        )

    def _scrollable_tab(self, parent: tk.Frame) -> tk.Frame:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=BG, padx=8, pady=16)
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(content_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        self._bind_mousewheel(canvas)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        return content

    def _bind_mousewheel(self, canvas: tk.Canvas) -> None:
        def on_mousewheel(event: tk.Event) -> str:
            delta = self._mousewheel_units(event)
            if delta:
                canvas.yview_scroll(delta, "units")
            return "break"

        def bind(_: tk.Event) -> None:
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            canvas.bind_all("<Button-4>", on_mousewheel)
            canvas.bind_all("<Button-5>", on_mousewheel)

        def unbind(_: tk.Event) -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", bind)
        canvas.bind("<Leave>", unbind)

    def _mousewheel_units(self, event: tk.Event) -> int:
        if getattr(event, "num", None) == 4:
            return -1
        if getattr(event, "num", None) == 5:
            return 1
        return -int(event.delta / 120) if getattr(event, "delta", 0) else 0

    def _bind_text_mousewheel(self, text: tk.Text) -> None:
        def on_mousewheel(event: tk.Event) -> str:
            delta = self._mousewheel_units(event)
            if delta:
                text.yview_scroll(delta, "units")
            return "break"

        text.bind("<MouseWheel>", on_mousewheel)
        text.bind("<Button-4>", on_mousewheel)
        text.bind("<Button-5>", on_mousewheel)

    def _section_title(self, parent: tk.Widget, text: str) -> None:
        row = tk.Frame(parent, bg=BG)
        row.pack(anchor="w", fill="x", pady=(0, 12))
        tk.Frame(row, bg=ACCENT, width=3, height=18).pack(side="left", padx=(0, 10))
        tk.Label(row, text=text, bg=BG, fg=TEXT, font=self._font(16, "bold"), anchor="w").pack(side="left")

    def _info_band(self, parent: tk.Widget, text: str) -> None:
        band = tk.Frame(parent, bg=SUCCESS, highlightbackground=BORDER, highlightthickness=1)
        band.pack(fill="x", pady=(0, 18))
        tk.Frame(band, bg=ACCENT, width=3).pack(side="left", fill="y")
        tk.Label(band, text=text, bg=SUCCESS, fg="#c6fff6", font=self._font(11), anchor="w", justify="left", padx=16, pady=14).pack(side="left", anchor="w")

    def _divider(self, parent: tk.Widget) -> None:
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=18)

    def _wide_button(self, parent: tk.Widget, text: str, command: Callable[[], None] | None = None) -> tk.Button:
        button = tk.Button(parent, text=text, command=command, bg=BUTTON_DARK, fg=TEXT, activebackground=BUTTON_DARK_ACTIVE, activeforeground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, padx=18, pady=12, font=self._font(11, "bold"), cursor="hand2")
        self._button_feedback(button, BUTTON_DARK, BUTTON_DARK_ACTIVE)
        button.pack(fill="x", pady=(0, 12))
        return button

    def _build_task_status_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=14)
        panel.pack(fill="x", pady=(0, 12))
        tk.Label(panel, textvariable=self.task_summary_var, bg=SURFACE, fg=TEXT, font=self._font(12, "bold"), anchor="w").pack(anchor="w", pady=(0, 10))
        self.task_progress = ttk.Progressbar(panel, orient="horizontal", mode="determinate", maximum=100, style="Dark.Horizontal.TProgressbar")
        self.task_progress.pack(fill="x", pady=(0, 12))

        columns = ["stage", "status", "progress", "script"]
        self.task_tree = ttk.Treeview(panel, columns=columns, show="headings", style="Dark.Treeview", height=len(PIPELINE_STEPS))
        headings = {
            "stage": "阶段",
            "status": "状态",
            "progress": "进度",
            "script": "脚本",
        }
        widths = {
            "stage": 220,
            "status": 90,
            "progress": 90,
            "script": 520,
        }
        for column in columns:
            self.task_tree.heading(column, text=headings[column])
            self.task_tree.column(column, anchor="w", width=widths[column], stretch=True)
        self.task_tree.tag_configure("even", background=INPUT_BG)
        self.task_tree.tag_configure("odd", background="#0e1722")
        self.task_tree.tag_configure("success", foreground=STATUS_SUCCESS)
        self.task_tree.tag_configure("running", foreground=STATUS_RUNNING)
        self.task_tree.tag_configure("paused", foreground=STATUS_PAUSED)
        self.task_tree.tag_configure("pending", foreground=STATUS_PENDING)
        self.task_tree.tag_configure("failed", foreground=STATUS_FAILED)
        self.task_tree.pack(fill="x")
        panel.bind("<Configure>", lambda e: self._resize_task_tree_columns(e.width))
        self._refresh_pipeline_view()

    def _resize_task_tree_columns(self, width: int) -> None:
        if self.task_tree is None:
            return
        available = max(620, width - 44)
        status_width = 100
        progress_width = 90
        stage_width = max(220, min(340, int(available * 0.28)))
        script_width = max(280, available - stage_width - status_width - progress_width)
        self.task_tree.column("stage", width=stage_width)
        self.task_tree.column("status", width=status_width)
        self.task_tree.column("progress", width=progress_width)
        self.task_tree.column("script", width=script_width)

    def _panel_input(self, parent: tk.Widget, label: str, variable: tk.Variable) -> None:
        panel = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=14)
        panel.pack(fill="x", pady=(0, 12))
        tk.Label(panel, text=label, bg=SURFACE, fg=TEXT, font=self._font(11, "bold"), anchor="w").pack(anchor="w", pady=(0, 8))
        entry = tk.Entry(panel, textvariable=variable, bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT, font=self._font(11, mono=True))
        self._focus_highlight(entry)
        entry.pack(fill="x", ipady=9)

    def _placeholder_table(self, parent: tk.Widget, columns: list[str], rows: list[list[str]]) -> ttk.Treeview:
        frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", pady=(0, 12))
        tree = ttk.Treeview(frame, columns=columns, show="headings", style="Dark.Treeview", height=max(1, len(rows)))
        tree.tag_configure("even", background=INPUT_BG)
        tree.tag_configure("odd", background="#0e1722")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, anchor="w", width=max(120, 860 // max(1, len(columns))))
        for index, row in enumerate(rows):
            tree.insert("", "end", values=row, tags=("odd" if index % 2 else "even",))
        tree.pack(fill="x")
        return tree

    def _code_box(self, parent: tk.Widget, content: str) -> tk.Text:
        box = tk.Text(parent, height=24, wrap="word", bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, padx=12, pady=10, spacing1=1, spacing3=2, font=self._font(10, mono=True))
        box.pack(fill="x", pady=(0, 12))
        box.insert("1.0", content)
        box.configure(state="disabled")
        self._bind_text_mousewheel(box)
        return box

    def _refresh_dashboard_async(self) -> None:
        thread = threading.Thread(target=self._refresh_dashboard_worker, daemon=True)
        thread.start()

    def _refresh_dashboard_worker(self) -> None:
        try:
            payload = self._load_dashboard_payload()
        except Exception as exc:
            self._append_real_log(f"[dashboard] refresh failed: {exc}")
            return
        self.root.after(0, lambda: self._apply_dashboard_payload(payload))

    def _load_dashboard_payload(self) -> dict[str, object]:
        result_dir = self._console_value("result_dir")
        teacher_payload = self._read_json_result(self._path_join(result_dir, "scienceqa_control_suite_teacher_full.json"))
        student_payload = self._read_json_result(self._path_join(result_dir, "scienceqa_control_suite_full_fast.json"))
        report_payload = self._read_json_result(self._path_join(result_dir, "mm_eval_suite_report_full_fast.json"))
        reason_out_dir = self._resolve_reason_out_dir()
        reason_summary = self._read_tsv_result(self._path_join(reason_out_dir, "summary.tsv"))
        if not student_payload and report_payload:
            student_payload = {"metrics": report_payload.get("control_summary", {})} if isinstance(report_payload, dict) else {}
        return {
            "teacher": teacher_payload,
            "student": student_payload,
            "reason": reason_summary,
        }

    def _resolve_reason_out_dir(self) -> str:
        out_dir = self._console_value("reason_out_dir").strip()
        if not out_dir:
            out_dir = "outputs/judge_latest"
        if out_dir.startswith("/"):
            return out_dir
        return self._path_join(self._console_value("reason_judge_dir"), out_dir)

    def _read_json_result(self, path: str) -> dict[str, object]:
        text = self._read_text_result(path)
        if not text.strip():
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _read_tsv_result(self, path: str) -> dict[str, str]:
        text = self._read_text_result(path)
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return {}
        keys = lines[0].split("\t")
        values = lines[1].split("\t")
        return {key: values[idx] if idx < len(values) else "" for idx, key in enumerate(keys)}

    def _read_text_result(self, path: str) -> str:
        if not path:
            return ""
        if self._should_run_remote():
            command = f"if [ -f {shlex.quote(path)} ]; then cat {shlex.quote(path)}; fi"
            result = self._run_remote_config_command(command)
            if result.exit_code != 0:
                return ""
            return result.stdout
        local_path = Path(path)
        if not local_path.is_file():
            return ""
        return local_path.read_text(encoding="utf-8", errors="replace")

    def _apply_dashboard_payload(self, payload: dict[str, object]) -> None:
        teacher = payload.get("teacher") if isinstance(payload.get("teacher"), dict) else {}
        student = payload.get("student") if isinstance(payload.get("student"), dict) else {}
        reason = payload.get("reason") if isinstance(payload.get("reason"), dict) else {}

        teacher_summary = self._dashboard_control_summary(teacher)
        student_summary = self._dashboard_control_summary(student)
        teacher_acc = self._dashboard_baseline_accuracy(teacher)
        student_acc = self._dashboard_baseline_accuracy(student)
        retention = self._safe_ratio(student_acc, teacher_acc)
        risk_score = retention if retention is not None else student_acc
        reason_delta = self._safe_float(reason.get("delta_reason_score")) if isinstance(reason, dict) else None

        self.dashboard_vars["teacher_acc"].set(self._fmt_metric(teacher_acc))
        self.dashboard_vars["student_acc"].set(self._fmt_metric(student_acc))
        self.dashboard_vars["retention"].set(self._fmt_metric(retention))
        self.dashboard_vars["risk"].set(self._fmt_metric(risk_score))
        self.dashboard_vars["defense"].set(self._fmt_metric(1.0 - risk_score if risk_score is not None else None))
        self.dashboard_vars["reason_delta"].set(self._fmt_metric(reason_delta, signed=True))

        control_names = sorted(set(teacher_summary) | set(student_summary))
        control_rows: list[list[str]] = []
        for name in control_names:
            teacher_score = self._safe_float(teacher_summary.get(name, {}).get("accuracy")) if isinstance(teacher_summary.get(name), dict) else None
            student_score = self._safe_float(student_summary.get(name, {}).get("accuracy")) if isinstance(student_summary.get(name), dict) else None
            delta = student_score - teacher_score if student_score is not None and teacher_score is not None else None
            control_rows.append([name, self._fmt_metric(teacher_score), self._fmt_metric(student_score), self._fmt_metric(delta, signed=True)])
        if not control_rows:
            control_rows = [["暂无结果", "-", "-", "-"]]
        self._set_tree_rows(self.dashboard_control_tree, control_rows)

        reason_rows = [[
            str(reason.get("n", "-")) if isinstance(reason, dict) else "-",
            self._fmt_metric(self._safe_float(reason.get("stage2_reason_score")) if isinstance(reason, dict) else None),
            self._fmt_metric(self._safe_float(reason.get("stage3_reason_score")) if isinstance(reason, dict) else None),
            self._fmt_metric(reason_delta, signed=True),
            self._fmt_metric(self._safe_float(reason.get("stage2_win_rate")) if isinstance(reason, dict) else None),
        ]]
        self._set_tree_rows(self.dashboard_reason_tree, reason_rows)

    def _dashboard_control_summary(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            return {}
        metrics = payload.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("control_summary"), dict):
            return metrics["control_summary"]
        control_summary = payload.get("control_summary")
        if isinstance(control_summary, dict):
            return control_summary
        return {}

    def _dashboard_baseline_accuracy(self, payload: object) -> float | None:
        if not isinstance(payload, dict):
            return None
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            value = self._safe_float(metrics.get("baseline_accuracy"))
            if value is not None:
                return value
            summary = metrics.get("control_summary")
            if isinstance(summary, dict) and isinstance(summary.get("baseline"), dict):
                return self._safe_float(summary["baseline"].get("accuracy"))
        summary = payload.get("control_summary")
        if isinstance(summary, dict) and isinstance(summary.get("baseline"), dict):
            return self._safe_float(summary["baseline"].get("accuracy"))
        return None

    def _safe_float(self, value: object) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_ratio(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    def _fmt_metric(self, value: float | None, signed: bool = False) -> str:
        if value is None:
            return "-"
        return f"{value:+.4f}" if signed else f"{value:.4f}"

    def _set_tree_rows(self, tree: ttk.Treeview | None, rows: list[list[str]]) -> None:
        if tree is None:
            return
        for item_id in tree.get_children():
            tree.delete(item_id)
        for row in rows:
            tree.insert("", "end", values=row)

    def _two_col_form(
        self,
        parent: tk.Widget,
        left_fields: list[tuple[str, str | tk.Variable]],
        right_fields: list[tuple[str, str | tk.Variable]],
    ) -> None:
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, 12))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        left = tk.Frame(row, bg=BG)
        right = tk.Frame(row, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        for label, value in left_fields:
            self._static_field(left, label, value)
        for label, value in right_fields:
            self._static_field(right, label, value)

    def _three_col_form(
        self,
        parent: tk.Widget,
        col1: list[tuple[str, str | tk.Variable]],
        col2: list[tuple[str, str | tk.Variable]],
        col3: list[tuple[str, str | tk.Variable]],
    ) -> None:
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, 12))
        for idx in range(3):
            row.grid_columnconfigure(idx, weight=1)
        columns = []
        for idx in range(3):
            frame = tk.Frame(row, bg=BG)
            frame.grid(row=0, column=idx, sticky="nsew", padx=(0, 10) if idx < 2 else (10, 0))
            columns.append(frame)
        for label, value in col1:
            self._static_field(columns[0], label, value)
        for label, value in col2:
            self._static_field(columns[1], label, value)
        for label, value in col3:
            self._static_field(columns[2], label, value)

    def _static_field(self, parent: tk.Widget, label: str, value: str | tk.Variable) -> None:
        panel = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=12)
        panel.pack(fill="x", pady=(0, 10))
        tk.Label(panel, text=label, bg=SURFACE, fg=TEXT, font=self._font(10, "bold"), anchor="w").pack(anchor="w", pady=(0, 8))
        if isinstance(value, tk.Variable):
            variable = value
        else:
            key = self._form_env_key(label)
            saved_value = self.config.form_vars.get(key, value)
            variable = self.form_vars.setdefault(key, tk.StringVar(value=saved_value))
            variable.trace_add("write", lambda *_: self._schedule_config_save())
        entry_kwargs = {"textvariable": variable}
        entry = tk.Entry(panel, bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT, font=self._font(10, mono=True), **entry_kwargs)
        self._focus_highlight(entry)
        entry.pack(fill="x", ipady=9)

    def _metric_card(self, parent: tk.Widget, column: int, label: str, value: str | tk.Variable) -> None:
        card = tk.Frame(parent, bg=BORDER, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=column, sticky="ew", padx=(0, 12) if column < 4 else (0, 0))
        tk.Frame(card, bg=ACCENT, height=3).pack(fill="x")
        body = tk.Frame(card, bg=SURFACE, padx=16, pady=14)
        body.pack(fill="both", expand=True)
        tk.Label(body, text=label, bg=SURFACE, fg=MUTED, font=self._font(10, "bold"), anchor="w").pack(anchor="w")
        value_kwargs = {"textvariable": value} if isinstance(value, tk.Variable) else {"text": value}
        tk.Label(body, bg=SURFACE, fg=TEXT, font=self._font(20, "bold"), anchor="w", **value_kwargs).pack(anchor="w", pady=(6, 0))

    def _action_button(self, parent: tk.Widget, text: str, column: int, command: Callable[[], None] | None = None) -> tk.Button:
        btn = tk.Button(parent, text=text, command=command, bg=BUTTON_DARK if column == 1 else ACCENT, fg=TEXT if column == 1 else "#041311", activebackground=BUTTON_DARK_ACTIVE if column == 1 else ACCENT_ACTIVE, activeforeground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, padx=16, pady=13, font=self._font(11, "bold"), cursor="hand2")
        if column == 1:
            self._button_feedback(btn, BUTTON_DARK, BUTTON_DARK_ACTIVE)
        else:
            self._button_feedback(btn, ACCENT, ACCENT_ACTIVE, "#041311", "#041311")
        btn.grid(row=0, column=column, sticky="ew", padx=(0, 12) if column == 0 else (12, 0))
        return btn

    def _sidebar_section(self, parent: tk.Widget, title: str) -> None:
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=16)
        tk.Label(parent, text=title, bg=SURFACE, fg=TEXT, font=self._font(13, "bold")).pack(anchor="w", padx=12, pady=(0, 10))

    def _sidebar_input(self, parent: tk.Widget, label: str, variable: tk.Variable, show: str | None = None) -> None:
        tk.Label(parent, text=label, bg=SURFACE, fg=MUTED, font=self._font(9, "bold")).pack(anchor="w", padx=12, pady=(0, 5))
        entry = tk.Entry(parent, textvariable=variable, show=show or "", bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT, font=self._font(10, mono=True))
        self._focus_highlight(entry)
        entry.pack(fill="x", padx=12, ipady=8, pady=(0, 10))

    def _sidebar_button(self, parent: tk.Widget, text: str) -> None:
        button = tk.Button(parent, text=text, bg=BUTTON_DARK, fg=TEXT, activebackground=BUTTON_DARK_ACTIVE, activeforeground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, padx=14, pady=10, font=self._font(10, "bold"), cursor="hand2")
        self._button_feedback(button, BUTTON_DARK, BUTTON_DARK_ACTIVE)
        button.pack(fill="x", padx=12, pady=(4, 6))

    def _sidebar_scale(self, parent: tk.Widget, label: str, variable: tk.Variable, start: int, end: int) -> None:
        tk.Label(parent, text=label, bg=SURFACE, fg=MUTED, font=self._font(9, "bold")).pack(anchor="w", padx=12, pady=(0, 6))
        tk.Scale(parent, from_=start, to=end, orient="horizontal", variable=variable, bg=SURFACE, fg=TEXT, troughcolor=BUTTON_DARK, highlightthickness=0, activebackground=ACCENT, length=185).pack(anchor="w", padx=12, pady=(0, 10))

    def _sidebar_check(self, parent: tk.Widget, label: str, variable: tk.Variable) -> None:
        tk.Checkbutton(parent, text=label, variable=variable, bg=SURFACE, fg=TEXT, selectcolor=INPUT_BG, activebackground=SURFACE, activeforeground=TEXT, font=self._font(10)).pack(anchor="w", padx=12, pady=(0, 10))

    def _build_input(self, parent: tk.Frame, row: int, column: int, label: str, variable: tk.StringVar, example: str, show: str | None = None, wide: bool = False) -> None:
        field = tk.Frame(parent, bg=SURFACE)
        span = 2 if wide else 1
        padx = (0, 18) if span == 1 and column == 0 else (0, 0)
        field.grid(row=row, column=column, columnspan=span, sticky="ew", padx=padx, pady=(0, 18))
        field.grid_columnconfigure(0, weight=1)

        tk.Label(field, text=label, bg=SURFACE, fg=TEXT, font=self._font(12, "bold"), anchor="w").grid(row=0, column=0, sticky="w", pady=(0, 10))
        entry = tk.Entry(field, textvariable=variable, show=show or "", bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT, font=self._font(12, mono=True))
        self._focus_highlight(entry)
        entry.grid(row=1, column=0, sticky="ew", ipady=12)
        tk.Label(field, text=example, bg=SURFACE, fg=SUBTLE, font=self._font(10), anchor="w").grid(row=2, column=0, sticky="w", pady=(8, 0))

    def _toggle_pipeline(self) -> None:
        if self.debug:
            self._toggle_pipeline_simulation()
        else:
            self._start_full_pipeline_real()

    def _start_single_task(self, step_id: str) -> None:
        if self.debug:
            self._start_full_pipeline_simulation()
        else:
            self._start_real_tasks([step_id])

    def _task_definitions(self) -> dict[str, tuple[str, str]]:
        definitions = {step_id: (label, script) for step_id, label, script in PIPELINE_STEPS}
        definitions["watermark_detect"] = ("水印失效风险评估", "VLA-mark/detect_vq_lord_result_watermark.py")
        return definitions

    def _console_value(self, key: str) -> str:
        variable = self.console_vars.get(key)
        return str(variable.get()).strip() if variable is not None else ""

    def _form_value(self, key: str) -> str:
        variable = self.form_vars.get(key)
        return str(variable.get()).strip() if variable is not None else ""

    def _normalize_openai_base_url(self, value: str) -> str:
        text = str(value or "").strip().rstrip("/")
        for suffix in ("/chat/completions", "/responses"):
            if text.lower().endswith(suffix):
                text = text[: -len(suffix)].rstrip("/")
        return text

    def _form_env_key(self, label: str) -> str:
        key = label.split(" ", 1)[0].split("(", 1)[0].split("（", 1)[0]
        return key.replace("/", "_").replace("-", "_").strip().upper()

    def _task_env(self, step_id: str | None = None) -> dict[str, str]:
        env: dict[str, str] = {}
        root_dir = self._console_value("root_dir")
        dataset_name = self._console_value("dataset_name") or "scienceqa"
        dataset_path = self._console_value("dataset_path")
        teacher_key = self._console_value("teacher_api_key")
        teacher_base = self._normalize_openai_base_url(self._console_value("teacher_api_base"))
        judge_key = self._console_value("judge_api_key") or teacher_key
        judge_base = self._normalize_openai_base_url(self._console_value("judge_api_base") or teacher_base)
        updates = {
            "PYTHONUNBUFFERED": "1",
            "ROOT_DIR": root_dir,
            "PYTHON_BIN": self._console_value("python_bin"),
            "MODEL_PATH": self._console_value("model_path"),
            "DATASET_NAME": dataset_name,
            "DATASET_TAG": dataset_name,
            "TRAIN_DATASET_NAME": dataset_name,
            "DATASET_PATH": dataset_path,
            "SCIENCEQA_PATH": dataset_path,
            "CUDA_VISIBLE_DEVICES": self._console_value("cuda_devices"),
            "TEACHER_API_KEY": teacher_key,
            "TEACHER_API_BASE": teacher_base,
            "OPENAI_API_KEY": judge_key,
            "OPENAI_BASE_URL": judge_base,
            "VICTIM_MODEL": self._console_value("victim_model"),
            "STAGE1_CKPT_PATH": self._console_value("stage1_ckpt"),
            "STAGE2_FINAL_ADAPTER_PATH": self._console_value("stage2_adapter"),
            "ADAPTER_PATH": self._console_value("stage2_adapter"),
            "VQ_CODEBOOK_PATH": self._console_value("stage2_codebook"),
            "RESULT_DIR": self._console_value("result_dir"),
            "JUDGE_MODEL": self._console_value("judge_model"),
            "SAMPLE_NUM": self._console_value("judge_sample_num"),
            "STAGE2": self._console_value("reason_stage2_json"),
            "STAGE3": self._console_value("reason_stage3_json"),
            "TEACHER": self._console_value("reason_teacher_json"),
            "DATASET": self._console_value("reason_dataset_path"),
            "OUT_DIR": self._console_value("reason_out_dir"),
            "REQUIRE_VALID_FORMAT": "1",
        }
        env.update({key: value for key, value in updates.items() if value})
        protected_console_keys = {
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
        obsolete_form_keys = {
            "RUN_TEACHER_SPECIAL_BENCHMARKS",
            "TEACHER_BENCHMARKS",
            "TEACHER_MAX_SAMPLES_PER_BENCHMARK",
            "AI2D_DATASET",
            "AI2D_SPLIT",
            "CHARTQA_DATASET",
            "CHARTQA_SPLIT",
        }
        env.update(
            {
                key: value.get().strip()
                for key, value in self.form_vars.items()
                if value.get().strip() and key not in protected_console_keys and key not in obsolete_form_keys
            }
        )
        self._normalize_bool_env_values(env)
        if step_id == "stage1_train":
            self._map_form_env(env, "STAGE1_EPOCHS", "EPOCHS")
            self._map_form_env(env, "STAGE1_BATCH_SIZE", "BATCH_SIZE")
            self._map_form_env(env, "STAGE1_LR", "LR")
            self._map_form_env(env, "STAGE1_MAX_LENGTH", "MAX_LENGTH")
        elif step_id == "stage2_train":
            self._map_form_env(env, "STAGE2_EPOCHS", "EPOCHS")
            self._map_form_env(env, "STAGE2_LR", "LR")
            self._map_form_env(env, "STAGE2_MAX_LENGTH", "MAX_LENGTH")
        elif step_id == "student_eval":
            self._map_form_env(env, "EVAL_MAX_NEW_TOKENS", "MAX_NEW_TOKENS")
        elif step_id == "reason_judge":
            judge_dataset_name = self._form_value("JUDGE_DATASET_NAME")
            if judge_dataset_name:
                env["DATASET_NAME"] = judge_dataset_name
        return env

    def _normalize_bool_env_values(self, env: dict[str, str]) -> None:
        bool_keys = {
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
        for key in bool_keys:
            value = env.get(key)
            if value is None:
                continue
            normalized = self._normalize_bool_text(value)
            if normalized is not None:
                env[key] = normalized

    def _normalize_bool_text(self, value: object) -> str | None:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return "1"
        if text in {"0", "false", "no", "n", "off"}:
            return "0"
        return None

    def _map_form_env(self, env: dict[str, str], source: str, target: str) -> None:
        value = env.get(source) or self._form_value(source)
        if value:
            env[target] = value

    def _resolve_task_command(self, step_id: str) -> tuple[list[str] | None, str, str]:
        definitions = self._task_definitions()
        label, script = definitions[step_id]
        root_dir = self._console_value("root_dir")
        if script == "dashboard aggregation":
            return None, root_dir, label
        if step_id == "reason_judge":
            cwd = self._console_value("reason_judge_dir")
            return ["bash", self._path_join(cwd, "run_judge.sh")], cwd, label
        if step_id == "watermark_detect":
            cwd = self._console_value("vla_mark_dir")
            output_path = self._path_join(cwd, "outputs", "watermark_detect_vqlord.json")
            result_path = os.environ.get("WATERMARK_RESULT_PATH") or self._path_join(self._console_value("result_dir"), "stage3_test_generate_vq1_bucketed_parallel.json")
            command = [
                self._console_value("python_bin") or sys.executable,
                "-u",
                self._path_join(cwd, "detect_vq_lord_result_watermark.py"),
                "--result_path",
                result_path,
                "--scienceqa_path",
                self._console_value("dataset_path"),
                "--output_path",
                output_path,
                "--model_path",
                self._console_value("model_path"),
                "--sample_size",
                "0",
            ]
            return command, cwd, label
        return ["bash", self._path_join(root_dir, script)], root_dir, label

    def _start_full_pipeline_real(self) -> None:
        self._start_real_tasks([step_id for step_id, _, _ in PIPELINE_STEPS])

    def _start_real_tasks(self, step_ids: list[str]) -> None:
        if self.real_pipeline_running:
            messagebox.showinfo("任务运行中", "已有真实任务正在运行，请等待完成。")
            return
        self._save_app_config()
        self.local_real_log_path = self._create_local_real_log_file()
        definitions = self._task_definitions()
        self.real_pipeline_rows = [
            {"id": step_id, "stage": definitions[step_id][0], "status": "pending", "progress": "0%", "script": definitions[step_id][1]}
            for step_id in step_ids
        ]
        self.real_pipeline_logs = ["[real] starting task runner"]
        if self.local_real_log_path is not None:
            self.real_pipeline_logs.append(f"[local-log] {self.local_real_log_path}")
        self.real_pipeline_progress = 0.0
        self.real_pipeline_running = True
        if self.run_pipeline_button is not None:
            self.run_pipeline_button.configure(text="Pipeline 运行中", state="disabled", bg=SELECTED_BG, fg=TEXT)
        self._refresh_pipeline_view()
        thread = threading.Thread(target=self._run_real_tasks, args=(step_ids,), daemon=True)
        thread.start()

    def _create_local_real_log_file(self) -> Path | None:
        try:
            log_dir = APP_DIR / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"real_pipeline_{timestamp}.log"
            path.write_text("[real] starting task runner\n", encoding="utf-8")
            return path
        except OSError:
            return None

    def _should_run_remote(self) -> bool:
        return bool(self.connection_command_var.get().strip())

    def _remote_shell_command(self, command: list[str], cwd: str, env: dict[str, str]) -> str:
        assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items() if value)
        command_text = " ".join(shlex.quote(part) for part in command)
        if assignments:
            command_text = f"env {assignments} {command_text}"
        return f"cd {shlex.quote(cwd)} && {command_text}"

    def _masked_env_for_log(self, env: dict[str, str]) -> dict[str, str]:
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
            if is_secret:
                masked[key] = "***"
            else:
                masked[key] = value
        return masked

    def _run_remote_command(self, command: list[str], cwd: str, env: dict[str, str]) -> int:
        target = parse_connection_command(self.connection_command_var.get().strip())
        password = self.password_var.get() or self.ssh_password
        remote_command = self._remote_shell_command(command, cwd, env)
        display_command = self._remote_shell_command(command, cwd, self._masked_env_for_log(env))
        self._append_real_log(f"[ssh] {target.username}@{target.hostname}:{target.port} $ {display_command}")
        with SSHClientManager(target, password) as ssh:
            if ssh.client is None:
                raise RuntimeError("SSH client is not connected")
            _, stdout, stderr = ssh.client.exec_command(remote_command, get_pty=True)
            channel = stdout.channel
            while not channel.exit_status_ready():
                self._drain_ssh_stream(stdout)
                self._drain_ssh_stream(stderr)
                time.sleep(0.1)
            self._drain_ssh_stream(stdout)
            self._drain_ssh_stream(stderr)
            return channel.recv_exit_status()

    def _run_remote_config_command(self, command: str):
        target = parse_connection_command(self.connection_command_var.get().strip())
        password = self.password_var.get() or self.ssh_password
        with SSHClientManager(target, password) as ssh:
            return ssh.run(command)

    def _remote_config_path(self, repo_path: str | None = None) -> str:
        root_dir = (repo_path or self._console_value("root_dir")).strip()
        return self._path_join(root_dir, ".remote-console-config.json")

    def _load_remote_config_async(self, repo_path: str) -> None:
        if not self._should_run_remote():
            return
        thread = threading.Thread(target=self._load_remote_config_worker, args=(repo_path,), daemon=True)
        thread.start()

    def _load_remote_config_worker(self, repo_path: str) -> None:
        config_path = self._remote_config_path(repo_path)
        command = f"if [ -f {shlex.quote(config_path)} ]; then cat {shlex.quote(config_path)}; else echo __REMOTE_CONFIG_MISSING__; fi"
        try:
            result = self._run_remote_config_command(command)
        except Exception as exc:
            self._append_real_log(f"[config] failed to load remote config: {exc}")
            return
        if result.exit_code != 0:
            self._append_real_log(f"[config] failed to load remote config: {result.stderr.strip() or result.stdout.strip()}")
            return
        if "__REMOTE_CONFIG_MISSING__" in result.stdout:
            self._append_real_log(f"[config] remote config missing, uploading current config to {config_path}")
            self._save_remote_config(repo_path)
            return
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            self._append_real_log(f"[config] remote config is invalid JSON: {exc}")
            return
        self.root.after(0, lambda: self._apply_remote_config_payload(payload))

    def _apply_remote_config_payload(self, payload: dict[str, object]) -> None:
        console_vars = payload.get("console_vars", {})
        if isinstance(console_vars, dict):
            for key, value in console_vars.items():
                variable = self.console_vars.get(str(key))
                if variable is None:
                    continue
                try:
                    if isinstance(variable, tk.BooleanVar):
                        variable.set(self._as_bool(value))
                    elif isinstance(variable, tk.IntVar):
                        variable.set(int(value))
                    else:
                        variable.set(str(value))
                except (tk.TclError, TypeError, ValueError):
                    continue
        form_vars = payload.get("form_vars", {})
        if isinstance(form_vars, dict):
            for key, value in form_vars.items():
                variable = self.form_vars.setdefault(str(key), tk.StringVar())
                variable.set(str(value))
        self._save_app_config()
        self._append_real_log("[config] loaded remote frontend config")

    def _save_remote_config(self, repo_path: str | None = None) -> None:
        if not self._should_run_remote():
            return
        root_dir = (repo_path or self._console_value("root_dir")).strip()
        if not root_dir:
            return
        config_path = self._remote_config_path(root_dir)
        payload = json.dumps(self._config_payload(), ensure_ascii=False, indent=2)
        command = (
            f"mkdir -p {shlex.quote(root_dir)} && "
            f"printf %s {shlex.quote(payload)} > {shlex.quote(config_path)}"
        )
        try:
            result = self._run_remote_config_command(command)
        except Exception as exc:
            self._append_real_log(f"[config] failed to save remote config: {exc}")
            return
        if result.exit_code != 0:
            self._append_real_log(f"[config] failed to save remote config: {result.stderr.strip() or result.stdout.strip()}")
        else:
            self._append_real_log(f"[config] saved remote frontend config: {config_path}")

    def _drain_ssh_stream(self, stream) -> None:
        channel = stream.channel
        while channel.recv_ready():
            chunk = channel.recv(4096).decode("utf-8", errors="replace")
            self._append_real_log_chunk(chunk)
        while channel.recv_stderr_ready():
            chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
            self._append_real_log_chunk(chunk)

    def _append_real_log_chunk(self, chunk: str) -> None:
        for line in chunk.splitlines():
            self._append_real_log(line)

    def _run_local_command(self, command: list[str], cwd: str, env: dict[str, str]) -> int:
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
            self._append_real_log(line.rstrip())
        return process.wait()

    def _run_real_tasks(self, step_ids: list[str]) -> None:
        total = max(1, len(step_ids))
        if self._should_run_remote():
            self._save_remote_config()
        for index, step_id in enumerate(step_ids):
            env = self._task_env(step_id)
            command, cwd, label = self._resolve_task_command(step_id)
            self._set_real_step_state(step_id, "running", "0%")
            if command is None:
                self._append_real_log(f"[real] {label}: no external command, marked success")
                self._set_real_step_state(step_id, "success", "100%")
                self.real_pipeline_progress = (index + 1) / total
                continue
            self._append_real_log(f"[real] start {label}: {' '.join(command)}")
            try:
                if self._should_run_remote():
                    exit_code = self._run_remote_command(command, cwd, env)
                else:
                    exit_code = self._run_local_command(command, cwd, env)
            except Exception as exc:
                self._append_real_log(f"[real] failed to start {label}: {exc}")
                self._set_real_step_state(step_id, "failed", "0%")
                break
            if exit_code != 0:
                self._append_real_log(f"[real] {label} failed with exit code {exit_code}")
                self._set_real_step_state(step_id, "failed", "0%")
                break
            self._set_real_step_state(step_id, "success", "100%")
            self.real_pipeline_progress = (index + 1) / total
        else:
            self._append_real_log("[real] all tasks completed")
        self.real_pipeline_running = False
        self.root.after(0, self._finish_real_tasks)

    def _set_real_step_state(self, step_id: str, status: str, progress: str) -> None:
        if self.real_pipeline_rows is None:
            return
        for row in self.real_pipeline_rows:
            if row.get("id") == step_id:
                row["status"] = status
                row["progress"] = progress
                break
        self.root.after(0, self._refresh_pipeline_view)

    def _append_real_log(self, line: str) -> None:
        self.real_pipeline_logs.append(line)
        self.real_pipeline_logs = self.real_pipeline_logs[-1000:]
        if self.local_real_log_path is not None:
            try:
                with self.local_real_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        self.root.after(0, self._refresh_pipeline_view)

    def _finish_real_tasks(self) -> None:
        if self.run_pipeline_button is not None:
            self.run_pipeline_button.configure(text="一键跑完整 Pipeline", state="normal", bg=ACCENT, fg="#041311")
        self._refresh_pipeline_view()
        self._refresh_dashboard_async()

    def _real_pipeline_progress(self) -> tuple[float, list[dict[str, str]]]:
        if self.real_pipeline_rows is not None:
            return self.real_pipeline_progress, self.real_pipeline_rows
        rows = [
            {"stage": label, "status": "pending", "progress": "0%", "script": script}
            for _, label, script in PIPELINE_STEPS
        ]
        return 0.0, rows

    def _toggle_pipeline_simulation(self) -> None:
        if self.pipeline_task is None:
            self._start_full_pipeline_simulation()
            return

        progress, _ = self._pipeline_progress()
        if progress >= 1.0:
            self._start_full_pipeline_simulation()
            return

        if bool(self.pipeline_task.get("paused")):
            self._resume_pipeline_simulation()
        else:
            self._pause_pipeline_simulation()

    def _start_full_pipeline_simulation(self) -> None:
        per_step_seconds = max(1, int(self.console_vars["sim_duration"].get()))
        self.pipeline_task = {
            "started_at": time.time(),
            "duration": float(per_step_seconds * len(PIPELINE_STEPS)),
            "per_step": float(per_step_seconds),
            "paused": False,
            "paused_elapsed": 0.0,
        }
        if self.run_pipeline_button is not None:
            self.run_pipeline_button.configure(text="暂停 Pipeline（仿真）", state="normal", bg=SELECTED_BG, fg=TEXT)
        self._refresh_pipeline_view()
        self._schedule_pipeline_refresh()

    def _pause_pipeline_simulation(self) -> None:
        if self.pipeline_task is None:
            return
        elapsed = max(0.0, time.time() - self.pipeline_task["started_at"])
        self.pipeline_task["paused"] = True
        self.pipeline_task["paused_elapsed"] = elapsed
        if self.run_pipeline_button is not None:
            self.run_pipeline_button.configure(text="继续 Pipeline（仿真）", state="normal", bg=ACCENT, fg="#041311")
        self._refresh_pipeline_view()

    def _resume_pipeline_simulation(self) -> None:
        if self.pipeline_task is None:
            return
        paused_elapsed = float(self.pipeline_task.get("paused_elapsed", 0.0))
        self.pipeline_task["started_at"] = time.time() - paused_elapsed
        self.pipeline_task["paused"] = False
        if self.run_pipeline_button is not None:
            self.run_pipeline_button.configure(text="暂停 Pipeline（仿真）", state="normal", bg=SELECTED_BG, fg=TEXT)
        self._refresh_pipeline_view()
        self._schedule_pipeline_refresh()

    def _pipeline_progress(self) -> tuple[float, list[dict[str, str]]]:
        if not self.debug:
            return self._real_pipeline_progress()
        if self.pipeline_task is None:
            rows = [
                {"stage": label, "status": "pending", "progress": "0%", "script": script}
                for _, label, script in PIPELINE_STEPS
            ]
            return 0.0, rows

        paused = bool(self.pipeline_task.get("paused"))
        elapsed = float(self.pipeline_task.get("paused_elapsed", 0.0)) if paused else max(0.0, time.time() - self.pipeline_task["started_at"])
        duration = max(0.001, self.pipeline_task["duration"])
        per_step = max(0.001, self.pipeline_task["per_step"])
        total_progress = min(1.0, elapsed / duration)
        rows = []
        for index, (_, label, script) in enumerate(PIPELINE_STEPS):
            local_elapsed = elapsed - index * per_step
            if local_elapsed <= 0:
                status = "pending"
                progress = 0.0
            elif local_elapsed >= per_step:
                status = "success"
                progress = 1.0
            else:
                status = "paused" if paused else "running"
                progress = min(1.0, local_elapsed / per_step)
            rows.append(
                {
                    "stage": label,
                    "status": status,
                    "progress": f"{progress * 100:.0f}%",
                    "script": script,
                }
            )
        return total_progress, rows

    def _refresh_pipeline_view(self) -> None:
        progress, rows = self._pipeline_progress()
        if self.task_progress is not None:
            self.task_progress["value"] = progress * 100

        if not self.debug:
            if self.real_pipeline_rows is None:
                self.task_summary_var.set("暂无真实任务")
            elif self.real_pipeline_running:
                running = next((row["stage"] for row in rows if row["status"] == "running"), "准备中")
                self.task_summary_var.set(f"真实任务运行中：{progress * 100:.0f}% · 当前阶段：{running}")
            elif any(row["status"] == "failed" for row in rows):
                failed = next((row["stage"] for row in rows if row["status"] == "failed"), "未知阶段")
                self.task_summary_var.set(f"真实任务失败：{failed}")
            else:
                self.task_summary_var.set(f"真实任务完成：{progress * 100:.0f}%")
        elif self.pipeline_task is None:
            self.task_summary_var.set("暂无仿真任务")
        elif progress >= 1.0:
            self.task_summary_var.set("完整 Pipeline 仿真完成：100%")
            if self.run_pipeline_button is not None:
                self.run_pipeline_button.configure(text="一键跑完整 Pipeline（仿真）", state="normal", bg=ACCENT, fg="#041311")
        elif bool(self.pipeline_task.get("paused")):
            paused_stage = next((row["stage"] for row in rows if row["status"] == "paused"), "准备中")
            self.task_summary_var.set(f"完整 Pipeline 仿真已暂停：{progress * 100:.0f}% · 当前阶段：{paused_stage}")
        else:
            running = next((row["stage"] for row in rows if row["status"] == "running"), "准备中")
            self.task_summary_var.set(f"完整 Pipeline 仿真运行中：{progress * 100:.0f}% · 当前阶段：{running}")

        if self.task_tree is not None:
            for item_id in self.task_tree.get_children():
                self.task_tree.delete(item_id)
            for index, row in enumerate(rows):
                stripe = "odd" if index % 2 else "even"
                self.task_tree.insert("", "end", values=(row["stage"], row["status"], row["progress"], row["script"]), tags=(stripe, row["status"]))

        if self.task_log_box is not None:
            lines = self._pipeline_log_lines(progress, rows)
            first, last = self.task_log_box.yview()
            should_follow = last >= 0.98
            self.task_log_box.configure(state="normal")
            self.task_log_box.delete("1.0", "end")
            self.task_log_box.insert("end", "\n".join(lines))
            self.task_log_box.configure(state="disabled")
            if should_follow:
                self.task_log_box.see("end")
            else:
                self.task_log_box.yview_moveto(first)

    def _pipeline_log_lines(self, progress: float, rows: list[dict[str, str]]) -> list[str]:
        if not self.debug:
            return self.real_pipeline_logs or ["No real tasks yet."]
        if self.pipeline_task is None:
            return ["No simulated tasks yet."]
        lines = [
            "[simulate] mode=sequential full pipeline",
            f"[simulate] progress={progress * 100:.0f}%",
        ]
        for row in rows:
            lines.append(f"[simulate] {row['status']:>7} {row['progress']:>4} :: {row['script']}")
        if progress >= 1.0:
            lines.append("[simulate] completed successfully")
        return lines

    def _schedule_pipeline_refresh(self) -> None:
        if self.pipeline_task is None:
            return
        progress, _ = self._pipeline_progress()
        if progress >= 1.0 or bool(self.pipeline_task.get("paused")):
            self._refresh_pipeline_view()
            return
        if not bool(self.console_vars["auto_refresh"].get()):
            return
        self.pipeline_refresh_after_id = self.root.after(1000, self._pipeline_refresh_tick)

    def _pipeline_refresh_tick(self) -> None:
        self._refresh_pipeline_view()
        self._schedule_pipeline_refresh()

    def _start_clone(self) -> None:
        request = CloneRequest(
            connection_command=self.connection_command_var.get().strip(),
            project_path=self.project_var.get().strip(),
            password=self.password_var.get(),
        )
        if not request.connection_command or not request.project_path or not request.password:
            messagebox.showwarning("参数不完整", "请填写服务器连接命令、项目地址和密码。")
            return

        self.ssh_password = request.password
        self._set_busy(True)
        self._save_recent_config()
        thread = threading.Thread(target=self._run_clone, args=(request,), daemon=True)
        thread.start()

    def _run_clone(self, request: CloneRequest) -> None:
        result = self.clone_service.clone_repository(request)
        self.root.after(0, lambda: self._finish_clone(result.success, result.message, result.details))

    def _finish_clone(self, success: bool, message: str, details: str) -> None:
        self._set_busy(False)
        self.status_var.set(message)
        if success:
            self.password_var.set("")
            self._show_console_screen(details or f"{self.project_var.get().strip()}/{REPO_DIR_NAME}")
        else:
            messagebox.showerror("执行失败", f"{message}\n\n{details}".strip())

    def _show_console_screen(self, repo_path: str) -> None:
        self._apply_repo_path_defaults(repo_path)
        if self.console_repo_path is not None:
            self.console_repo_path.configure(text=repo_path)
        if self.console_frame is not None:
            self.console_frame.tkraise()
        self._load_remote_config_async(repo_path)

    def _show_main_screen(self) -> None:
        self.status_var.set("准备就绪")
        if self.main_frame is not None:
            self.main_frame.tkraise()

    def _set_busy(self, busy: bool) -> None:
        if self.clone_button is not None:
            self.clone_button.configure(state="disabled" if busy else "normal")
            self.clone_button.configure(bg=ACCENT if not busy else SELECTED_BG, fg="#041311" if not busy else TEXT)
        self.status_var.set("正在连接服务器并执行 clone..." if busy else "准备就绪")

    def _save_recent_config(self) -> None:
        self._save_app_config()

    def _variable_snapshot(self, variable: tk.Variable) -> object:
        value = variable.get()
        if isinstance(variable, tk.BooleanVar):
            return bool(value)
        if isinstance(variable, tk.IntVar):
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        return str(value)

    def _config_payload(self) -> dict[str, object]:
        obsolete_form_keys = {
            "RUN_TEACHER_SPECIAL_BENCHMARKS",
            "TEACHER_BENCHMARKS",
            "TEACHER_MAX_SAMPLES_PER_BENCHMARK",
            "AI2D_DATASET",
            "AI2D_SPLIT",
            "CHARTQA_DATASET",
            "CHARTQA_SPLIT",
        }
        return {
            "connection_command": self.connection_command_var.get().strip(),
            "project_path": self.project_var.get().strip(),
            "console_vars": {key: self._variable_snapshot(variable) for key, variable in self.console_vars.items()},
            "form_vars": {key: variable.get() for key, variable in self.form_vars.items() if key not in obsolete_form_keys},
        }

    def _save_app_config(self) -> None:
        payload = self._config_payload()
        save_config(
            AppConfig(
                connection_command=str(payload["connection_command"]),
                project_path=str(payload["project_path"]),
                console_vars=dict(payload["console_vars"]),
                form_vars=dict(payload["form_vars"]),
            )
        )

    def _schedule_config_save(self) -> None:
        if not hasattr(self, "root"):
            return
        if self.config_save_after_id is not None:
            self.root.after_cancel(self.config_save_after_id)
        self.config_save_after_id = self.root.after(500, self._flush_scheduled_config_save)

    def _flush_scheduled_config_save(self) -> None:
        self.config_save_after_id = None
        self._save_app_config()

    def _on_close(self) -> None:
        if self.config_save_after_id is not None:
            self.root.after_cancel(self.config_save_after_id)
            self.config_save_after_id = None
        self._save_app_config()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch_app(debug: bool = False) -> None:
    MainWindow(debug=debug).run()
