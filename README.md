# MLLM 能力泄漏风险检测平台

本项目是运行在笔记本或工作站上的实验控制前端，用于通过 SSH 管理远程
`CISCN-MICAD` 仓库、配置实验参数、执行能力泄漏检测 Pipeline，并查看风险报告、
水印结果和历史归档。

Web 前端是当前主要界面；Tk 前端作为兼容入口保留。

## 当前功能

- 连接远程 Linux 服务器，并在指定目录 clone 或进入固定仓库：
  `https://gitee.com/yang-wenxiao-111/CISCN-MICAD.git`
- 根据 `CISCN-MICAD` 仓库根目录自动推导脚本、检查点、结果和评测路径。
- 在程序或 EXE 所在目录附近检测到 `CISCN-MICAD` 时，直接进入实验界面。
- 在本地和远程服务器之间保存、读取前端实验配置。
- 执行完整 Pipeline，或单独运行指定阶段。
- 复用已有教师标注 JSON 和教师风险基线 JSON，跳过对应阶段。
- 实时显示阶段进度和终端日志，并将真实运行日志保存到本机。
- 归档实验结果，在前端选择、查看和删除历史记录。
- 聚合 ACC、VR、CoT 三个维度的能力泄漏风险：

  ```text
  CLR = 0.50 * ACC + 0.30 * VR + 0.20 * CoT
  ```

- 运行当前 VLA-Mark 水印检测，展示 z-score 统计和水印削弱风险。
- 提供 Ask / Agent 两种 AI 助手模式，支持流式回答、折叠思考过程、Markdown
  表格、参数修改、配置检查、结果读取和 Pipeline 执行确认。
- 使用 `--debug` 启动前端仿真，演示进度、日志和结果，不连接真实训练后端。

## Pipeline

完整 Pipeline 当前包含以下阶段：

1. 教师 API 数据采集
2. 教师完整风险基线
3. 原始学生风险基线
4. Stage1 学生蒸馏
5. Stage2 学生蒸馏
6. 学生完整风险评估
7. 思维链评估
8. 风险报告聚合

水印检测是独立任务，不属于上述八阶段主 Pipeline。

前端复用远程仓库现有的 `scripts2/`、`reason_judge/`、`vq_lord3/` 和
`VLA-mark/`，不会在笔记本上重新实现训练与评测逻辑。

## 环境要求

本地前端建议使用 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Tkinter 由 Python 发行版或操作系统提供，不包含在 `requirements.txt` 中。Ubuntu
需要 Tk 前端时可安装：

```bash
sudo apt install python3-tk
```

远程服务器需要提前准备：

- 可用的 SSH 连接和认证方式
- `git`、`bash` 和实验所需 Python/Conda 环境
- 模型、ScienceQA 数据集和相应检查点
- `CISCN-MICAD` 后端依赖
- 教师模型、Judge 或 AI 助手所需的 API 地址和密钥

## 启动方式

### Web 前端

```bash
python main.py
```

默认绑定 `127.0.0.1:8011`，并自动打开浏览器。端口被占用或需要自行选择端口时：

```bash
python main.py --host 127.0.0.1 --port 9000 --no-browser
```

如果需要从局域网访问，显式绑定 `0.0.0.0`，并自行配置防火墙和访问控制：

```bash
python main.py --host 0.0.0.0 --port 9000 --no-browser
```

### Debug 仿真

```bash
python main.py --debug
```

Debug 模式只生成前端仿真进度、日志和演示结果，不会启动远程训练或评测。

### Tk 兼容前端

```bash
python main.py --tk
```

Tk 前端没有同步 Web 端的 Agent 和部分新交互，新增功能默认以 Web 端为准。

### 命令行参数

```text
--debug       使用仿真 Pipeline
--tk          启动 Tk 前端
--host HOST   Web 绑定地址，默认 127.0.0.1
--port PORT   Web 绑定端口，默认 8011
--no-browser  启动后不自动打开浏览器
```

## 首次配置

1. 填写 SSH 连接命令，例如：

   ```text
   ssh root@example.com -p 22
   ```

2. 填写远程项目父目录，例如 `/root/workspace`。
3. 使用“Clone/进入”创建或进入 `CISCN-MICAD`。
4. 检查前端自动推导的 `ROOT_DIR`、Python、模型、数据集、检查点和结果路径。
5. 配置教师 API、Judge API、样本数量和训练参数。
6. 将配置保存到服务器，或读取服务器上已有的配置。
7. 先用少量样本验证单阶段，再执行完整 Pipeline。

如果仓库已经存在，可直接填写 SSH 和仓库路径，无需重复 clone。程序在本地启动目录
附近发现名为 `CISCN-MICAD` 的仓库时，也会自动使用该仓库推导默认路径。

## 配置和本地数据

前端数据默认保存在运行前端的电脑上：

```text
~/.remote-clone-tool/
  config.json              # 当前前端配置
  logs/                    # 真实 Pipeline 本地日志
  history/                 # 实验结果历史归档
  assistant_sessions/      # AI 助手会话
```

保存远程配置后，服务器仓库中还会生成：

```text
<ROOT_DIR>/.remote-console-config.json
```

历史归档会移除 API Key；AI 助手会话不会保存 API Key 或 SSH 密码。当前本地配置和
远程配置是普通 JSON 文件，可能包含 API Key，请限制文件权限，不要提交到 Git 或发送给
无关人员。SSH 密码只保存在当前进程内。

## AI 助手

助手设置位于悬浮窗口内，使用 OpenAI-compatible API：

```text
Base URL
API Key
Model（默认 dpsk-v4-flash）
启用思考
```

### Ask 模式

用于解释平台功能、参数、运行流程和报错，不调用平台工具。

### Agent 模式

当前提供以下受限能力：

- 读取脱敏后的应用状态
- 读取 Pipeline 状态和限定长度的日志
- 读取风险、水印和历史结果摘要
- 检查 SSH、模型、数据集、Python、检查点和关键 JSON
- 修改白名单参数
- 应用 `demo`、`recommended`、`low_memory` 和“演示推荐参数”预设
- 提议运行完整或部分 Pipeline
- 提议同步远程配置
- 创建历史归档

参数修改可以直接保存；启动 Pipeline 和同步服务器配置必须由用户点击确认。Agent
不提供任意 Shell、任意文件写入、服务器源码修改或密钥读取能力。查看历史归档时，
Agent 自动进入只读状态。

## 教师结果复用

数据配置页支持复用：

- 已有教师标注 JSON
- 已有教师风险基线 JSON

路径应填写服务器上的绝对路径。启动 Pipeline 时，前端检查文件存在后跳过对应阶段。
复用文件与默认目标文件相同时不会重复复制。

## 风险评估

风险大盘使用三个维度：

- `ACC`：学生相对原始模型和教师模型的能力增益/迁移程度
- `VR`：教师与学生在视觉控制实验中的下降模式相似度
- `CoT`：思维链 Judge 对视觉依据、逻辑正确性、答案支持等维度的评价

当部分维度缺失时，平台只在可用维度上按覆盖权重重新归一化，并显示缺失维度和置信信息。

水印削弱风险使用教师水印分数、学生水印分数和可选无水印基线计算。当前自动检测仍绑定
`VLA-mark/detect_vq_lord_result_watermark.py`，使用 z-score 和 `z > 4` 检出率。通用私有
水印 JSON/脚本接口尚未实现。

## 水印检测

当前水印页输入学生生成结果 JSON，并调用远程 VLA-Mark 检测器。主要配置包括：

- 学生四字段回答 JSON
- 抽样条数或抽样比例
- VLA-Mark Python 环境
- 模型名称、设备和 torch dtype
- 教师、学生和可选无水印基线分数

输出默认写入：

```text
<VLA_MARK_DIR>/outputs/watermark_detect_vqlord.json
```

不同模型采用私有水印时，应由对应水印提供方先生成分数；不要假设所有水印都适用当前
VLA-Mark 的 z-score 和阈值。

## 项目结构

```text
.
├── app/
│   ├── config.py                 # 本地配置持久化
│   ├── models.py                 # 基础数据模型
│   ├── services/
│   │   ├── git_service.py        # 远程 clone
│   │   └── ssh_client.py         # SSH 命令执行
│   ├── ui/
│   │   └── main_window.py        # Tk 兼容前端
│   └── web/
│       ├── server.py             # FastAPI、Web UI、任务调度与历史归档
│       ├── risk_evaluation.py    # ACC/VR/CoT/WER 聚合
│       ├── agent_service.py      # PydanticAI Agent、流式响应和会话
│       ├── agent_tools.py        # Agent 白名单工具和预设
│       └── agent_schemas.py      # Agent 请求、响应和确认动作模型
├── test/                         # 回归测试
├── main.py                       # 程序入口
├── requirements.txt
├── RemoteCloneTool.spec          # PyInstaller 配置
└── build_exe.bat
```

## 测试

运行全部 unittest：

```bash
PYTHONPATH=. python -m unittest discover -s test -p "test_*.py"
```

也可以只运行 Agent 测试：

```bash
PYTHONPATH=. python test/test_agent.py
```

## 打包为 Windows EXE

建议使用仓库中的 spec 文件，确保包含 Web Agent 所需依赖：

```powershell
python -m pip install -r requirements.txt
pyinstaller --noconfirm --clean RemoteCloneTool.spec
```

输出文件位于：

```text
dist\RemoteCloneTool.exe
```

将 EXE 放在 `CISCN-MICAD` 文件夹旁边时，程序会尝试自动识别仓库并进入实验界面。

## 已知边界

- Web 是主要维护目标；Tk 不保证与 Web 功能完全一致。
- 真实训练和评测依赖远程 `CISCN-MICAD` 后端及其运行环境。
- 当前水印自动检测仅支持仓库内的 VLA-Mark 流程。
- Agent 依赖模型服务正确支持 OpenAI-compatible Tool Calling；供应商字段差异可能需要适配。
- 前端不会自动安装远程模型、数据集、CUDA 或训练环境。
- 程序不提供任务强制终止、任意远程 Shell 或自动 OOM 重试。
