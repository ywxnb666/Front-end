"""Static product knowledge for the in-app assistant.

This is deliberately kept separate from the model prompt so the assistant's
answering rules and the product facts can evolve independently. Runtime state
and secrets must still come from the existing redacted snapshot/tools.
"""

APPLICATION_KNOWLEDGE = r"""
【平台知识库：事实优先】

一、平台定位和页面结构
本应用的名称是“MLLM能力泄漏风险检测平台”，Web 页面用于连接远程实验服务器、配置参数、运行 MICAD/VQ-LoRD Pipeline、查看日志、查看风险评估、水印检测结果以及回看历史归档。当前主要支持 Web 前端；不要声称存在一个独立的“历史归档专用页面”。历史归档选择器就在左侧栏的“历史测评数据”区域。

页面的主要区域包括：
1. 左侧配置栏：服务器连接、全局路径、教师/评测 API、模型和检查点、训练/评测参数、教师标注与教师基线复用、水印参数、历史测评数据。
2. 右侧主区域：任务状态和日志、Stage1/Stage2 蒸馏、完整评测、风险大盘、思维链评估、水印检测等标签页。
3. 右下角 AI 助手：Ask 用于解释和排错，Agent 用于受控读取状态、修改白名单参数、生成待确认任务。

二、数据位置和路径语义
必须区分两类路径：
1. 前端本机路径：配置文件、助手会话和历史归档保存在运行前端的笔记本上。默认目录是 ~/.remote-clone-tool/；配置是 ~/.remote-clone-tool/config.json，历史归档是 ~/.remote-clone-tool/history/，助手会话是 ~/.remote-clone-tool/assistant_sessions/。这些文件不是服务器文件。
2. 远程服务器路径：当用户通过 SSH 连接后，root_dir、dataset_path、model_path、stage1_ckpt、stage2_adapter、result_dir、reason_judge_dir 等路径指向服务器。前端通过 SSH 在服务器上执行脚本、读取日志和结果，笔记本不能直接用文件管理器打开这些 Linux 路径。
因此，看到 /root/...、/home/...、/media/... 等路径时，先说明它们属于远程服务器；看到 ~/.remote-clone-tool/ 时，才是前端笔记本本地路径。不要建议用户在笔记本上直接打开服务器路径。需要确认远程文件是否存在时，应使用 Agent 的 validate_remote_setup 工具或让用户查看系统后台日志。

三、历史归档的准确用法
历史归档不是重新运行实验，也不是服务器结果目录的浏览器。归档是当前前端把当时的配置快照、风险大盘、水印摘要、Pipeline 阶段状态和有限日志保存到本机 ~/.remote-clone-tool/history/ 的 JSON 文件。
使用方式：
1. 在左侧“历史测评数据”中打开“选择归档记录”。
2. 选择记录后，页面进入历史回看模式，右侧数据被冻结，运行、连接和修改相关按钮会锁定。
3. 点击“返回默认数据”退出回看，页面恢复当前实时状态并重新轮询。
4. “归档当前结果”只创建本地快照，不重新运行 Pipeline；“删除选中记录”删除本地归档文件。
5. Pipeline 完成且存在可归档结果时，系统通常会自动创建一份归档；也可以手动归档。
6. 归档中会脱敏 API Key、SSH 密码等敏感值。因此归档不是完整凭据备份。
回答“如何查看历史归档”时，应直接说明以上左侧选择器和本机目录，不要回答“没有历史归档页面”。如果用户想查看服务器上的原始 JSON，则应说明那是远程结果文件，使用读取结果/日志工具或 SSH 检查，不要把它与本机归档混为一谈。

四、Pipeline 阶段
当前标准阶段 ID 和含义如下：
1. teacher_collect：教师 API 数据采集，生成/复用教师结构化标注 JSON。
2. teacher_eval：教师完整风险基线，评测教师在控制集上的能力和视觉依赖。
3. origin_eval：原始学生能力基线，评测未蒸馏学生。
4. stage1_train：Stage1 学生蒸馏，通常学习教师观察事实、上下文、思维和答案字段。
5. stage2_train：Stage2 学生蒸馏/MLoRD 或 VQ-LoRD 训练。
6. student_eval：学生完整风险评估，生成学生控制集、阶段输出和风险所需结果。
7. reason_judge：思维链评估，比较教师/学生推理质量；需要正确的 stage2/stage3 JSON、数据集、split、judge API。
8. risk_report：风险报告聚合，汇总 ACC、VR、CoT、水印等维度。
用户只想查看结果时不应启动 Pipeline。完整或部分 Pipeline 都是长时间、可能占用 GPU/API 的操作，Agent 只能先提出计划并等待确认。

五、关键参数规则
1. 教师采样数量：通常同时设置 TRAIN_NUM 和 MAX_SAMPLES；只改其中一个可能导致采集量与后续阶段不一致。
2. 思维链评估抽样数量：使用 judge_sample_num；它不等于教师采样数量。
3. Stage1/Stage2 batch size：单卡显存不足时保持 batch size=1，使用 gradient accumulation、USE_4BIT、FREEZE_VISION_TOWER 和较短 MAX_LENGTH；gradient accumulation 不会降低单个样本的峰值显存，但能保持有效 batch。
4. CUDA_VISIBLE_DEVICES 与 DDP_NPROC：可见 GPU 数少于 DDP 进程数时会启动失败，单卡应将 DDP_NPROC 设为 1 或使用单进程回退。
5. STRICT_TEACHER_DISTILL：脚本参数需要整数 0/1，不要传 Python 字符串 True/False。
6. 数据集：dataset_name、DATASET_NAME、SCIENCEQA_SPLIT、SPLIT、JUDGE_DATASET_NAME 要保持一致；ScienceQA 与 AOKVQA 不能混用路径、控制集或输出文件名。
7. 教师标注 JSON 和教师风险基线 JSON 可以复用。复用时应确认文件存在、数据集和 split 一致，并避免把同一个源文件复制到自身。
8. API Key 不应在回答、日志或工具结果中显示；Base URL 可以说明，但不要猜测 URL 是否有效，应通过实际请求或配置检查确认。

六、风险评估口径
平台完整风险报告优先使用 capability leakage 标准，而不是只看教师 ACC 和学生 ACC。核心展示维度包括：ACC（正确率迁移）、VR（视觉依赖）、CoT（思维链能力泄漏）和 WER/水印维度。当前大盘中的 CLR、risk、risk_level、confidence、coverage 等字段以实际聚合结果为准；缺失维度应显示未测量，不要自行补数。
如果用户问“为什么没有分数”，先检查对应阶段、结果 JSON、数据集/split、样本数量和聚合日志。没有有效评分样本时不能把失败当成 0 分。

七、水印检测
水印 z 分数通常越高表示越接近检测到水印，阈值 4 的检出率是超过 4 的样本比例；但不同水印方法的分数方向可能不同，通用水印计算器不能直接替代具体检测器。需要区分教师、Stage1、Stage2 和原始学生的角色，不能因为文件名含 teacher_cache 就称为学生结果。
VLA-Mark 的 delta=0.0 通常表示零强度/clean 条件；这类数据可作为无水印基线，但若没有学生模型或学生评测输出，不能称为“学生数据”。水印检测 JSON 中的 cache_path、model_path 和 output_path 必须结合脚本语义判断角色。

八、AI 助手自身
Ask 模式不调用工具，只解释本平台事实、参数和错误；Agent 模式可调用受控工具。读取状态、日志和结果可以直接执行；白名单参数修改可直接保存到本地配置；同步远程配置、运行 Pipeline 等操作必须返回待确认卡片，用户确认后才执行。
思考策略由模式固定决定：Ask 对 dpsk-v4-flash 发送 thinking.type=disabled；Agent 发送 thinking.type=enabled 和 reasoning_effort=high。界面不提供独立思考开关。思考内容与最终答案分离，前端默认折叠。若模型把思考直接写进普通文本，助手应清理 <think> 标签，但不能把普通回答中的分析段臆测为供应商思考。

九、回答原则
先使用知识库回答确定事实，再结合当前脱敏状态和工具结果回答动态状态。当前状态摘要中的 history_archive.count 是本机实际归档数量，recent_items 是最近归档；count 大于 0 时绝不能说“当前没有历史归档”。对“当前”“这里”“为什么”这类问题，不能只引用默认值；应先读取状态。上下文没有提供某个字段只代表信息不足，不代表该对象不存在。知识库和动态状态冲突时，以动态状态为准，并明确说明。无法确认时说“当前信息不足，需要读取某项状态”，不要编造路径、阶段完成情况、分数或页面。
""".strip()
