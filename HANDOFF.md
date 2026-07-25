# 交接文档 — StageDouble Prototype vol.1

> 写给完全没有上下文的新对话看。开始工作前请先完整读完这份文档。

## 一句话summary

这是 `Pengkai-Gao-AMLC-Final-Project` 仓库，正在做的是 **StageDouble**：一条纯 Python 的实时管线，从麦克风采集人声特征 → 用一个可训练的小型回归模型 → 输出参数去驱动一个数字人（digital human）。当前所在阶段是 **prototype vol.1**。

之所以是"纯 Python"，是因为原计划用 Wekinator 做交互式机器学习（IML）部分，但 **Wekinator 的 Java Swing 界面在 macOS Sonoma 上会因为 Carbon 菜单不兼容而崩溃**，所以自己写了一个终端版的替代品 `ml_trainer.py`。

## 已经完成了什么

代码已经**写完**，目录结构如下（都在 `prototype vol.1/` 下，注意路径里有空格，shell 里要加引号）：

- `audio/audio_extractor.py` — 采集麦克风音频，用 librosa 提取音高（pitch）、音量、13 维 MFCC、节奏（tempo），共 16 维特征向量，按 10Hz 通过 OSC 发送到 `127.0.0.1:6448 /wek/inputs`。
- `audio/ml_trainer.py` — 监听上面那条 OSC 流，提供一个终端命令行（`record` / `train` / `run` / `stop` / `status` / `clear` / `help` / `quit`），用 scikit-learn 的 `MLPRegressor`（单隐层 10 个神经元 + StandardScaler + lbfgs solver）做交互式训练，训练完后把 2 维预测输出实时发到 `127.0.0.1:12000 /stagedouble/outputs`。
- `audio/osc_listener.py` — 调试用的诊断工具，监听同一个输入地址并把数值打印出来，用来单独验证 `audio_extractor.py` 有没有正常工作（**只能替代 `ml_trainer.py` 运行，不能和它同时跑**，两者抢同一个端口 6448）。
- `requirements.txt` — 依赖：`librosa numpy sounddevice python-osc scikit-learn`。
- `README.md` — 已经写好，包含架构图（mermaid）、安装、使用说明。
- 每个 `.py` 文件顶部都有中英双语的模块文档字符串，解释这个文件做什么、怎么运行、跟谁连接。

**依赖已在本机确认安装**（`/opt/miniconda3` 的 base conda 环境，不是项目专属 venv）：librosa 0.11.0、numpy 2.4.1、python-osc 1.10.2、scikit-learn 1.7.2、sounddevice 0.5.5，以及 pyin 需要的 numba 0.65.1 / llvmlite 0.47.0 也都在。

## 现在卡在哪 / 尚未验证

**代码从写完到现在，还没有实际跑过一次完整流程。** 依据：
- 没有任何 `__pycache__`，说明这几个 `.py` 文件从来没被 Python 执行过。
- 三个源文件的 mtime 都停在 7 月 17 日（写作时间），今天是 7 月 25 日，中间没有任何运行痕迹。

所以以下这些都还只是"设计上应该没问题"，**没有实测验证过**：
- macOS 麦克风权限弹窗流程能不能顺利跑通。
- `audio_extractor.py` 实际发出的特征数值是否合理（pitch/volume/MFCC/tempo 的量级是否符合预期）。
- `osc_listener.py` 能否收到并正确打印。
- `ml_trainer.py` 的 record → train → run 完整闭环，包括后台线程和终端 `input()` 是否会互相卡住。
- 实时性能是否真的能撑住 10Hz（README 里说 M2 上 `librosa.pyin()` 单次调用有约 50ms 的固定下限，是靠经验估算的，不是这台机器上实测的）。

另外，**下游消费者（"digital human / synth"）目前完全不存在**——`ml_trainer.py` 会把预测结果发到 `127.0.0.1:12000 /stagedouble/outputs`，但没有任何东西在监听这个地址。这在 prototype vol.1 的范围里是预期之内的（README 里写的是"whatever consumes them"），但如果后面要验证端到端效果，需要额外写一个接收端或者用 `osc_listener.py` 改一份来验证。

Git 状态：**目前完全没有提交任何跟这个 prototype 相关的 commit**。远程仓库只有一个 "Initial commit"（只有一行 README）。工作区状态：
- `README.md` → `prototype vol.1/README.md`（rename，已 staged）
- `prototype vol.1/README.md` 内容相对 staged 版本又有改动（unstaged）
- `prototype vol.1/audio/`、`prototype vol.1/requirements.txt`（未跟踪 untracked）

也就是说所有代码都还只在工作区里，随时可能因为误操作丢失，还没有落进 git 历史。

## 下一步计划

1. 先跑通端到端流程验证代码可用性，建议顺序：
   - `python "prototype vol.1/audio/audio_extractor.py" --list-devices` 确认麦克风设备号。
   - 单独跑 `audio_extractor.py`，另开一个终端跑 `osc_listener.py`，确认能收到合理数值（此时**不要**同时跑 `ml_trainer.py`）。
   - 确认无误后，改成跑 `ml_trainer.py` 代替 `osc_listener.py`，走一遍 `record` → `train` → `run` 的完整命令序列，观察终端打印的实时预测输出是否合理、`stop`/`quit` 能否正常退出。
2. 跑通后，把 `prototype vol.1/` 下的所有改动（README rename + 内容改动 + audio/ + requirements.txt）整理成一次 git commit。
3. 视验证结果决定是否需要调参（比如 `ml_trainer.py` 里的隐藏层大小、`audio_extractor.py` 里的 `HOP_DURATION`），或者开始搭建下游"digital human"消费端来验证 `/stagedouble/outputs` 真正被使用起来的效果。

## 踩过的坑 / 绝对不要再踩

- **不要把 Wekinator 加回流程。** 它在 macOS Sonoma 上因为 Carbon 菜单不兼容会直接崩溃，这正是自己写 `ml_trainer.py` 的原因。
- **`ml_trainer.py` 和 `osc_listener.py` 不能同时跑。** 两者默认都监听 `127.0.0.1:6448`，端口会冲突。
- **不要把 `HOP_DURATION`（10Hz）调得更快而不做实测。** `librosa.pyin()` 在 M2 上单次调用有约 50ms 的固定处理下限（跟输入音频长短无关，是它内部 `frame_length=2048` 窗口决定的），100ms 的间隔是留了余量算出来的，调快之前必须先跑通 `QUEUE_BACKLOG_WARNING` 观察是否积压。
- **不要删掉 `audio_extractor.py` 里的 "warm-up" 调用**（`main()` 里在打开麦克风流之前先空跑一次 `extract_features()`）。这是为了把 `librosa.pyin()` 首次调用触发的 numba JIT 编译耗时（1 秒多）提前消化掉，否则实时循环刚启动就会积压。
- **不要跳过 `StandardScaler`。** 16 维特征里数值量级差异极大（pitch 几十~几千 Hz，volume 0~1，MFCC -400~400），不标准化的话大数值特征会主导训练、掩盖其他特征。
- **不要把 `MLPRegressor` 的 solver 从 `lbfgs` 换回默认的 `adam`。** IML 现场录制的数据集通常很小、一次性（非流式），`lbfgs` 收敛更稳定可靠；`adam` 这类随机梯度方法在这种小数据集上表现不好。
- **不要用 `-uall` 之类的方式看 git 未跟踪文件**（仓库可能很大，容易引发内存问题——这是通用注意事项，本仓库目前体量小但养成习惯）。
- **路径里有空格**：项目目录名是 `prototype vol.1`，命令行操作这个目录下的文件时必须用引号包住路径，否则会被 shell 拆成两个参数。
- **依赖装在 base conda 环境里，不是项目专属 venv。** 如果之后换机器或换环境跑，记得先 `pip install -r requirements.txt`，不要想当然认为环境已经装好。
- **不要仅凭代码写完就当作"完成"汇报给用户。** 目前状态就是"写完了但一次都没跑过"，跟"验证过能用"是两回事，交接、汇报进度时要如实说明这一点。


