# model vol.1

## English

**StageDouble vol.2**: adds a second real-time input stream — dance motion — alongside the vocal features from `prototype vol.1`. Where prototype vol.1 mapped voice alone to output parameters, `model vol.1` maps **voice + body movement together** to output parameters, using the same terminal-based interactive machine learning (IML) workflow.

### What's different from `prototype vol.1`

| | prototype vol.1 | model vol.1 |
|---|---|---|
| Inputs | audio only (16 values) | audio (16) **+** motion (99) = **115 values** |
| Motion source | — | dance video file, via MediaPipe BlazePose (33 keypoints x, y, z) |
| Trainer | `ml_trainer.py` | `ml_trainer_v2.py` (listens on two OSC ports at once, merges both streams before every record/train/run step) |
| Output | 2 values -> `12000 /stagedouble/outputs` | unchanged: 2 values -> `12000 /stagedouble/outputs` |
| Terminal commands | record / train / run / stop / status / clear / quit | unchanged |

`audio_extractor.py` itself is unchanged — it's copied as-is into `model vol.1/audio/` so this folder is self-contained and doesn't depend on `prototype vol.1/` being present.

### What's in this folder

- `motion_extractor.py` — reads a dance video file, runs MediaPipe BlazePose on every frame to extract 33 body keypoints (x, y, z = 99 values), and streams them over OSC in real time, looping the video by default.
- `ml_trainer_v2.py` — listens for the audio stream (16 values, port 6448) *and* the motion stream (99 values, port 6449) simultaneously, merges them into one 115-dimensional input vector, and runs the same record / train / run IML workflow as `ml_trainer.py`, sending 2 predicted output values onward.
- `audio/audio_extractor.py` — unmodified copy of `prototype vol.1/audio/audio_extractor.py` (captures mic audio, extracts pitch/volume/MFCC/tempo, sends 16 values over OSC).
- `requirements.txt` — Python dependencies for this folder.
- `NOTES.md` — build/test log and next steps.

### Pipeline diagram

```
 microphone                          dance video file
     |                                      |
     v                                      v
 audio_extractor.py                  motion_extractor.py
 (audio/audio_extractor.py)          (MediaPipe BlazePose)
     |                                      |
     | 16 values                            | 99 values
     | OSC 127.0.0.1:6448                   | OSC 127.0.0.1:6449
     | /wek/inputs                          | /stagedouble/motion
     |                                      |
     +------------------+-------------------+
                         |
                         v
              ml_trainer_v2.py
       merges both streams into one
        115-dim vector (16 + 99)
       record -> train -> run (MLP)
                         |
                         | 2 values
                         | OSC 127.0.0.1:12000
                         | /stagedouble/outputs
                         v
          digital human / synth / downstream consumer
```

### Setup

```bash
pip install -r requirements.txt
```

### Running all three scripts together

Run each in its own terminal, in this order:

```bash
# terminal 1: capture mic audio and extract features (16 values -> port 6448)
python "model vol.1/audio/audio_extractor.py"

# terminal 2: read a dance video and extract pose keypoints (99 values -> port 6449)
python "model vol.1/motion_extractor.py" --video path/to/dance.mp4

# terminal 3: merge both streams (115 values), train, and run predictions
python "model vol.1/ml_trainer_v2.py"
```

In `ml_trainer_v2.py`'s terminal prompt:

```
record <out1> <out2> [seconds]   hold a state (sing + move) and capture it
                                  for `seconds` (default 3), labeled with
                                  target output (out1, out2)
train                            fit the regressor on examples recorded so far
run                              start live prediction + OSC output
stop                             stop live prediction
status                           show example count / training state /
                                  whether each input stream has data yet
clear                            discard all recorded examples
help                             show this message
quit                             shut down
```



Typical session: sing and dance a first state, run `record 0 0 3`; sing and dance a different combined state, run `record 1 1 3`; repeat with a few more states and target values; then `train`; then `run` to stream live predictions to whatever consumes them (default `127.0.0.1:12000/stagedouble/outputs`).

`status` will tell you if either input stream hasn't sent any data yet (e.g. if you forgot to start `audio_extractor.py` or `motion_extractor.py`) — `record` needs both streams active to capture meaningful 115-dim examples.

macOS will prompt for microphone access on first run of `audio_extractor.py` — approve it under System Settings -> Privacy & Security -> Microphone.

Every script in this folder also has a bilingual (English/Chinese) header docstring explaining what it does, how to run it, and what it connects to.

---

## 中文

**StageDouble vol.2**：在 `prototype vol.1` 已有的人声特征基础上，新增第二路实时输入——舞蹈动作。prototype vol.1 只用"声音"来映射输出参数，`model vol.1` 则用"声音 + 肢体动作"两者共同映射输出参数，沿用同样的终端交互式机器学习（IML）工作方式。

### 和 `prototype vol.1` 的区别

| | prototype vol.1 | model vol.1 |
|---|---|---|
| 输入 | 仅音频（16 维） | 音频（16 维）**+** 动作（99 维）= **115 维** |
| 动作来源 | 无 | 舞蹈视频文件，通过 MediaPipe BlazePose 提取（33 个关键点 x, y, z） |
| 训练器 | `ml_trainer.py` | `ml_trainer_v2.py`（同时监听两个 OSC 端口，在每次 record/train/run 之前合并两路数据） |
| 输出 | 2 个数值 -> `12000 /stagedouble/outputs` | 不变：2 个数值 -> `12000 /stagedouble/outputs` |
| 终端命令 | record / train / run / stop / status / clear / quit | 不变 |

`audio_extractor.py` 本身没有任何改动——直接原样复制进了 `model vol.1/audio/`，这样这个文件夹是自包含的，不依赖 `prototype vol.1/` 目录是否存在。

### 这个文件夹里有什么

- `motion_extractor.py` —— 读取一段舞蹈视频文件，对每一帧运行 MediaPipe BlazePose，提取 33 个人体关键点（x, y, z = 99 个数值），通过 OSC 实时发送出去，默认循环播放视频。
- `ml_trainer_v2.py` —— 同时监听音频数据流（16 个值，端口 6448）和动作数据流（99 个值，端口 6449），合并成一个 115 维的输入向量，用和 `ml_trainer.py` 一样的 record / train / run 交互式机器学习流程，把预测出的 2 个输出值发送出去。
- `audio/audio_extractor.py` —— `prototype vol.1/audio/audio_extractor.py` 的原样拷贝（采集麦克风音频，提取音高/音量/MFCC/节奏，通过 OSC 发送 16 个数值）。
- `requirements.txt` —— 本文件夹的 Python 依赖。
- `NOTES.md` —— 构建/测试记录以及下一步计划。

### 流水线示意图

```
   麦克风                                舞蹈视频文件
     |                                      |
     v                                      v
 audio_extractor.py                  motion_extractor.py
 (audio/audio_extractor.py)          (MediaPipe BlazePose)
     |                                      |
     | 16 个数值                             | 99 个数值
     | OSC 127.0.0.1:6448                   | OSC 127.0.0.1:6449
     | /wek/inputs                          | /stagedouble/motion
     |                                      |
     +------------------+-------------------+
                         |
                         v
              ml_trainer_v2.py
         把两路数据合并成一个
        115 维向量（16 + 99）
       record -> train -> run（MLP）
                         |
                         | 2 个数值
                         | OSC 127.0.0.1:12000
                         | /stagedouble/outputs
                         v
             数字人 / 合成器 / 下游消费者
```

### 安装

```bash
pip install -r requirements.txt
```

### 如何一起运行这三个脚本

在三个不同的终端里分别运行，按下面的顺序：

```bash
# 终端 1：采集麦克风音频，提取特征（16 个数值 -> 端口 6448）
python "model vol.1/audio/audio_extractor.py"

# 终端 2：读取舞蹈视频，提取姿态关键点（99 个数值 -> 端口 6449）
python "model vol.1/motion_extractor.py" --video path/to/dance.mp4

# 终端 3：合并两路数据（115 个数值），训练并运行预测
python "model vol.1/ml_trainer_v2.py"
```

在 `ml_trainer_v2.py` 的终端提示符里：

```
record <out1> <out2> [秒数]       保持某个"唱歌+跳舞"的组合状态，采集
                                  `秒数`（默认 3）秒，标记为目标输出
                                  (out1, out2)
train                            用目前已录制的所有样本训练回归模型
run                              开始实时预测并通过 OSC 发送
stop                             停止实时预测
status                           查看样本数量 / 是否已训练 / 两路输入
                                  是否已经收到过数据
clear                            清空所有已录制的训练样本
help                             显示这份命令列表
quit                             退出程序
```

典型使用流程：唱一段歌并做一个动作，运行 `record 0 0 3`；换一种"唱+跳"的组合状态，运行 `record 1 1 3`；重复几次不同的状态和目标值；然后 `train`；然后 `run`，把实时预测结果发送给下游消费者（默认 `127.0.0.1:12000/stagedouble/outputs`）。

`status` 命令会告诉你是否有某一路输入还没收到过数据（比如忘了启动 `audio_extractor.py` 或 `motion_extractor.py`）——`record` 需要两路数据流都在正常工作，才能采集到有意义的 115 维样本。

macOS 在第一次运行 `audio_extractor.py` 时会弹出请求麦克风权限的提示，需要在"系统设置 -> 隐私与安全性 -> 麦克风"里允许访问。

这个文件夹里的每个脚本文件顶部也都有中英双语的模块文档字符串，解释这个文件做什么、怎么运行、跟谁连接。
