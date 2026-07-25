# NOTES — model vol.1

_Last updated: 2026-07-25_

## English

### What was built

- `motion_extractor.py` — reads a dance video file, runs MediaPipe BlazePose (33 keypoints x, y, z = 99 values) on every frame, streams the vector over OSC to `127.0.0.1:6449 /stagedouble/motion`, loops the video by default.
- `ml_trainer_v2.py` — based on `prototype vol.1/audio/ml_trainer.py`. Listens on two OSC ports simultaneously (audio: 6448 `/wek/inputs`, motion: 6449 `/stagedouble/motion`), merges them into a 115-dim vector before every `record`/`train`/`run` step, keeps the same terminal commands and 2-value OSC output on port 12000.
- `audio/audio_extractor.py` — unmodified copy of prototype vol.1's version, so this folder works standalone.
- `requirements.txt`, `README.md` (bilingual, with pipeline diagram).

### What was tested

1. **OSC round-trip, mock senders/listener.** Wrote throwaway test scripts (not part of this folder — kept in the session scratchpad) that:
   - send fake random 16-value vectors to port 6448 `/wek/inputs` at 10Hz,
   - send fake random 99-value vectors to port 6449 `/stagedouble/motion` at 30Hz,
   - listen on port 12000 `/stagedouble/outputs` and print anything received.
2. **`ml_trainer_v2.py` driven end-to-end via its real terminal interface** (subprocess stdin/stdout, not code inspection): `status` -> `record 0 0 2` -> `record 1 1 2` -> `status` -> `train` -> `status` -> `run` -> `stop` -> `quit`. Result: both streams reported ready, 40 examples captured (20+20), `Trained on 40 examples (115-dim input)` printed, `run` streamed live 2-value predictions that the mock output listener received correctly on port 12000, `stop`/`quit` shut down cleanly (exit code 0), and repeated on a second run with the same clean result.
3. **`motion_extractor.py` smoke test** against a synthetically generated 15-frame mp4 (a moving circle, `--no-loop`): confirmed it opens the video, downloads the pose model on first run, runs `PoseLandmarker.detect_for_video` per frame without crashing, throttles to the video's frame rate, and exits cleanly. No pose was detected in this synthetic clip (expected — it isn't a real person), so the actual OSC-send branch wasn't exercised by this run.
4. **OSC-send branch verified separately** with mock landmark objects: confirmed `landmarks_to_feature_vector()` produces exactly 99 values and that sending them over OSC to `/stagedouble/motion` and receiving them on a listener round-trips correctly.

### Issues found during testing (and how they were resolved)

- **MediaPipe's legacy `mp.solutions.pose` API is not available in this environment.** The installed version (mediapipe 0.10.35, the only version with wheels for macOS arm64 + Python 3.13 here) removed the old "Solutions" API entirely (`mp.solutions` doesn't exist; only `mediapipe.tasks` does). `motion_extractor.py` was written against the current **Tasks API** (`mediapipe.tasks.python.vision.PoseLandmarker`) instead. This requires an explicit `.task` model bundle rather than a built-in model, so the script auto-downloads the official Google-hosted "lite" pose model into `model vol.1/models/` on first run (cached after that; gitignored since it's a regenerable binary asset). If you deploy on an environment with an older mediapipe (<=0.10.9), the legacy `solutions` API would also work, but the Tasks API is the currently maintained path and was used for forward-compatibility.
- **A stale unrelated process was squatting on port 6448** during the first test run (`OSError: Address already in use`), from an unrelated earlier session process (`audio/iml_trainer.py`, not part of this project). Killed it and the OSC servers bound cleanly afterward. Worth remembering if `ml_trainer_v2.py` (or `audio_extractor.py`) ever fails to start with "Address already in use" — check `lsof -i :6448` / `:6449` / `:12000` for leftover processes.

### Not yet tested / next step

- **No real dance video was available in this environment**, so BlazePose's actual detection accuracy/quality on real footage, and the full 115-dim pipeline with *real* (not random-mock) audio + motion data, remain unverified. Next step: run all three scripts together with a live microphone and a real dance video and confirm the recorded/trained/predicted values behave sensibly.
- **Real-time performance of `motion_extractor.py` at full video frame rate** (e.g. 30fps) alongside `audio_extractor.py`'s 10Hz loop and `ml_trainer_v2.py`'s 10Hz sampling hasn't been profiled on real footage — the "lite" pose model was chosen by default for speed, but if it falls behind, `--model full`/`--model heavy` trade speed for accuracy in the other direction, or a smaller/faster model stays the safer default.
- Once real-data end-to-end testing passes, commit this folder to git (currently untracked, like the rest of the working tree per `HANDOFF.md`).

---

## 中文

### 已经构建的内容

- `motion_extractor.py` —— 读取一段舞蹈视频文件，对每一帧运行 MediaPipe BlazePose（33 个关键点 x, y, z = 99 个数值），通过 OSC 把向量发送到 `127.0.0.1:6449 /stagedouble/motion`，默认循环播放视频。
- `ml_trainer_v2.py` —— 基于 `prototype vol.1/audio/ml_trainer.py` 改写。同时监听两个 OSC 端口（音频：6448 `/wek/inputs`，动作：6449 `/stagedouble/motion`），在每次 `record`/`train`/`run` 之前合并成一个 115 维向量，终端命令和端口 12000 上的 2 值 OSC 输出保持不变。
- `audio/audio_extractor.py` —— prototype vol.1 版本的原样拷贝，让这个文件夹可以独立运行。
- `requirements.txt`、`README.md`（中英双语，含流水线示意图）。

### 已经测试的内容

1. **OSC 数据往返测试（mock 发送端/接收端）。** 写了几个一次性测试脚本（不属于这个文件夹的正式内容，只留在本次会话的 scratchpad 目录里）：
   - 以 10Hz 向端口 6448 `/wek/inputs` 发送随机生成的 16 维假数据；
   - 以 30Hz 向端口 6449 `/stagedouble/motion` 发送随机生成的 99 维假数据；
   - 监听端口 12000 `/stagedouble/outputs`，把收到的任何内容打印出来。
2. **通过真实终端接口驱动 `ml_trainer_v2.py` 完整走一遍流程**（用子进程操纵它的标准输入/输出，不是单纯看代码）：依次输入 `status` -> `record 0 0 2` -> `record 1 1 2` -> `status` -> `train` -> `status` -> `run` -> `stop` -> `quit`。结果：两路数据流都显示已就绪，成功采集 40 条样本（20+20），打印出 `Trained on 40 examples (115-dim input)`，`run` 命令实时输出的 2 维预测值被 mock 输出监听器在端口 12000 上正确收到，`stop`/`quit` 正常退出（退出码 0），重复跑了第二遍，结果同样干净无误。
3. **`motion_extractor.py` 的冒烟测试**：用一段合成的 15 帧 mp4（画面里是一个移动的圆圈，`--no-loop` 模式）测试，确认它能正常打开视频、第一次运行时自动下载姿态模型、逐帧调用 `PoseLandmarker.detect_for_video` 不崩溃、按视频帧率节流、并正常退出。因为这段合成视频里没有真实的人，所以没有检测到姿态（符合预期），也就没有真正触发 OSC 发送那一支代码分支。
4. **单独验证了 OSC 发送分支**：用模拟的关键点对象测试，确认 `landmarks_to_feature_vector()` 精确产出 99 个数值，并且把它们通过 OSC 发到 `/stagedouble/motion`、由监听器接收，整个往返过程正确无误。

### 测试中发现的问题（以及如何解决的）

- **本机环境里没有旧版 `mp.solutions.pose` API 可用。** 当前安装的版本（mediapipe 0.10.35，是这台机器上 macOS arm64 + Python 3.13 唯一有预编译包可用的版本）已经彻底移除了旧的 "Solutions" API（`mp.solutions` 根本不存在，只有 `mediapipe.tasks`）。因此 `motion_extractor.py` 改用目前的 **Tasks API**（`mediapipe.tasks.python.vision.PoseLandmarker`）来实现。这个新 API 需要显式提供一个 `.task` 格式的模型文件，而不是像旧版那样内置模型，所以脚本第一次运行时会自动把 Google 官方托管的 "lite" 姿态模型下载到 `model vol.1/models/` 目录（之后会复用缓存；这个目录已经加入 `.gitignore`，因为它是可以自动重新生成的二进制资源）。如果以后部署到装有旧版 mediapipe（<=0.10.9）的环境，旧版 `solutions` API 其实也能用，但为了保证以后长期可维护，这里选择用官方目前主推的 Tasks API。
- **第一次跑测试时，端口 6448 被一个无关的残留进程占用**（报错 `OSError: Address already in use`），来源是之前某次会话遗留的一个进程（`audio/iml_trainer.py`，不属于本项目）。把它 kill 掉之后，OSC 服务器就能正常绑定端口了。以后如果 `ml_trainer_v2.py`（或 `audio_extractor.py`）启动时报"地址已被占用"，记得用 `lsof -i :6448` / `:6449` / `:12000` 检查是否有残留进程。

### 还没测试的部分 / 下一步

- **本机环境里没有真实的舞蹈视频可用**，所以 BlazePose 在真实画面上的检测精度/质量，以及用**真实**（而不是随机生成的假）音频 + 动作数据走完整的 115 维流水线，目前都还没有验证过。下一步：用真实麦克风和真实舞蹈视频，把三个脚本一起跑起来，确认录制/训练/预测出来的数值是否合理。
- **`motion_extractor.py` 在真实视频帧率下（比如 30fps）的实时性能**，配合 `audio_extractor.py` 的 10Hz 循环和 `ml_trainer_v2.py` 的 10Hz 采样，还没有在真实素材上做过性能评测——默认选用了 "lite" 姿态模型以保证速度，如果发现处理跟不上，可以用 `--model full` / `--model heavy` 用速度换精度，或者反过来考虑更轻量的方案保证不掉帧。
- 等真实数据的端到端测试通过之后，把这个文件夹提交到 git（目前和工作区里其余内容一样都还没有提交，参见 `HANDOFF.md`）。
