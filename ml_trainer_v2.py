"""
StageDouble - Interactive Machine Learning Trainer v2 (audio + motion)
StageDouble - 交互式机器学习训练器 v2（音频 + 动作双输入版）

[EN] What this file does:
This is the vol.2 evolution of prototype vol.1's ml_trainer.py. Where the
original only listened to one 16-value audio feature stream, this version
listens to TWO simultaneous OSC input streams:
  - audio features (16 values) from audio_extractor.py on port 6448
    /wek/inputs
  - motion features (99 values) from motion_extractor.py on port 6449
    /stagedouble/motion
and merges them into a single 115-value input vector (16 + 99 = 115) before
every record / train / run step. Everything else works exactly the same as
ml_trainer.py: the same "interactive machine learning" (IML) workflow of
demonstrating labeled examples, training a small neural network regressor,
and running it live — same terminal commands, same OSC output (now 6 values:
mouth openness, expression intensity, head angle, body movement amplitude,
arm height, movement speed).

[中] 这个文件做什么：
这是 prototype vol.1 里 ml_trainer.py 的 vol.2 升级版本。原来的版本只
监听一路 16 维的音频特征流，这个版本会**同时**监听两路 OSC 输入流：
  - 音频特征（16 个数值），来自 audio_extractor.py，监听端口 6448
    地址 /wek/inputs
  - 动作特征（99 个数值），来自 motion_extractor.py，监听端口 6449
    地址 /stagedouble/motion
在每一次 record（录制）/ train（训练）/ run（运行推理）之前，会把这
两路数据合并成一个 115 维的输入向量（16 + 99 = 115）。除此之外，其余
部分沿用了 ml_trainer.py 的设计思路：同样的"交互式机器学习"（IML）工作流程
--示范带标签的样本、训练一个回归模型、实时运行它--同样
的终端命令、同样输出 6 个数值。

[EN] How to run (terminal commands available once running):
    python ml_trainer_v2.py
    python ml_trainer_v2.py --audio-port 6448 --motion-port 6449 --out-port 12000

    record <out1> <out2> <out3> <out4> <out5> <out6> [seconds]   capture live merged features for
                                      `seconds` (default 3), each labeled
                                      with the target outputs (mouth,
                                      expression, head, body, arm, speed)
    train                            fit the regressor on all recorded
                                      examples so far
    run                               start continuously predicting outputs
                                      from live input and sending them by OSC
    stop                               stop the run loop
    status                            show example count / training state
    clear                             discard all recorded examples
    help                              show this command list
    quit                              shut down

[中] 如何运行（程序启动后可在终端里输入以下命令）：
    python ml_trainer_v2.py
    python ml_trainer_v2.py --audio-port 6448 --motion-port 6449 --out-port 12000

    record <out1> <out2> <out3> <out4> <out5> <out6> [秒数]       保持某个状态，采集 `秒数`（默认 3）秒
                                      的实时合并特征，并把它们都标记为
                                      目标输出（嘴巴、表情、头部、身体、
                                      手臂、速度）
    train                            用目前已录制的所有样本训练回归模型
    run                              开始实时预测：不断读取输入特征、
                                      预测输出，并通过 OSC 发送出去
    stop                             停止实时预测
    status                           查看已录制样本数量 / 是否已训练
    clear                            清空所有已录制的训练样本
    help                             显示这份命令列表
    quit                             退出程序

[EN] What this connects to:
Pipeline: audio_extractor.py --(OSC, 16 values, port 6448)--> \\
          motion_extractor.py --(OSC, 99 values, port 6449)--> \\
          this script --(merge to 115 values -> predict 6 values)--> \\
          OSC out (port 12000 /stagedouble/outputs) --> your digital human /
          synth / whatever consumes the 6 output parameters.
Both audio_extractor.py and motion_extractor.py must be running and sending
to this script's default listen addresses (127.0.0.1:6448 /wek/inputs and
127.0.0.1:6449 /stagedouble/motion respectively) before `record` will
capture meaningful data.

[中] 这个文件如何与其他文件连接：
数据流向：audio_extractor.py --(OSC，16 个数值，端口 6448)--> \\
          motion_extractor.py --(OSC，99 个数值，端口 6449)--> \\
          本脚本 --(合并成 115 维 -> 预测出 6 个数值)--> \\
          OSC 输出（端口 12000 /stagedouble/outputs）--> 你的数字人 /
          合成器 / 或任何消费这 6 个输出参数的下游程序。
必须先让 audio_extractor.py 和 motion_extractor.py 都运行起来，并且
分别发送到本脚本默认监听的地址（127.0.0.1:6448 /wek/inputs 和
127.0.0.1:6449 /stagedouble/motion），`record` 命令才能采集到有意义
的数据。
"""

import argparse
import threading
import time
import warnings

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Small IML datasets rarely hit lbfgs's convergence criterion within
# max_iter; that's fine for our purposes and not worth alarming the user.
#
# 交互式机器学习的训练集通常很小，lbfgs 求解器在 max_iter 次迭代内往往
# 达不到严格的收敛标准——但这对我们的场景没有实际影响，所以直接屏蔽这个
# 警告，避免吓到用户。
warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_OUTPUTS = 6  # 输出参数的个数：嘴巴开合、表情强度、头部角度、身体动作幅度、手臂高度、动作速度。

N_AUDIO_FEATURES = 16  # audio_extractor.py 发送的特征维度：pitch, volume, 13 x MFCC, tempo。
N_MOTION_FEATURES = 99  # motion_extractor.py 发送的特征维度：33 个关键点 x (x, y, z)。
N_INPUT_FEATURES = N_AUDIO_FEATURES + N_MOTION_FEATURES  # 合并后的总输入维度：16 + 99 = 115。

SAMPLE_INTERVAL = 0.1  # seconds; matches audio_extractor.py's ~10Hz hop rate
# 采样间隔（秒）：0.1 秒，与 audio_extractor.py 大约 10Hz 的发送频率保持一致。
# motion_extractor.py 通常按视频帧率发送（比如 30fps），比这个间隔更快，
# 所以每次我们采样时，两路数据基本都能读到各自的最新值。


class StreamReceiver:
    """[EN] Thread-safe holder for the most recent value of ONE OSC stream
    (either the audio stream or the motion stream).
    [中] 一个线程安全的容器，用来保存"某一路 OSC 数据流"最新收到的一条
    消息（可以是音频流，也可以是动作流；每一路各自用一个实例）。
    """

    def __init__(self, expected_length):
        self._lock = threading.Lock()
        self._latest = None  # 最近一次收到的特征向量，尚未收到数据时为 None。
        self._expected_length = expected_length  # 这一路数据预期的向量长度，用来过滤畸形消息。

    def update(self, address, *args):
        """[中] OSC 收到新消息时的回调函数：把新特征向量存起来。
        如果收到的数值个数不对（比如上游脚本版本不一致），直接丢弃这条
        消息并打印警告，避免污染后续的合并向量。
        """
        if len(args) != self._expected_length:
            print(
                f"Warning: got {len(args)} values on {address}, expected {self._expected_length}. Ignoring."
            )
            return
        with self._lock:
            self._latest = list(args)

    def latest(self):
        """[中] 读取当前最新的特征向量的一份拷贝（还没收到过数据则返回 None）。"""
        with self._lock:
            return list(self._latest) if self._latest is not None else None


class MergedFeatureReceiver:
    """[EN] Combines the audio StreamReceiver and the motion StreamReceiver
    into a single 115-value feature vector.
    [中] 把音频那一路 StreamReceiver 和动作那一路 StreamReceiver 合并起来，
    对外提供一个统一的 115 维特征向量接口。
    """

    def __init__(self):
        self.audio = StreamReceiver(N_AUDIO_FEATURES)
        self.motion = StreamReceiver(N_MOTION_FEATURES)

    def latest(self):
        """[中] 返回当前合并后的 115 维特征向量：[16 个音频值, 99 个动作值]。
        只要有任何一路还没收到过数据，就返回 None（因为合并向量还不完整）。
        """
        audio_features = self.audio.latest()
        motion_features = self.motion.latest()
        if audio_features is None or motion_features is None:
            return None
        return audio_features + motion_features

    def status(self):
        """[中] 返回两路数据流各自是否已经收到过数据，用于诊断排查。"""
        return {
            "audio_ready": self.audio.latest() is not None,
            "motion_ready": self.motion.latest() is not None,
        }


def build_model(model_type="random_forest"):
    """[EN] Build a fresh, untrained regression model.
    [中] 构建一个全新的、尚未训练的回归模型。

    Three model types are supported (select via --model CLI argument):
    - random_forest: 100-tree forest, stable on small IML datasets (default)
    - gradient_boost: 100-stage boosting, wrapped in MultiOutputRegressor
    - mlp: deeper neural network (64→32 hidden units, StandardScaler + lbfgs)

    支持三种模型类型（通过 --model 命令行参数选择）：
    - random_forest: 100 棵树的随机森林，小数据集上最稳定（默认）
    - gradient_boost: 100 轮梯度提升，用 MultiOutputRegressor 包装
    - mlp: 更深的神经网络（64→32 隐藏层，StandardScaler + lbfgs 求解器）
    """
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "gradient_boost":
        return MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        )
    elif model_type == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), solver="lbfgs", max_iter=2000),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Choose random_forest, gradient_boost, or mlp.")


class Trainer:
    """[EN] Owns the training examples, the model, and the live "run" loop
    that turns incoming merged features into outgoing OSC predictions.
    [中] 负责管理训练样本、模型本身，以及实时"运行"循环——不断把收到的
    合并输入特征转换成预测输出并通过 OSC 发送出去。
    """

    def __init__(self, osc_out_client, osc_out_address, model_type="random_forest"):
        self.examples_X = []  # 训练样本的输入部分：每个元素是一个 115 维特征向量。
        self.examples_y = []  # 训练样本的输出部分：每个元素是一个包含 6 个目标值的列表。
        self.model_type = model_type  # 当前使用的模型类型。
        self.model = build_model(model_type)  # 当前的（尚未训练的）模型。
        self.trained = False  # 模型是否已经训练过。
        self.osc_out_client = osc_out_client  # 用来发送预测结果的 OSC 客户端。
        self.osc_out_address = osc_out_address  # 发送预测结果时使用的 OSC 地址。
        self._run_thread = None  # 后台"运行"线程（还没启动时为 None）。
        self._run_stop_event = threading.Event()  # 用来通知后台线程"该停止了"的信号量。

    def add_example(self, features, target):
        """[中] 添加一条训练样本（一个 115 维特征向量 + 它对应的目标输出）。"""
        self.examples_X.append(features)
        self.examples_y.append(target)

    def train(self):
        """[中] 用目前收集到的所有训练样本，重新训练一个模型。
        至少需要 2 条样本才能训练（否则 scikit-learn 无法拟合）。
        """
        if len(self.examples_X) < 2:
            print("Need at least 2 training examples before training (see 'record').")
            return
        self.model = build_model(self.model_type)  # 每次训练都从头构建一个全新模型，避免受上一次训练结果影响。
        self.model.fit(np.array(self.examples_X), np.array(self.examples_y))
        self.trained = True
        print(f"Trained on {len(self.examples_X)} examples ({N_INPUT_FEATURES}-dim input).")

    def predict(self, features):
        """[中] 用当前模型对一个 115 维特征向量做预测，返回 6 个输出值。
        输出值截断到 [0, 1] 范围内，防止下游渲染端收到异常值。
        """
        out = self.model.predict([features])[0]
        return np.clip(out, 0.0, 1.0)

    def start_run(self, receiver):
        """[中] 启动后台"实时运行"线程：持续读取最新合并特征、预测、发送 OSC。"""
        if not self.trained:
            print("Train the model first with 'train'.")
            return
        if self._run_thread and self._run_thread.is_alive():
            print("Already running.")
            return
        self._run_stop_event.clear()
        self._run_thread = threading.Thread(target=self._run_loop, args=(receiver,), daemon=True)
        self._run_thread.start()
        print("Running. Outputs below; type 'stop' + Enter to stop.")

    def _run_loop(self, receiver):
        """[中] 后台运行线程的主循环：每隔 SAMPLE_INTERVAL 秒读取一次最新
        合并特征、用模型预测、通过 OSC 发送预测结果，并在终端上打印出来。
        这个循环独立运行在后台线程里，这样主线程的命令提示符（input()）
        仍然可以正常接受用户输入（比如 'stop' 命令）。
        """
        while not self._run_stop_event.is_set():
            features = receiver.latest()
            if features is not None:
                out = self.predict(features)
                self.osc_out_client.send_message(self.osc_out_address, out.tolist())
                # 打印 6 个输出值：mouth, expression, head, body, arm, speed
                vals = " ".join(f"{v:7.3f}" for v in out)
                print(f"\routput: {vals}   ", end="", flush=True)
            time.sleep(SAMPLE_INTERVAL)
        print()  # move off the \r-overwritten line before the next prompt
        # 换行，避免下一次打印的命令提示符和上面用 \r 覆盖打印的那一行粘在一起。

    def stop_run(self):
        """[中] 停止后台"实时运行"线程，并等待它真正退出。"""
        if not self._run_thread or not self._run_thread.is_alive():
            print("Not running.")
            return
        self._run_stop_event.set()
        self._run_thread.join()
        print("Stopped.")


def record_examples(receiver, target, seconds):
    """[EN] Block for `seconds`, sampling the live MERGED feature stream at
    SAMPLE_INTERVAL and returning every 115-value feature vector captured —
    all of which the caller will label with the same `target` output.
    [中] 阻塞等待 `seconds` 秒，期间每隔 SAMPLE_INTERVAL 秒从实时的
    "合并特征流"里采一次样，把采集到的所有 115 维特征向量收集起来并
    返回——调用者会把这些样本全部标记为同一个 `target` 目标输出。
    """
    status = receiver.status()
    if not status["audio_ready"]:
        print("No audio feature data received yet - is audio_extractor.py running?")
    if not status["motion_ready"]:
        print("No motion feature data received yet - is motion_extractor.py running?")
    print(f"Recording target {target} for {seconds:.1f}s...")
    end_time = time.time() + seconds
    examples = []
    while time.time() < end_time:
        features = receiver.latest()
        if features is not None:
            examples.append(features)
        time.sleep(SAMPLE_INTERVAL)
    print(f"Captured {len(examples)} examples.")
    return examples


HELP_TEXT = """\
Commands:
  record <out1>..<out6> [seconds]  hold a state and capture it for `seconds`
                                    (default 3), labeled with target outputs
                                    (mouth, expression, head, body, arm, speed)
  train                            fit the regressor on examples recorded so far
  run                              start live prediction + OSC output
  stop                             stop live prediction
  status                           show example count / training state
  clear                            discard all recorded examples
  help                             show this message
  quit                             shut down"""
# 上面这份 HELP_TEXT 是英文命令帮助文本，会在程序启动时以及输入 help/? 时打印。
# 具体每个命令的中文说明见文件顶部的模块级文档字符串（docstring）。


def print_help():
    """[中] 打印命令帮助信息。"""
    print(HELP_TEXT)


def parse_args():
    """[中] 解析命令行参数：两路输入的监听地址、输出预测结果的发送地址。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-ip", default="127.0.0.1", help="address to listen on for both input streams")

    parser.add_argument("--audio-port", type=int, default=6448, help="must match audio_extractor.py's --port")
    parser.add_argument("--audio-address", default="/wek/inputs", help="must match audio_extractor.py's --address")

    parser.add_argument("--motion-port", type=int, default=6449, help="must match motion_extractor.py's --port")
    parser.add_argument(
        "--motion-address", default="/stagedouble/motion", help="must match motion_extractor.py's --address"
    )

    parser.add_argument("--out-ip", default="127.0.0.1", help="where to send predicted outputs")
    parser.add_argument("--out-port", type=int, default=12000, help="OSC port for predicted outputs")
    parser.add_argument("--out-address", default="/stagedouble/outputs", help="OSC address for predicted outputs")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "gradient_boost", "mlp"],
        help="regression model type (default: random_forest)",
    )
    return parser.parse_args()


def start_osc_server(ip, port, address, callback):
    """[EN] Start one OSC server on its own daemon thread, listening for one
    address on one port. We need two of these (audio + motion) since they
    arrive on two different ports simultaneously.
    [中] 在独立的后台守护线程里启动一个 OSC 服务器，监听某个端口上的某个
    地址。因为音频和动作数据是同时从两个不同端口发来的，所以本脚本需要
    启动两个这样的服务器（各自监听各自的端口）。
    """
    dispatcher = Dispatcher()
    dispatcher.map(address, callback)
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    """[EN] Entry point: start both OSC listeners (audio + motion) in the
    background, then run an interactive terminal command loop in the
    foreground.
    [中] 主入口函数：在后台同时启动两个 OSC 监听服务器（音频 + 动作），
    然后在前台运行一个交互式终端命令循环，等待用户输入命令。
    """
    args = parse_args()

    receiver = MergedFeatureReceiver()

    audio_server = start_osc_server(args.in_ip, args.audio_port, args.audio_address, receiver.audio.update)
    print(f"Listening for audio features ({N_AUDIO_FEATURES}-dim) on {args.in_ip}:{args.audio_port}{args.audio_address}")

    motion_server = start_osc_server(args.in_ip, args.motion_port, args.motion_address, receiver.motion.update)
    print(f"Listening for motion features ({N_MOTION_FEATURES}-dim) on {args.in_ip}:{args.motion_port}{args.motion_address}")

    print(f"Merged input vector size: {N_INPUT_FEATURES} (= {N_AUDIO_FEATURES} audio + {N_MOTION_FEATURES} motion)")

    osc_out_client = SimpleUDPClient(args.out_ip, args.out_port)
    print(f"Sending outputs to {args.out_ip}:{args.out_port}{args.out_address}")
    print(f"Model: {args.model}")

    trainer = Trainer(osc_out_client, args.out_address, args.model)
    print_help()

    # 主命令循环：不断读取用户在终端输入的一行命令并执行对应操作。
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        cmd, *rest = line.split()
        cmd = cmd.lower()

        if cmd in ("quit", "exit"):
            # 退出前先确保后台"运行"线程已经停止。
            trainer.stop_run()
            break

        elif cmd == "record":
            # record <mouth>..<speed> [seconds] -- 录制训练样本（音频+动作合并）。
            if len(rest) < N_OUTPUTS:
                print("Usage: record <mouth> <expression> <head> <body> <arm> <speed> [seconds]")
                continue
            try:
                target = [float(rest[i]) for i in range(N_OUTPUTS)]
                seconds = float(rest[N_OUTPUTS]) if len(rest) > N_OUTPUTS else 3.0
            except ValueError:
                print("Outputs and seconds must be numbers.")
                continue
            for features in record_examples(receiver, target, seconds):
                trainer.add_example(features, target)

        elif cmd == "train":
            # train —— 用已录制的样本训练模型。
            trainer.train()

        elif cmd == "run":
            # run —— 开始实时预测并输出。
            trainer.start_run(receiver)

        elif cmd == "stop":
            # stop —— 停止实时预测。
            trainer.stop_run()

        elif cmd == "status":
            # status —— 查看当前状态（样本数、是否已训练、两路输入是否就绪）。
            stream_status = receiver.status()
            print(
                f"{len(trainer.examples_X)} examples recorded. Trained: {trainer.trained}. "
                f"Audio stream ready: {stream_status['audio_ready']}. "
                f"Motion stream ready: {stream_status['motion_ready']}."
            )

        elif cmd == "clear":
            # clear —— 清空所有已录制的训练样本，并重置训练状态。
            trainer.examples_X.clear()
            trainer.examples_y.clear()
            trainer.trained = False
            print("Cleared training data.")

        elif cmd in ("help", "?"):
            # help / ? —— 显示命令帮助。
            print_help()

        else:
            # 未知命令。
            print(f"Unknown command: {cmd!r}. Type 'help' for a list of commands.")

    print("Shutting down.")
    audio_server.shutdown()
    audio_server.server_close()
    motion_server.shutdown()
    motion_server.server_close()


if __name__ == "__main__":
    main()
