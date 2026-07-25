"""
StageDouble - Interactive Machine Learning Trainer (Wekinator replacement)
StageDouble - 交互式机器学习训练器（用于替代 Wekinator）

[EN] What this file does:
A pure-Python stand-in for Wekinator, which crashes on macOS Sonoma due to a
Carbon menu incompatibility in its old Java Swing UI. This script follows
the same "interactive machine learning" (IML) workflow Wekinator uses: you
demonstrate examples (hold a vocal/movement state and label what output it
should produce), train a small neural network regressor on those examples,
then run it live to turn incoming audio features into output parameters.

[中] 这个文件做什么：
这是一个纯 Python 实现，用来替代 Wekinator —— 因为 Wekinator 那套老旧的
Java Swing 界面在 macOS Sonoma 上会因为 Carbon 菜单不兼容而崩溃。这个脚本
沿用了 Wekinator 同样的"交互式机器学习"（Interactive Machine Learning，
简称 IML）工作方式：你先"示范"一些例子（保持某种唱歌/动作状态，并告诉
程序这种状态应该对应什么输出），然后训练一个小型神经网络回归模型来学习
这些例子，最后实时运行这个模型，把源源不断输入的音频特征转换成输出参数。

[EN] How to run (terminal commands available once running):
    python ml_trainer.py
    python ml_trainer.py --in-port 6448 --out-port 12000

    record <out1> <out2> [seconds]   capture live features for `seconds`
                                      (default 3), each labeled with the
                                      target output (out1, out2)
    train                            fit the regressor on all recorded
                                      examples so far
    run                               start continuously predicting outputs
                                      from live input and sending them by OSC
    stop                              stop the run loop
    status                            show example count / training state
    clear                             discard all recorded examples
    help                              show this command list
    quit                              shut down

[中] 如何运行（程序启动后可在终端里输入以下命令）：
    python ml_trainer.py
    python ml_trainer.py --in-port 6448 --out-port 12000

    record <out1> <out2> [秒数]       保持某个状态，采集 `秒数`（默认 3）秒
                                      的实时特征，并把它们都标记为目标
                                      输出 (out1, out2)
    train                            用目前已录制的所有样本训练回归模型
    run                               开始实时预测：不断读取输入特征、
                                      预测输出，并通过 OSC 发送出去
    stop                              停止实时预测
    status                            查看已录制样本数量 / 是否已训练
    clear                             清空所有已录制的训练样本
    help                              显示这份命令列表
    quit                              退出程序

[EN] What this connects to:
Pipeline: audio_extractor.py --(OSC in)--> this script --(OSC out)--> your
digital human / synth / whatever consumes the 2 output parameters. This
script listens for the 16-value feature vector on the same default
address audio_extractor.py sends to (127.0.0.1:6448 /wek/inputs), and
sends its own 2-value prediction onward to 127.0.0.1:12000
/stagedouble/outputs by default.

[中] 这个文件如何与其他文件连接：
数据流向：audio_extractor.py --(OSC 输入)--> 本脚本 --(OSC 输出)-->
你的数字人 / 合成器 / 或任何消费这 2 个输出参数的下游程序。本脚本默认
监听的地址（127.0.0.1:6448 /wek/inputs）和 audio_extractor.py 默认发送
的地址完全一致；训练完成后，本脚本会把预测出的 2 个输出值发送到默认地址
127.0.0.1:12000 /stagedouble/outputs，供下游程序接收。
"""

import argparse
import threading
import time
import warnings

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from sklearn.exceptions import ConvergenceWarning
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

N_OUTPUTS = 2  # 输出参数的个数：本项目固定输出 2 个数值（例如驱动数字人的两个控制维度）。
SAMPLE_INTERVAL = 0.1  # seconds; matches audio_extractor.py's ~10Hz hop rate
# 采样间隔（秒）：0.1 秒，与 audio_extractor.py 大约 10Hz 的发送频率保持一致，
# 这样每次 audio_extractor.py 发来新特征时，我们基本都能及时读到最新值。


class FeatureReceiver:
    """Thread-safe holder for the most recent OSC feature vector.

    ----------------------------------------------------------------------
    [中] 一个线程安全的容器，用来保存"最新收到的一条 OSC 特征向量"。

    为什么需要它：OSC 服务器在后台线程里持续接收数据，而主线程（负责
    终端交互、录制训练样本、运行推理）需要随时读取"当前最新的特征值"。
    用一把锁（Lock）保护读写操作，避免两个线程同时读写导致数据错乱。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None  # 最近一次收到的特征向量（16 个浮点数的列表），尚未收到数据时为 None。

    def update(self, address, *args):
        """[中] OSC 收到新消息时的回调函数：把新特征向量存起来。"""
        with self._lock:
            self._latest = list(args)

    def latest(self):
        """[中] 读取当前最新的特征向量的一份拷贝（还没收到过数据则返回 None）。"""
        with self._lock:
            return list(self._latest) if self._latest is not None else None


def build_model():
    """[EN] Build a fresh, untrained scaler + neural network regression
    pipeline.
    [中] 构建一个全新的、尚未训练的"标准化 + 神经网络回归"流水线（pipeline）。
    """
    # One small hidden layer is enough to fit the handful of examples a
    # live demonstration session produces; StandardScaler matters a lot
    # here since pitch (~Hz), volume (~0-1), and MFCCs (~-400 to 400) are
    # on wildly different scales. lbfgs converges reliably on small,
    # non-streaming datasets like these, unlike the default 'adam' solver.
    #
    # 关键决策说明：
    # 1）为什么用神经网络（MLP）：这是一个通用、易用的非线性回归器，
    #    能够学习"特征组合 -> 输出参数"之间复杂的映射关系，而且 scikit-learn
    #    里开箱即用，不需要自己实现反向传播。
    # 2）为什么隐藏层只有 10 个神经元（一层）：现场示范采集到的训练样本
    #    通常只有几十到上百条，模型太大容易过拟合，一个小隐藏层就足够
    #    拟合这种小数据集。
    # 3）为什么要先做 StandardScaler（标准化）：16 个特征的数值范围差异
    #    极大——音高大约是几十到几千 Hz，音量大约在 0~1 之间，MFCC 大约
    #    在 -400~400 之间。如果不做标准化，数值范围大的特征会主导训练
    #    过程，掩盖其他特征的影响。
    # 4）为什么用 'lbfgs' 而不是默认的 'adam'：lbfgs 是一种拟牛顿法，
    #    在小规模、一次性（非流式）数据集上收敛更稳定可靠；'adam' 这类
    #    随机梯度方法通常需要更多数据和更多轮次才能收敛好。
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(10,), solver="lbfgs", max_iter=2000),
    )


class Trainer:
    """[EN] Owns the training examples, the model, and the live "run" loop
    that turns incoming features into outgoing OSC predictions.
    [中] 负责管理训练样本、模型本身，以及实时"运行"循环——不断把收到的
    输入特征转换成预测输出并通过 OSC 发送出去。
    """

    def __init__(self, osc_out_client, osc_out_address):
        self.examples_X = []  # 训练样本的输入部分：每个元素是一个 16 维特征向量。
        self.examples_y = []  # 训练样本的输出部分：每个元素是一个 [out1, out2] 目标值。
        self.model = build_model()  # 当前的（尚未训练的）模型。
        self.trained = False  # 模型是否已经训练过。
        self.osc_out_client = osc_out_client  # 用来发送预测结果的 OSC 客户端。
        self.osc_out_address = osc_out_address  # 发送预测结果时使用的 OSC 地址。
        self._run_thread = None  # 后台"运行"线程（还没启动时为 None）。
        self._run_stop_event = threading.Event()  # 用来通知后台线程"该停止了"的信号量。

    def add_example(self, features, target):
        """[中] 添加一条训练样本（一个特征向量 + 它对应的目标输出）。"""
        self.examples_X.append(features)
        self.examples_y.append(target)

    def train(self):
        """[中] 用目前收集到的所有训练样本，重新训练一个模型。
        至少需要 2 条样本才能训练（否则 scikit-learn 无法拟合）。
        """
        if len(self.examples_X) < 2:
            print("Need at least 2 training examples before training (see 'record').")
            return
        self.model = build_model()  # 每次训练都从头构建一个全新模型，避免受上一次训练结果影响。
        self.model.fit(np.array(self.examples_X), np.array(self.examples_y))
        self.trained = True
        print(f"Trained on {len(self.examples_X)} examples.")

    def predict(self, features):
        """[中] 用当前模型对一个特征向量做预测，返回 [out1, out2]。"""
        return self.model.predict([features])[0]

    def start_run(self, receiver):
        """[中] 启动后台"实时运行"线程：持续读取最新特征、预测、发送 OSC。"""
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
        特征、用模型预测、通过 OSC 发送预测结果，并在终端上打印出来。
        这个循环独立运行在后台线程里，这样主线程的命令提示符（input()）
        仍然可以正常接受用户输入（比如 'stop' 命令）。
        """
        while not self._run_stop_event.is_set():
            features = receiver.latest()
            if features is not None:
                out = self.predict(features)
                self.osc_out_client.send_message(self.osc_out_address, out.tolist())
                print(f"\routput: {out[0]:8.3f}, {out[1]:8.3f}   ", end="", flush=True)
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
    """[EN] Block for `seconds`, sampling the live feature stream at
    SAMPLE_INTERVAL and returning every feature vector captured — all of
    which the caller will label with the same `target` output.
    [中] 阻塞等待 `seconds` 秒，期间每隔 SAMPLE_INTERVAL 秒从实时特征流
    里采一次样，把采集到的所有特征向量收集起来并返回——调用者会把这些
    样本全部标记为同一个 `target` 目标输出（也就是"在这段时间内保持
    某个状态，这段时间采到的所有特征都算作这个状态的例子"）。
    """
    if receiver.latest() is None:
        print("No feature data received yet - is audio_extractor.py running?")
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
  record <out1> <out2> [seconds]   hold a state and capture it for `seconds`
                                    (default 3), labeled with target output
                                    (out1, out2)
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
    """[中] 解析命令行参数：输入特征的监听地址、输出预测结果的发送地址。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-ip", default="127.0.0.1", help="address to listen on for input features")
    parser.add_argument("--in-port", type=int, default=6448, help="must match audio_extractor.py's --port")
    parser.add_argument("--in-address", default="/wek/inputs", help="must match audio_extractor.py's --address")
    parser.add_argument("--out-ip", default="127.0.0.1", help="where to send predicted outputs")
    parser.add_argument("--out-port", type=int, default=12000, help="OSC port for predicted outputs")
    parser.add_argument("--out-address", default="/stagedouble/outputs", help="OSC address for predicted outputs")
    return parser.parse_args()


def main():
    """[EN] Entry point: start the OSC listener in the background, then run
    an interactive terminal command loop in the foreground.
    [中] 主入口函数：在后台启动 OSC 监听服务器，然后在前台运行一个
    交互式终端命令循环，等待用户输入命令。
    """
    args = parse_args()

    receiver = FeatureReceiver()
    dispatcher = Dispatcher()
    dispatcher.map(args.in_address, receiver.update)
    server = BlockingOSCUDPServer((args.in_ip, args.in_port), dispatcher)
    # 用一个后台守护线程（daemon thread）运行 OSC 服务器，这样它不会
    # 阻塞下面的终端命令循环，主程序退出时这个线程也会自动被清理掉。
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Listening for features on {args.in_ip}:{args.in_port}{args.in_address}")

    osc_out_client = SimpleUDPClient(args.out_ip, args.out_port)
    print(f"Sending outputs to {args.out_ip}:{args.out_port}{args.out_address}")

    trainer = Trainer(osc_out_client, args.out_address)
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
            # record <out1> <out2> [seconds] —— 录制训练样本。
            if len(rest) < N_OUTPUTS:
                print("Usage: record <out1> <out2> [seconds]")
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
            # status —— 查看当前状态（样本数、是否已训练）。
            print(f"{len(trainer.examples_X)} examples recorded. Trained: {trainer.trained}")

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
    server.shutdown()
    server.server_close()


if __name__ == "__main__":
    main()
