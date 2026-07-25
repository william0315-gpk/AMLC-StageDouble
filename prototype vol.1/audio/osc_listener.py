"""
OSC feature listener for testing StageDouble's audio pipeline.
OSC 特征监听器 —— 用于测试 StageDouble 音频流水线的调试工具。

[EN] What this file does:
Listens on the same host/port/address audio_extractor.py sends to, and
prints each incoming feature vector with real, human-readable numbers so
you can verify audio_extractor.py is working correctly without needing
ml_trainer.py (or Wekinator) running.

[中] 这个文件做什么：
监听和 audio_extractor.py 发送目标完全相同的地址，把每一条收到的特征
向量以易读的形式（带上每个数值对应的名字，比如 pitch_hz、volume 等）
打印出来。这样你不需要真的启动 ml_trainer.py（更不需要 Wekinator），
就能单独确认 audio_extractor.py 是否在正确工作、发出的数值是否合理。

[EN] How to run:
    python osc_listener.py
    python osc_listener.py --port 6448 --address /wek/inputs

[中] 如何运行：
    python osc_listener.py
    python osc_listener.py --port 6448 --address /wek/inputs

[EN] What this connects to:
This is a standalone diagnostic tool, not part of the main pipeline. Run
it instead of (not alongside) ml_trainer.py, since both would otherwise
try to listen on the same port 6448. It only receives — it does not send
anything onward.

[中] 这个文件如何与其他文件连接：
这是一个独立的调试工具，不属于主数据流水线的一部分。使用时应该用它
代替 ml_trainer.py 来运行（而不是两者同时运行），因为它们默认会争抢
同一个端口 6448。这个脚本只接收数据，不会把数据转发给任何下游程序。
"""

import argparse

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# Must match the feature order produced by audio_extractor.extract_features().
# 这里的顺序必须和 audio_extractor.py 里 extract_features() 函数生成特征向量
# 的顺序完全一致，这样打印出来的名字才能和数值一一对应。
FEATURE_NAMES = ["pitch_hz", "volume"] + [f"mfcc_{i + 1}" for i in range(13)] + ["tempo_bpm"]


def print_features(address, *args):
    """[中] OSC 收到一条特征消息时的回调函数：把 16 个数值和它们各自的
    名字（FEATURE_NAMES）配对，格式化打印成一行，方便肉眼查看。
    """
    parts = [f"{name}={value:7.2f}" for name, value in zip(FEATURE_NAMES, args)]
    print(" | ".join(parts))


def main():
    """[中] 主入口函数：解析命令行参数，启动 OSC 服务器并持续监听，
    直到用户按下 Ctrl+C 停止。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="127.0.0.1", help="address to listen on")
    parser.add_argument("--port", type=int, default=6448, help="OSC port to listen on (must match the sender)")
    parser.add_argument("--address", default="/wek/inputs", help="OSC address to match")
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.map(args.address, print_features)

    server = BlockingOSCUDPServer((args.ip, args.port), dispatcher)
    print(f"OSC listener on {args.ip}:{args.port}{args.address}. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
