"""
StageDouble - OSC Output Listener / 输出监听器

[EN] A simple debugging tool that listens for the 6-value OSC output from
ml_trainer_v2.py and prints it in real time. Use this to verify your
trained model is producing sensible outputs before connecting to Unreal
Engine.

[中] 一个简单的调试工具：监听 ml_trainer_v2.py 发出的 6 个输出值并实时
打印。在对接 Unreal Engine 之前，先用这个工具验证模型输出是否合理。

How to run / 如何运行：
    python osc_listener.py
    python osc_listener.py --port 12000 --address /stagedouble/outputs
"""

import argparse
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

OUTPUT_LABELS = [
    "mouth   ",
    "express ",
    "head    ",
    "body    ",
    "arm     ",
    "speed   ",
]


def on_output(address, *args):
    """[中] 收到 6 个输出值时打印出来。"""
    parts = []
    for label, val in zip(OUTPUT_LABELS, args):
        parts.append(f"{label}: {val:7.3f}")
    print("\r" + "  ".join(parts), end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="127.0.0.1", help="listen address")
    parser.add_argument("--port", type=int, default=12000, help="OSC port to listen on")
    parser.add_argument("--address", default="/stagedouble/outputs", help="OSC address to listen for")
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.map(args.address, on_output)

    server = BlockingOSCUDPServer((args.ip, args.port), dispatcher)

    print(f"Listening on {args.ip}:{args.port}{args.address}")
    print("Waiting for predictions... (Ctrl+C to stop)\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
