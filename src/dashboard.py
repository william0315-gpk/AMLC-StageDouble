"""
StageDouble - Visual Dashboard / 可视化面板

[EN] A real-time visual dashboard that listens for the 6-value OSC output
from ml_trainer_v2.py and displays them as animated bars. Much easier to
understand than reading numbers in a terminal.

[中] 实时可视化面板：监听 ml_trainer_v2.py 发出的 6 个输出值，用动态
柱状图显示。比看终端里的数字直观得多。

How to run / 如何运行：
    python dashboard.py
    python dashboard.py --port 12000
"""

import argparse
import threading
import tkinter as tk
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# 6 个输出参数的中文标签和颜色
PARAMS = [
    ("嘴巴开合", "#FF6B6B"),
    ("表情强度", "#FFD93D"),
    ("头部角度", "#6BCB77"),
    ("身体幅度", "#4D96FF"),
    ("手臂高度", "#9D4EDD"),
    ("动作速度", "#FF9F45"),
]


class Dashboard:
    def __init__(self, root, osc_ip, osc_port, osc_address):
        self.root = root
        self.latest_values = [0.0] * 6
        self.lock = threading.Lock()
        self.connected = False

        root.title("StageDouble - 实时输出面板")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)

        # 标题
        title = tk.Label(
            root,
            text="StageDouble 实时输出",
            font=("Microsoft YaHei", 18, "bold"),
            fg="#ffffff",
            bg="#1a1a2e",
        )
        title.pack(pady=(15, 5))

        # 状态指示
        self.status_label = tk.Label(
            root,
            text="等待数据...",
            font=("Microsoft YaHei", 10),
            fg="#888888",
            bg="#1a1a2e",
        )
        self.status_label.pack(pady=(0, 10))

        # 柱状图区域
        self.bars = []
        self.bar_labels = []
        self.bar_values = []

        chart_frame = tk.Frame(root, bg="#1a1a2e")
        chart_frame.pack(padx=30, pady=10)

        for i, (label, color) in enumerate(PARAMS):
            # 左侧标签
            lbl = tk.Label(
                chart_frame,
                text=label,
                font=("Microsoft YaHei", 11),
                fg="#cccccc",
                bg="#1a1a2e",
                width=8,
                anchor="e",
            )
            lbl.grid(row=i, column=0, padx=(0, 8), pady=3, sticky="e")

            # 柱状图背景
            bar_bg = tk.Frame(chart_frame, bg="#2d2d44", width=300, height=24)
            bar_bg.grid(row=i, column=1, pady=3)
            bar_bg.pack_propagate(False)

            # 柱状图前景（实际数值）
            bar = tk.Frame(bar_bg, bg=color, width=0, height=24)
            bar.place(x=0, y=0)
            self.bars.append(bar)

            # 数值文字
            val_lbl = tk.Label(
                chart_frame,
                text="0.000",
                font=("Consolas", 11),
                fg="#ffffff",
                bg="#1a1a2e",
                width=7,
                anchor="w",
            )
            val_lbl.grid(row=i, column=2, padx=(8, 0), pady=3, sticky="w")
            self.bar_values.append(val_lbl)

        # 底部提示
        hint = tk.Label(
            root,
            text="先启动 audio_extractor.py 和 motion_extractor.py\n再在 ml_trainer_v2.py 中输入 record / train / run",
            font=("Microsoft YaHei", 9),
            fg="#666666",
            bg="#1a1a2e",
            justify="center",
        )
        hint.pack(pady=(5, 15))

        # 启动 OSC 接收线程
        dispatcher = Dispatcher()
        dispatcher.map(osc_address, self.on_output)
        self.server = BlockingOSCUDPServer((osc_ip, osc_port), dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        # 定时刷新界面
        self.update_ui()

    def on_output(self, address, *args):
        """收到 OSC 数据时更新数值。"""
        with self.lock:
            for i in range(min(6, len(args))):
                self.latest_values[i] = float(args[i])
            self.connected = True

    def update_ui(self):
        """每 50ms 刷新一次柱状图。"""
        with self.lock:
            values = list(self.latest_values)
            connected = self.connected

        if connected:
            self.status_label.config(text="● 已连接，正在接收数据", fg="#6BCB77")
        else:
            self.status_label.config(text="等待数据...（确认 ml_trainer_v2.py 已运行 run 命令）", fg="#888888")

        max_bar_width = 300
        for i, val in enumerate(values):
            val = max(0.0, min(1.0, val))
            self.bars[i].config(width=int(val * max_bar_width))
            self.bar_values[i].config(text=f"{val:.3f}")

        self.root.after(50, self.update_ui)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="127.0.0.1", help="listen address")
    parser.add_argument("--port", type=int, default=12000, help="OSC port to listen on")
    parser.add_argument("--address", default="/stagedouble/outputs", help="OSC address to listen for")
    args = parser.parse_args()

    root = tk.Tk()
    Dashboard(root, args.ip, args.port, args.address)
    root.mainloop()


if __name__ == "__main__":
    main()
