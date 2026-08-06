"""
StageDouble - 图形控制台 / Visual Control Panel

[EN] A single-window GUI that replaces the terminal-based ml_trainer_v2.py.
Drag sliders to set target outputs, click buttons to record / train / run,
and watch the 2D character preview react in real time.

[中] 单窗口图形界面，替代终端版 ml_trainer_v2.py。拖滑块设参数，点按钮
录制/训练/运行，右侧 2D 小人实时跟着动。

How to run / 如何运行：
    python app.py

Still need audio_extractor.py and motion_extractor.py running in separate
terminals beforehand.
仍然需要先在另外两个终端启动 audio_extractor.py 和 motion_extractor.py。
"""

import argparse
import threading
import time
import math
import tkinter as tk
from tkinter import ttk

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from ml_trainer_v2 import (
    MergedFeatureReceiver,
    Trainer,
    build_model,
    start_osc_server,
    record_examples,
    N_OUTPUTS,
    N_INPUT_FEATURES,
    SAMPLE_INTERVAL,
)

# 6 个参数的标签和颜色
PARAMS = [
    ("嘴巴开合", "#FF6B6B"),
    ("表情强度", "#FFD93D"),
    ("头部角度", "#6BCB77"),
    ("身体幅度", "#4D96FF"),
    ("手臂高度", "#9D4EDD"),
    ("动作速度", "#FF9F45"),
]

BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
WIDGET_BG = "#0f3460"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#e94560"


class CharacterCanvas:
    """[中] 用 tkinter Canvas 画一个简笔画小人，根据 6 个参数实时变形。"""

    def __init__(self, canvas, width=320, height=380):
        self.canvas = canvas
        self.w = width
        self.h = height
        self.values = [0.0] * 6
        self.phase = 0.0  # 用于动作速度的摆动相位

        # 小人的基准位置
        self.cx = width // 2
        self.cy = height // 2 + 20

    def update_values(self, values):
        """[中] 更新 6 个参数值，下一帧绘制时生效。"""
        self.values = [max(0.0, min(1.0, v)) for v in values]

    def draw(self):
        """[中] 清空画布并重新绘制小人。"""
        self.canvas.delete("all")

        mouth, expr, head, body, arm, speed = self.values

        # 动作速度驱动摆动相位
        self.phase += 0.05 + speed * 0.3
        swing = math.sin(self.phase) * (0.3 + speed * 0.7)

        # 身体幅度影响整体偏移
        body_offset = swing * 15 * (0.3 + body * 0.7)

        # --- 头部 ---
        head_size = 35
        head_tilt = (head - 0.5) * 0.6  # -0.3 ~ 0.3 弧度
        head_cx = self.cx + body_offset * 0.3
        head_cy = self.cy - 70 + swing * 5

        # 画头（圆）
        self._draw_circle(head_cx, head_cy, head_size, "#FFD93D", outline="#e94560")

        # --- 眼睛 ---
        eye_offset_x = 10
        eye_y = head_cy - 5
        eye_size = 3 + expr * 4
        self._draw_circle(head_cx - eye_offset_x, eye_y, eye_size, "#1a1a2e")
        self._draw_circle(head_cx + eye_offset_x, eye_y, eye_size, "#1a1a2e")

        # 表情强度影响眉毛
        if expr > 0.3:
            brow_y = eye_y - 8 - expr * 3
            brow_angle = expr * 0.3
            self.canvas.create_line(
                head_cx - eye_offset_x - 5, brow_y + brow_angle * 5,
                head_cx - eye_offset_x + 5, brow_y - brow_angle * 5,
                fill="#e94560", width=2,
            )
            self.canvas.create_line(
                head_cx + eye_offset_x - 5, brow_y - brow_angle * 5,
                head_cx + eye_offset_x + 5, brow_y + brow_angle * 5,
                fill="#e94560", width=2,
            )

        # --- 嘴巴 ---
        mouth_w = 16
        mouth_h = 2 + mouth * 14  # 嘴巴开合决定椭圆高度
        mouth_y = head_cy + 12
        self.canvas.create_oval(
            head_cx - mouth_w // 2, mouth_y - mouth_h // 2,
            head_cx + mouth_w // 2, mouth_y + mouth_h // 2,
            fill="#e94560", outline="#c41e3a",
        )

        # --- 身体 ---
        neck_y = head_cy + head_size
        hip_y = self.cy + 30 + body_offset * 0.2
        body_sway = body_offset * 0.5
        self.canvas.create_line(
            head_cx, neck_y,
            self.cx + body_sway, hip_y,
            fill="#4D96FF", width=4,
        )

        # --- 手臂 ---
        shoulder_y = neck_y + 8
        arm_length = 45
        arm_angle_base = -0.3  # 手臂基础角度（略向下）
        arm_raise = arm * 1.4  # 手臂高度决定抬起角度

        # 左臂
        left_angle = arm_angle_base - arm_raise + swing * 0.2
        left_hand_x = self.cx + body_sway - math.sin(left_angle) * arm_length
        left_hand_y = shoulder_y + math.cos(left_angle) * arm_length
        self.canvas.create_line(
            self.cx + body_sway - 5, shoulder_y,
            left_hand_x, left_hand_y,
            fill="#9D4EDD", width=3,
        )

        # 右臂
        right_angle = arm_angle_base + arm_raise - swing * 0.2
        right_hand_x = self.cx + body_sway + math.sin(right_angle) * arm_length
        right_hand_y = shoulder_y + math.cos(right_angle) * arm_length
        self.canvas.create_line(
            self.cx + body_sway + 5, shoulder_y,
            right_hand_x, right_hand_y,
            fill="#9D4EDD", width=3,
        )

        # --- 腿 ---
        leg_length = 50
        leg_swing = swing * 0.4
        # 左腿
        left_foot_x = self.cx + body_sway - 10 + math.sin(leg_swing) * 15
        left_foot_y = hip_y + leg_length
        self.canvas.create_line(
            self.cx + body_sway - 5, hip_y,
            left_foot_x, left_foot_y,
            fill="#4D96FF", width=3,
        )
        # 右腿
        right_foot_x = self.cx + body_sway + 10 - math.sin(leg_swing) * 15
        right_foot_y = hip_y + leg_length
        self.canvas.create_line(
            self.cx + body_sway + 5, hip_y,
            right_foot_x, right_foot_y,
            fill="#4D96FF", width=3,
        )

        # 底部阴影
        shadow_w = 30 + abs(swing) * 10
        self.canvas.create_oval(
            self.cx - shadow_w, left_foot_y + 5,
            self.cx + shadow_w, left_foot_y + 12,
            fill="#2d2d44", outline="",
        )

    def _draw_circle(self, cx, cy, r, fill, outline="#333"):
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=outline)


class StageDoubleApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.latest_output = [0.0] * 6
        self.output_lock = threading.Lock()
        self.recording = False

        root.title("StageDouble 控制台")
        root.configure(bg=BG_COLOR)
        root.resizable(False, False)

        # --- OSC 基础设施 ---
        self.receiver = MergedFeatureReceiver()
        self.osc_out_client = SimpleUDPClient(args.out_ip, args.out_port)
        self.trainer = Trainer(self.osc_out_client, args.out_address, args.model)

        # 启动两路 OSC 监听
        start_osc_server(args.in_ip, args.audio_port, args.audio_address, self.receiver.audio.update)
        start_osc_server(args.in_ip, args.motion_port, args.motion_address, self.receiver.motion.update)

        # 输出监听（接收自己发出的预测结果，用于柱状图和小人）
        out_dispatcher = Dispatcher()
        out_dispatcher.map(args.out_address, self._on_output_received)
        self.out_server = BlockingOSCUDPServer(("127.0.0.1", args.out_port + 1), out_dispatcher)
        # 注意：ml_trainer_v2 发到 out_port，我们监听 out_port+1 来接收回显
        # 实际上我们直接在 _run_loop 里更新 latest_output，不需要额外监听

        # --- 界面布局 ---
        self._build_ui()

        # --- 定时刷新 ---
        self._update_ui()

    def _build_ui(self):
        # 顶部标题
        title = tk.Label(
            self.root, text="StageDouble 控制台",
            font=("Microsoft YaHei", 16, "bold"),
            fg="#ffffff", bg=BG_COLOR,
        )
        title.pack(pady=(10, 5))

        # 主体：左右两栏
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(fill="x", padx=15, pady=5)

        # === 左栏：录制参数 ===
        left = tk.LabelFrame(main_frame, text="录制参数", font=("Microsoft YaHei", 11),
                             fg=TEXT_COLOR, bg=PANEL_COLOR, bd=1, relief="solid")
        left.pack(side="left", fill="y", padx=(0, 8))

        self.sliders = []
        for i, (label, color) in enumerate(PARAMS):
            tk.Label(left, text=label, font=("Microsoft YaHei", 10),
                     fg=TEXT_COLOR, bg=PANEL_COLOR).grid(row=i, column=0, padx=8, pady=6, sticky="w")

            sv = tk.DoubleVar(value=0.3)
            slider = tk.Scale(left, from_=0.0, to=1.0, resolution=0.05,
                              orient="horizontal", variable=sv,
                              length=150, bg=PANEL_COLOR, fg=TEXT_COLOR,
                              highlightthickness=0, troughcolor=WIDGET_BG,
                              activebackground=ACCENT_COLOR)
            slider.grid(row=i, column=1, padx=5, pady=6)
            self.sliders.append(sv)

        # 录制秒数
        tk.Label(left, text="录制秒数", font=("Microsoft YaHei", 10),
                 fg=TEXT_COLOR, bg=PANEL_COLOR).grid(row=6, column=0, padx=8, pady=6, sticky="w")
        self.seconds_var = tk.DoubleVar(value=3.0)
        tk.Spinbox(left, from_=1, to=30, increment=1, textvariable=self.seconds_var,
                   width=5, font=("Consolas", 11), bg=WIDGET_BG, fg=TEXT_COLOR).grid(
            row=6, column=1, padx=5, pady=6, sticky="w")

        # 按钮区
        btn_frame = tk.Frame(left, bg=PANEL_COLOR)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=10)

        self.btn_record = tk.Button(btn_frame, text="录制", command=self._on_record,
                                    font=("Microsoft YaHei", 11), width=6,
                                    bg=ACCENT_COLOR, fg="white", relief="flat")
        self.btn_record.grid(row=0, column=0, padx=3)

        self.btn_train = tk.Button(btn_frame, text="训练", command=self._on_train,
                                   font=("Microsoft YaHei", 11), width=6,
                                   bg=WIDGET_BG, fg="white", relief="flat")
        self.btn_train.grid(row=0, column=1, padx=3)

        self.btn_run = tk.Button(btn_frame, text="运行", command=self._on_run,
                                 font=("Microsoft YaHei", 11), width=6,
                                 bg="#6BCB77", fg="white", relief="flat")
        self.btn_run.grid(row=1, column=0, padx=3, pady=5)

        self.btn_stop = tk.Button(btn_frame, text="停止", command=self._on_stop,
                                  font=("Microsoft YaHei", 11), width=6,
                                  bg="#FF6B6B", fg="white", relief="flat")
        self.btn_stop.grid(row=1, column=1, padx=3, pady=5)

        self.btn_clear = tk.Button(btn_frame, text="清空", command=self._on_clear,
                                   font=("Microsoft YaHei", 11), width=6,
                                   bg="#555555", fg="white", relief="flat")
        self.btn_clear.grid(row=2, column=0, columnspan=2, padx=3)

        # 模型选择
        tk.Label(left, text="模型", font=("Microsoft YaHei", 10),
                 fg=TEXT_COLOR, bg=PANEL_COLOR).grid(row=8, column=0, padx=8, pady=4, sticky="w")
        self.model_var = tk.StringVar(value=self.args.model)
        model_menu = ttk.Combobox(left, textvariable=self.model_var, width=12,
                                  values=["random_forest", "gradient_boost", "mlp"],
                                  state="readonly")
        model_menu.grid(row=8, column=1, padx=5, pady=4, sticky="w")
        model_menu.bind("<<ComboboxSelected>>", self._on_model_change)

        # === 右栏：输出可视化 ===
        right = tk.Frame(main_frame, bg=BG_COLOR)
        right.pack(side="left", fill="both", expand=True)

        # 柱状图
        bar_frame = tk.LabelFrame(right, text="实时输出", font=("Microsoft YaHei", 11),
                                  fg=TEXT_COLOR, bg=PANEL_COLOR, bd=1, relief="solid")
        bar_frame.pack(fill="x", pady=(0, 8))

        self.bars = []
        self.bar_values = []
        for i, (label, color) in enumerate(PARAMS):
            tk.Label(bar_frame, text=label, font=("Microsoft YaHei", 9),
                     fg=TEXT_COLOR, bg=PANEL_COLOR, width=6).grid(row=i, column=0, padx=4, pady=2, sticky="e")

            bar_bg = tk.Frame(bar_frame, bg="#2d2d44", width=200, height=18)
            bar_bg.grid(row=i, column=1, padx=4, pady=2)
            bar_bg.pack_propagate(False)
            bar = tk.Frame(bar_bg, bg=color, width=0, height=18)
            bar.place(x=0, y=0)
            self.bars.append(bar)

            val_lbl = tk.Label(bar_frame, text="0.000", font=("Consolas", 9),
                               fg="#ffffff", bg=PANEL_COLOR, width=6)
            val_lbl.grid(row=i, column=2, padx=4, pady=2, sticky="w")
            self.bar_values.append(val_lbl)

        # 2D 小人 Canvas
        char_frame = tk.LabelFrame(right, text="2D 预览", font=("Microsoft YaHei", 11),
                                   fg=TEXT_COLOR, bg=PANEL_COLOR, bd=1, relief="solid")
        char_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(char_frame, width=320, height=380, bg="#0f0f1e", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.character = CharacterCanvas(self.canvas)

        # === 底部状态栏 ===
        self.status_frame = tk.Frame(self.root, bg=PANEL_COLOR)
        self.status_frame.pack(fill="x", padx=15, pady=(5, 10))

        self.lbl_audio = tk.Label(self.status_frame, text="○ 音频", font=("Microsoft YaHei", 10),
                                  fg="#666666", bg=PANEL_COLOR)
        self.lbl_audio.pack(side="left", padx=10)

        self.lbl_motion = tk.Label(self.status_frame, text="○ 动作", font=("Microsoft YaHei", 10),
                                   fg="#666666", bg=PANEL_COLOR)
        self.lbl_motion.pack(side="left", padx=10)

        self.lbl_samples = tk.Label(self.status_frame, text="样本: 0", font=("Microsoft YaHei", 10),
                                    fg=TEXT_COLOR, bg=PANEL_COLOR)
        self.lbl_samples.pack(side="left", padx=10)

        self.lbl_trained = tk.Label(self.status_frame, text="未训练", font=("Microsoft YaHei", 10),
                                    fg="#666666", bg=PANEL_COLOR)
        self.lbl_trained.pack(side="left", padx=10)

        self.lbl_msg = tk.Label(self.status_frame, text="", font=("Microsoft YaHei", 9),
                                fg=ACCENT_COLOR, bg=PANEL_COLOR)
        self.lbl_msg.pack(side="right", padx=10)

    def _on_output_received(self, address, *args):
        """[中] 收到 OSC 输出时更新内部状态。"""
        with self.output_lock:
            for i in range(min(6, len(args))):
                self.latest_output[i] = float(args[i])

    def _update_ui(self):
        """[中] 每 50ms 刷新界面：状态栏、柱状图、2D 小人。"""
        # 状态栏
        status = self.receiver.status()
        self.lbl_audio.config(
            text="● 音频" if status["audio_ready"] else "○ 音频",
            fg="#6BCB77" if status["audio_ready"] else "#666666")
        self.lbl_motion.config(
            text="● 动作" if status["motion_ready"] else "○ 动作",
            fg="#6BCB77" if status["motion_ready"] else "#666666")
        self.lbl_samples.config(text=f"样本: {len(self.trainer.examples_X)}")
        self.lbl_trained.config(
            text="已训练" if self.trainer.trained else "未训练",
            fg="#6BCB77" if self.trainer.trained else "#666666")

        # 柱状图
        with self.output_lock:
            values = list(self.latest_output)
        for i, val in enumerate(values):
            val = max(0.0, min(1.0, val))
            self.bars[i].config(width=int(val * 200))
            self.bar_values[i].config(text=f"{val:.3f}")

        # 2D 小人
        self.character.update_values(values)
        self.character.draw()

        self.root.after(50, self._update_ui)

    def _set_msg(self, msg, duration=3.0):
        """[中] 在底部显示一条临时消息。"""
        self.lbl_msg.config(text=msg)
        if duration > 0:
            self.root.after(int(duration * 1000), lambda: self.lbl_msg.config(text=""))

    def _on_record(self):
        """[中] 录制按钮：用当前滑块值作为目标输出，采集指定秒数的样本。"""
        if self.recording:
            return
        target = [s.get() for s in self.sliders]
        seconds = self.seconds_var.get()

        status = self.receiver.status()
        if not status["audio_ready"] or not status["motion_ready"]:
            missing = []
            if not status["audio_ready"]:
                missing.append("音频")
            if not status["motion_ready"]:
                missing.append("动作")
            self._set_msg(f"缺少: {' + '.join(missing)}，请先启动采集程序")
            return

        self.recording = True
        self.btn_record.config(state="disabled", text="录制中...")

        def do_record():
            examples = record_examples(self.receiver, target, seconds)
            for features in examples:
                self.trainer.add_example(features, target)
            self.recording = False
            self.root.after(0, lambda: self.btn_record.config(state="normal", text="录制"))
            self.root.after(0, lambda: self._set_msg(f"采集了 {len(examples)} 个样本"))

        threading.Thread(target=do_record, daemon=True).start()

    def _on_train(self):
        """[中] 训练按钮。"""
        if len(self.trainer.examples_X) < 2:
            self._set_msg("至少需要 2 个样本才能训练")
            return

        self.btn_train.config(state="disabled", text="训练中...")

        def do_train():
            self.trainer.train()
            self.root.after(0, lambda: self.btn_train.config(state="normal", text="训练"))
            self.root.after(0, lambda: self._set_msg(f"训练完成（{len(self.trainer.examples_X)} 个样本）"))

        threading.Thread(target=do_train, daemon=True).start()

    def _on_run(self):
        """[中] 运行按钮：启动实时预测，同时启动输出回显线程。"""
        if not self.trainer.trained:
            self._set_msg("请先训练")
            return

        # 自定义 run loop：在发送 OSC 的同时更新本地 latest_output
        def custom_run_loop():
            while not self.trainer._run_stop_event.is_set():
                features = self.receiver.latest()
                if features is not None:
                    out = self.trainer.predict(features)
                    self.osc_out_client.send_message(self.args.out_address, out.tolist())
                    with self.output_lock:
                        for i in range(6):
                            self.latest_output[i] = float(out[i])
                time.sleep(SAMPLE_INTERVAL)

        if self.trainer._run_thread and self.trainer._run_thread.is_alive():
            self._set_msg("已经在运行了")
            return

        self.trainer._run_stop_event.clear()
        self.trainer._run_thread = threading.Thread(target=custom_run_loop, daemon=True)
        self.trainer._run_thread.start()
        self._set_msg("实时预测已启动")

    def _on_stop(self):
        """[中] 停止按钮。"""
        self.trainer.stop_run()
        self._set_msg("已停止")

    def _on_clear(self):
        """[中] 清空按钮。"""
        self.trainer.examples_X.clear()
        self.trainer.examples_y.clear()
        self.trainer.trained = False
        self._set_msg("已清空所有样本")

    def _on_model_change(self, event):
        """[中] 切换模型类型。"""
        new_model = self.model_var.get()
        self.trainer.model_type = new_model
        self.trainer.model = build_model(new_model)
        self.trainer.trained = False
        self._set_msg(f"已切换到 {new_model}，需重新训练")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-ip", default="127.0.0.1")
    parser.add_argument("--audio-port", type=int, default=6448)
    parser.add_argument("--audio-address", default="/wek/inputs")
    parser.add_argument("--motion-port", type=int, default=6449)
    parser.add_argument("--motion-address", default="/stagedouble/motion")
    parser.add_argument("--out-ip", default="127.0.0.1")
    parser.add_argument("--out-port", type=int, default=12000)
    parser.add_argument("--out-address", default="/stagedouble/outputs")
    parser.add_argument("--model", default="random_forest",
                        choices=["random_forest", "gradient_boost", "mlp"])
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    app = StageDoubleApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
