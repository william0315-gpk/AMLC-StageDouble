"""
StageDouble - 一键启动器

[EN] Single entry point: pick a dance video, click "Start", and all three
programs launch automatically. No need to open multiple terminals manually.

[中] 单一入口：选一个舞蹈视频，点"启动"，三个程序自动全部跑起来。
不用再手动开多个终端。

How to run / 如何运行：
    python start.py
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")


class Launcher:
    def __init__(self, root):
        self.root = root
        self.video_path = None
        self.processes = []

        root.title("StageDouble 启动器")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)

        # 标题
        tk.Label(
            root, text="StageDouble 启动器",
            font=("Microsoft YaHei", 16, "bold"),
            fg="#ffffff", bg="#1a1a2e",
        ).pack(pady=(20, 15))

        # 视频选择区
        video_frame = tk.Frame(root, bg="#1a1a2e")
        video_frame.pack(padx=30, pady=5, fill="x")

        tk.Label(
            video_frame, text="舞蹈视频：",
            font=("Microsoft YaHei", 11),
            fg="#cccccc", bg="#1a1a2e",
        ).pack(side="left", padx=(0, 8))

        self.video_label = tk.Label(
            video_frame, text="（未选择）",
            font=("Microsoft YaHei", 10),
            fg="#888888", bg="#1a1a2e",
            width=25, anchor="w",
        )
        self.video_label.pack(side="left", padx=(0, 8))

        tk.Button(
            video_frame, text="选择...",
            command=self._pick_video,
            font=("Microsoft YaHei", 10),
            bg="#0f3460", fg="white", relief="flat",
            width=6,
        ).pack(side="left")

        # 启动按钮
        self.btn_start = tk.Button(
            root, text="启动",
            command=self._start,
            font=("Microsoft YaHei", 14, "bold"),
            bg="#e94560", fg="white", relief="flat",
            width=12, height=1,
            cursor="hand2",
        )
        self.btn_start.pack(pady=20)

        # 状态提示
        self.status_label = tk.Label(
            root, text='选择视频后点"启动"',
            font=("Microsoft YaHei", 9),
            fg="#666666", bg="#1a1a2e",
        )
        self.status_label.pack(pady=(0, 5))

        # 说明
        tk.Label(
            root,
            text="启动后会打开三个窗口：\n"
                 "1. 声音采集（终端）\n"
                 "2. 动作采集（终端）\n"
                 "3. 图形控制台（窗口）\n"
                 "在图形控制台里录制、训练、运行即可",
            font=("Microsoft YaHei", 9),
            fg="#555555", bg="#1a1a2e",
            justify="center",
        ).pack(pady=(5, 20))

        # 窗口关闭时清理子进程
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="选择舞蹈视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
            initialdir=os.path.expanduser("~"),
        )
        if path:
            self.video_path = path
            self.video_label.config(text=os.path.basename(path), fg="#6BCB77")
            self.status_label.config(text='已选择视频，点"启动"开始')

    def _start(self):
        if not self.video_path:
            self.status_label.config(text="请先选择舞蹈视频！", fg="#FF6B6B")
            return

        python = sys.executable
        audio_script = os.path.join(SRC_DIR, "audio_extractor.py")
        motion_script = os.path.join(SRC_DIR, "motion_extractor.py")
        app_script = os.path.join(SRC_DIR, "app.py")

        self.status_label.config(text="正在启动...", fg="#FFD93D")
        self.btn_start.config(state="disabled", text="启动中...")

        # Windows 下用 CREATE_NEW_CONSOLE 让每个程序在独立终端运行
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE

        try:
            # 1. 启动声音采集
            p_audio = subprocess.Popen(
                [python, audio_script],
                creationflags=creationflags,
            )
            self.processes.append(p_audio)

            # 2. 启动动作采集
            p_motion = subprocess.Popen(
                [python, motion_script, "--video", self.video_path],
                creationflags=creationflags,
            )
            self.processes.append(p_motion)

            # 3. 启动图形界面（等 1 秒让采集程序先起来）
            self.root.after(1000, lambda: self._launch_app(python, app_script))

        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self.btn_start.config(state="normal", text="启动")

    def _launch_app(self, python, app_script):
        try:
            p_app = subprocess.Popen(
                [python, app_script],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            )
            self.processes.append(p_app)
            self.status_label.config(text="已启动！在图形控制台里操作", fg="#6BCB77")
            self.btn_start.config(state="normal", text="已启动")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self.btn_start.config(state="normal", text="启动")

    def _on_close(self):
        # 窗口关闭时不杀子进程，让它们继续运行
        self.root.destroy()


def main():
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
