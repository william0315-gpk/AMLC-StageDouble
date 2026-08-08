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
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#cccccc"
ACCENT_COLOR = "#e94560"


def show_error(title, message):
    """[中] 在主线程中显示错误弹窗。"""
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        print(f"[ERROR] {title}: {message}")


class Launcher:
    def __init__(self, root):
        self.root = root
        self.video_path = None
        self.processes = []
        self.start_errors = []
        self.mode = tk.StringVar(value="classic")  # "classic" 或 "style"

        root.title("StageDouble 启动器")
        root.configure(bg=BG_COLOR)
        root.resizable(False, False)

        # 标题
        tk.Label(
            root, text="StageDouble 启动器",
            font=("Microsoft YaHei", 16, "bold"),
            fg="#ffffff", bg=BG_COLOR,
        ).pack(pady=(20, 10))

        # 模式选择
        mode_frame = tk.Frame(root, bg=BG_COLOR)
        mode_frame.pack(padx=30, pady=5, fill="x")
        tk.Label(mode_frame, text="启动模式：", font=("Microsoft YaHei", 11),
                 fg=TEXT_COLOR, bg=BG_COLOR).pack(side="left", padx=(0, 8))
        tk.Radiobutton(mode_frame, text="经典 IML", variable=self.mode,
                       value="classic", command=self._on_mode_change,
                       font=("Microsoft YaHei", 10), fg=TEXT_COLOR, bg=BG_COLOR,
                       selectcolor="#0f3460").pack(side="left", padx=3)
        tk.Radiobutton(mode_frame, text="风格库", variable=self.mode,
                       value="style", command=self._on_mode_change,
                       font=("Microsoft YaHei", 10), fg=TEXT_COLOR, bg=BG_COLOR,
                       selectcolor="#0f3460").pack(side="left", padx=3)

        # 视频选择区
        video_frame = tk.Frame(root, bg=BG_COLOR)
        video_frame.pack(padx=30, pady=5, fill="x")

        tk.Label(
            video_frame, text="舞蹈视频：",
            font=("Microsoft YaHei", 11),
            fg=TEXT_COLOR, bg=BG_COLOR,
        ).pack(side="left", padx=(0, 8))

        self.video_label = tk.Label(
            video_frame, text="（未选择）",
            font=("Microsoft YaHei", 10),
            fg="#888888", bg=BG_COLOR,
            width=25, anchor="w",
        )
        self.video_label.pack(side="left", padx=(0, 8))

        self.btn_pick_video = tk.Button(
            video_frame, text="选择...",
            command=self._pick_video,
            font=("Microsoft YaHei", 10),
            bg="#0f3460", fg="white", relief="flat",
            width=6,
        )
        self.btn_pick_video.pack(side="left")

        # 启动按钮
        self.btn_start = tk.Button(
            root, text="启动",
            command=self._start,
            font=("Microsoft YaHei", 14, "bold"),
            bg=ACCENT_COLOR, fg="white", relief="flat",
            width=12, height=1,
            cursor="hand2",
        )
        self.btn_start.pack(pady=20)

        # 状态提示
        self.status_label = tk.Label(
            root, text='选择视频后点"启动"',
            font=("Microsoft YaHei", 9),
            fg="#666666", bg=BG_COLOR,
        )
        self.status_label.pack(pady=(0, 5))

        # 说明
        self.help_label = tk.Label(
            root,
            text="启动后会打开三个窗口：\n"
                 "1. 声音采集（终端）\n"
                 "2. 动作采集（终端）\n"
                 "3. 图形控制台（窗口）\n"
                 "在图形控制台里录制、训练、运行即可",
            font=("Microsoft YaHei", 9),
            fg="#555555", bg=BG_COLOR,
            justify="center",
        )
        self.help_label.pack(pady=(5, 20))

        # 窗口关闭时清理子进程
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 初始化模式相关文案
        self._on_mode_change()

    def _on_mode_change(self):
        """[中] 根据所选模式更新界面提示。"""
        if self.mode.get() == "classic":
            self.btn_pick_video.config(state="normal")
            self.video_label.config(text="（未选择）" if self.video_path is None else os.path.basename(self.video_path))
            self.status_label.config(text='选择视频后点"启动"', fg="#666666")
            self.help_label.config(
                text="启动后会打开三个窗口：\n"
                     "1. 声音采集（终端）\n"
                     "2. 动作采集（终端）\n"
                     "3. 图形控制台（窗口）\n"
                     "在图形控制台里录制、训练、运行即可"
            )
        else:
            self.btn_pick_video.config(state="disabled")
            self.video_label.config(text="（风格库模式不需要实时视频）", fg="#888888")
            self.status_label.config(text='点"启动"开始（只需声音采集+控制台）', fg="#666666")
            self.help_label.config(
                text="风格库模式只启动两个窗口：\n"
                     "1. 声音采集（终端）\n"
                     "2. 图形控制台（窗口）\n"
                     "在控制台里构建/加载风格库后点运行即可"
            )

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="选择舞蹈视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
            initialdir=os.path.expanduser("~"),
        )
        if path:
            self.video_path = path
            self.video_label.config(text=os.path.basename(path), fg="#6BCB77")
            self.status_label.config(text='已选择视频，点"启动"开始', fg="#666666")

    def _start(self):
        if self.mode.get() == "classic" and not self.video_path:
            self.status_label.config(text="请先选择舞蹈视频！", fg="#FF6B6B")
            return

        python = sys.executable
        audio_script = os.path.join(SRC_DIR, "audio_extractor.py")
        motion_script = os.path.join(SRC_DIR, "motion_extractor.py")
        app_script = os.path.join(SRC_DIR, "app.py")

        self.status_label.config(text="正在启动...", fg="#FFD93D")
        self.btn_start.config(state="disabled", text="启动中...")
        self.start_errors = []

        # Windows 下用 CREATE_NEW_CONSOLE 让每个程序在独立终端运行
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE

        try:
            # 1. 启动声音采集（两种模式都需要）
            p_audio = subprocess.Popen(
                [python, audio_script],
                creationflags=creationflags,
            )
            self.processes.append(p_audio)

            if self.mode.get() == "classic":
                # 2. 启动动作采集（仅经典模式需要）
                p_motion = subprocess.Popen(
                    [python, motion_script, "--video", self.video_path],
                    creationflags=creationflags,
                )
                self.processes.append(p_motion)

                # 启动后台线程，检查子进程是否过早退出
                self._monitor_processes(p_audio, p_motion)
            else:
                # 风格库模式不需要动作采集，只检查声音采集
                self._monitor_processes(p_audio)

            # 3. 启动图形界面（等 1.5 秒让采集程序先起来）
            self.root.after(1500, lambda: self._launch_app(python, app_script))

        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self.btn_start.config(state="normal", text="启动")

    def _monitor_processes(self, *procs):
        """[中] 后台检查子进程是否在启动后 3 秒内异常退出。"""
        def check():
            time.sleep(3)
            errors = []
            names = {0: "声音采集", 1: "动作采集"}
            for i, proc in enumerate(procs):
                if proc.poll() is not None:
                    errors.append(f"{names.get(i, f'进程{i}')} 异常退出，返回码: {proc.poll()}")
            if errors:
                msg = "\n".join(errors)
                msg += "\n\n可能原因：\n"
                msg += "- 动作采集：视频路径包含特殊字符，或 MediaPipe 模型加载失败\n"
                msg += "- 声音采集：麦克风被占用或无权限\n\n"
                msg += "请单独运行以下命令排查：\n"
                msg += f"python src\\audio_extractor.py\n"
                msg += f'python src\\motion_extractor.py --video "{self.video_path}"'
                self.root.after(0, lambda: messagebox.showwarning("启动警告", msg))
                self.root.after(0, lambda: self.btn_start.config(state="normal", text="启动"))
                self.root.after(0, lambda: self.status_label.config(text="启动异常，请看弹窗提示", fg="#FF6B6B"))

        threading.Thread(target=check, daemon=True).start()

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
        # 窗口关闭时清理子进程
        for proc in self.processes:
            try:
                proc.terminate()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
