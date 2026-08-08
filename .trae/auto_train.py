"""
StageDouble - 自动化训练脚本

[EN] Automates the training pipeline: starts audio_extractor + motion_extractor,
records 6 preset training states, trains the model, and runs live prediction.

[中] 自动化训练流程：启动声音采集和动作采集，录制 6 组预设状态，训练模型，
实时预测输出。

Usage / 使用方法：
    python .trae/auto_train.py
    python .trae/auto_train.py --video "data/videos/after like.mov"

User just needs to speak/sing into the microphone when prompted.
用户只需在提示时对着麦克风说话/唱歌。
"""

import os
import sys
import time
import subprocess
import argparse

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from ml_trainer_v2 import (
    MergedFeatureReceiver,
    Trainer,
    start_osc_server,
    record_examples,
    N_OUTPUTS,
    N_INPUT_FEATURES,
    SAMPLE_INTERVAL,
)
from pythonosc.udp_client import SimpleUDPClient

# 6 组预设训练状态
TRAINING_STATES = [
    {"name": "平静", "target": [0.1, 0.1, 0.5, 0.1, 0.2, 0.2], "desc": "正常说话，不大动"},
    {"name": "中等", "target": [0.4, 0.4, 0.5, 0.4, 0.4, 0.5], "desc": "说话带一点动作"},
    {"name": "强烈", "target": [0.8, 0.8, 0.7, 0.7, 0.7, 0.8], "desc": "大声唱歌，动作大"},
    {"name": "快速", "target": [0.5, 0.5, 0.5, 0.6, 0.5, 1.0], "desc": "快节奏说话/唱歌"},
    {"name": "慢速", "target": [0.3, 0.3, 0.4, 0.3, 0.3, 0.1], "desc": "慢慢说话"},
    {"name": "抬头", "target": [0.3, 0.5, 1.0, 0.4, 0.6, 0.4], "desc": "头抬高说话"},
]

RECORD_SECONDS = 4  # 每组录制秒数
PREDICT_SECONDS = 15  # 训练后预测展示秒数


def find_video():
    """[中] 在 data/videos/ 里找第一个视频文件。"""
    video_dir = os.path.join(ROOT_DIR, "data", "videos")
    if not os.path.isdir(video_dir):
        return None
    for f in sorted(os.listdir(video_dir)):
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            return os.path.join(video_dir, f)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=None, help="video file path (auto-detected if omitted)")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "gradient_boost", "mlp"])
    parser.add_argument("--skip-record", action="store_true", help="skip recording (use existing data)")
    args = parser.parse_args()

    video_path = args.video or find_video()
    if not video_path:
        print("[ERROR] No video found in data/videos/. Please add a video file.")
        return
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    print("=" * 60)
    print("  StageDouble 自动化训练")
    print("=" * 60)
    print(f"  视频: {os.path.basename(video_path)}")
    print(f"  模型: {args.model}")
    print(f"  录制组数: {len(TRAINING_STATES)}")
    print(f"  每组秒数: {RECORD_SECONDS}")
    print("=" * 60)
    print()

    # --- 启动采集程序 ---
    python = sys.executable
    audio_script = os.path.join(SRC_DIR, "audio_extractor.py")
    motion_script = os.path.join(SRC_DIR, "motion_extractor.py")

    print("[1/5] 启动声音采集...")
    creationflags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    p_audio = subprocess.Popen([python, audio_script], creationflags=creationflags)

    print("[2/5] 启动动作采集...")
    p_motion = subprocess.Popen([python, motion_script, "--video", video_path], creationflags=creationflags)

    # --- 初始化 OSC 接收 ---
    receiver = MergedFeatureReceiver()
    start_osc_server("127.0.0.1", 6448, "/wek/inputs", receiver.audio.update)
    start_osc_server("127.0.0.1", 6449, "/stagedouble/motion", receiver.motion.update)

    osc_out = SimpleUDPClient("127.0.0.1", 12000)
    trainer = Trainer(osc_out, "/stagedouble/outputs", args.model)

    # --- 等待数据就绪 ---
    print("\n[3/5] 等待数据就绪...", end="", flush=True)
    for _ in range(50):  # 最多等 5 秒
        status = receiver.status()
        if status["audio_ready"] and status["motion_ready"]:
            break
        time.sleep(0.1)
        print(".", end="", flush=True)
    print()

    status = receiver.status()
    if not status["audio_ready"]:
        print("[WARNING] 音频数据未就绪！请检查 audio_extractor 是否正常运行。")
    if not status["motion_ready"]:
        print("[WARNING] 动作数据未就绪！请检查 motion_extractor 是否正常运行。")

    if not status["audio_ready"] or not status["motion_ready"]:
        print("\n等待 10 秒后继续尝试...")
        time.sleep(10)
        status = receiver.status()

    if not status["audio_ready"] or not status["motion_ready"]:
        print("[ERROR] 数据源未就绪，无法继续。请确认两个采集程序都在运行。")
        p_audio.terminate()
        p_motion.terminate()
        return

    print("  音频: OK")
    print("  动作: OK")

    # --- 录制训练数据 ---
    if not args.skip_record:
        print(f"\n[4/5] 开始录制 {len(TRAINING_STATES)} 组训练数据")
        print("  每组录制时请对着麦克风说话/唱歌！")
        print()

        for i, state in enumerate(TRAINING_STATES):
            print(f"  --- 第 {i+1}/{len(TRAINING_STATES)} 组: {state['name']} ---")
            print(f"  描述: {state['desc']}")
            print(f"  目标值: {state['target']}")
            print(f"  >>> 请现在开始说话/唱歌！ <<<")

            # 倒计时 1 秒让用户准备
            time.sleep(1)

            examples = record_examples(receiver, state["target"], RECORD_SECONDS)
            for features in examples:
                trainer.add_example(features, state["target"])
            print(f"  采集了 {len(examples)} 个样本\n")
    else:
        print("\n[4/5] 跳过录制")

    print(f"  总样本数: {len(trainer.examples_X)}")

    # --- 训练 ---
    if len(trainer.examples_X) < 2:
        print("\n[ERROR] 样本不足，至少需要 2 个。")
        p_audio.terminate()
        p_motion.terminate()
        return

    print("\n[5/5] 训练模型...")
    trainer.train()

    # --- 实时预测 ---
    print(f"\n训练完成！开始实时预测 {PREDICT_SECONDS} 秒...")
    print("对着麦克风说话/唱歌，看输出值变化：\n")

    labels = ["mouth", "express", "head", "body", "arm", "speed"]
    end_time = time.time() + PREDICT_SECONDS
    while time.time() < end_time:
        features = receiver.latest()
        if features is not None:
            out = trainer.predict(features)
            osc_out.send_message("/stagedouble/outputs", out.tolist())
            vals = "  ".join(f"{l}: {v:.3f}" for l, v in zip(labels, out))
            print(f"\r  {vals}    ", end="", flush=True)
        time.sleep(SAMPLE_INTERVAL)

    print("\n\n预测结束。")

    # --- 清理 ---
    print("\n关闭采集程序...")
    p_audio.terminate()
    p_motion.terminate()

    print("\n" + "=" * 60)
    print("  自动化训练完成！")
    print(f"  训练样本: {len(trainer.examples_X)} 个")
    print(f"  模型: {args.model}")
    print("  如需使用图形界面，运行: python start.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
