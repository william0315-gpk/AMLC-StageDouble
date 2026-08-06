"""
StageDouble - Real-Time Dance Motion Feature Extractor
StageDouble - 实时舞蹈动作特征提取器

[EN] What this file does:
Reads a dance video file (mp4 or similar), runs Google MediaPipe's BlazePose
model (via MediaPipe's current Tasks API, PoseLandmarker) on every frame to
detect 33 body keypoints (x, y, z per point = 99 values), and streams that
99-value feature vector over OSC in real time — mirroring the way
audio_extractor.py streams a 16-value vocal feature vector. If the video
ends, it loops back to the start so the motion stream never stops (useful
for long training/demo sessions).

[中] 这个文件做什么：
读取一段舞蹈视频文件（mp4 或类似格式），对每一帧画面运行 Google
MediaPipe 的 BlazePose 姿态检测模型（通过 MediaPipe 目前的 Tasks API，
即 PoseLandmarker），识别出人体上的 33 个关键点（每个点有 x, y, z 三个
坐标，一共 33 x 3 = 99 个数值），然后把这 99 个数值打包成一个特征向量，
通过 OSC 协议实时发送出去——工作方式和 audio_extractor.py 发送 16 维
人声特征向量完全对应。如果视频播放到结尾，会自动从头循环播放，这样
动作数据流就不会中断（方便长时间的训练/演示会话）。

[EN] A note on the MediaPipe API used here:
Older BlazePose tutorials use `mediapipe.solutions.pose`, but that legacy
"Solutions" API was removed from recent MediaPipe releases (0.10.10+) in
favor of the "Tasks" API used below (`mediapipe.tasks.python.vision.
PoseLandmarker`). The Tasks API needs an explicit pose-landmark model
bundle (a `.task` file) rather than bundling one automatically; this script
downloads the official Google-hosted model bundle to `models/`
the first time it runs and reuses the cached copy after that.

[中] 关于这里使用的 MediaPipe API 版本说明：
早期的 BlazePose 教程大多使用 `mediapipe.solutions.pose`，但这个旧版
"Solutions" API 已经从近期的 MediaPipe 版本（0.10.10 及以后）中移除，
被下面用到的 "Tasks" API（`mediapipe.tasks.python.vision.PoseLandmarker`）
取代。Tasks API 需要显式提供一个姿态关键点模型文件（`.task` 格式），
不会像旧版那样自动内置模型。本脚本第一次运行时会自动把 Google 官方
托管的模型文件下载到 `models/` 目录下，之后会直接复用这份
缓存，不会重复下载。

[EN] How to run:
    python motion_extractor.py --video dance.mp4
    python motion_extractor.py --video dance.mp4 --ip 127.0.0.1 --port 6449 --address /stagedouble/motion
    python motion_extractor.py --video dance.mp4 --no-loop     # play once and stop
    python motion_extractor.py --video dance.mp4 --show         # show a debug preview window

[中] 如何运行：
    python motion_extractor.py --video dance.mp4
    python motion_extractor.py --video dance.mp4 --ip 127.0.0.1 --port 6449 --address /stagedouble/motion
    python motion_extractor.py --video dance.mp4 --no-loop     # 只播放一遍，播完就停止（不循环）
    python motion_extractor.py --video dance.mp4 --show         # 显示一个调试用的预览窗口

[EN] What this connects to:
This is the second input stream of the StageDouble vol.2 pipeline
(alongside audio_extractor.py). It sends its 99-value OSC feature vector
to ml_trainer_v2.py, which listens on 127.0.0.1:6449 /stagedouble/motion
by default and merges it with the 16-value audio stream into a single
115-dimensional input vector. Nothing needs to be running for this script
itself to work — like audio_extractor.py, it will happily send OSC
packets into the void if no one is listening.

[中] 这个文件如何与其他文件连接：
这是 StageDouble vol.2 流水线的第二路输入（与 audio_extractor.py 并列）。
它把提取好的 99 维特征向量通过 OSC 发送给 ml_trainer_v2.py，该脚本默认
监听 127.0.0.1:6449 /stagedouble/motion，并把这一路数据和音频那一路
16 维数据合并成一个 115 维的输入向量。这个脚本本身不依赖下游程序是否
在运行——和 audio_extractor.py 一样，即使没有人监听，它也会正常地把
OSC 数据包发送出去（只是没人收到而已）。
"""

import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from pythonosc.udp_client import SimpleUDPClient

# ---------------------------------------------------------------------------
# Configuration / 配置参数
# ---------------------------------------------------------------------------

N_LANDMARKS = 33  # BlazePose 输出的人体关键点数量（固定值，由模型本身决定）。
N_VALUES_PER_LANDMARK = 3  # 每个关键点包含 x, y, z 三个坐标值。
N_VALUES = N_LANDMARKS * N_VALUES_PER_LANDMARK  # 99：整个特征向量的总长度。

# Wekinator-style default matching the rest of the StageDouble pipeline:
# audio goes to port 6448, motion goes to its own port 6449 so the two
# streams never collide, both are consumed by ml_trainer_v2.py.
#
# 沿用 StageDouble 流水线里 Wekinator 风格的默认约定：音频走 6448 端口，
# 动作数据走独立的 6449 端口，这样两路数据流不会互相冲突，两者都会被
# ml_trainer_v2.py 接收。
DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 6449
DEFAULT_OSC_ADDRESS = "/stagedouble/motion"

# Target send rate. BlazePose inference is heavier than the audio feature
# pipeline, so we let cv2's native frame rate drive the loop instead of a
# fixed hop like audio_extractor.py's 10Hz — see extract loop below.
#
# 目标发送频率。BlazePose 姿态推理比音频特征提取更耗计算资源，所以这里
# 让视频本身的帧率来驱动循环节奏，而不是像 audio_extractor.py 那样固定
# 一个 10Hz 的节拍——具体逻辑见下面的主循环。
DEFAULT_FPS_FALLBACK = 30.0  # 如果读不到视频的帧率信息，就退回使用这个默认值。

# Official Google-hosted PoseLandmarker model bundles. "lite" is the
# fastest/smallest, good enough for real-time keypoint streaming; "full"
# and "heavy" trade speed for accuracy.
#
# Google 官方托管的 PoseLandmarker 模型文件。"lite" 是速度最快、体积最小
# 的版本，对实时关键点流来说已经足够；"full" 和 "heavy" 用速度换精度。
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
# 模型文件的本地缓存目录：项目根目录下的 models/


def parse_args():
    """[中] 解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="path to the dance video file (mp4 etc.)")
    parser.add_argument("--ip", default=DEFAULT_OSC_IP, help="target host/IP to send OSC features to")
    parser.add_argument("--port", type=int, default=DEFAULT_OSC_PORT, help="target OSC port")
    parser.add_argument("--address", default=DEFAULT_OSC_ADDRESS, help="OSC address the receiver listens on")
    parser.add_argument("--no-loop", action="store_true", help="play the video once instead of looping forever")
    parser.add_argument("--show", action="store_true", help="show a debug preview window with pose overlay")
    parser.add_argument(
        "--model",
        default="lite",
        choices=list(MODEL_URLS),
        help="BlazePose model size: lite=fastest, full=balanced, heavy=most accurate",
    )
    return parser.parse_args()


def ensure_model_downloaded(model_name):
    """[EN] Return the local path to the requested PoseLandmarker model
    bundle, downloading it from Google's official model hosting into
    MODEL_DIR the first time it's needed.
    [中] 返回所请求的 PoseLandmarker 模型文件在本地的路径；如果是第一次
    使用，会先从 Google 官方模型托管地址下载到 MODEL_DIR 目录，之后
    直接复用本地缓存的文件。
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"pose_landmarker_{model_name}.task")
    if not os.path.exists(model_path):
        url = MODEL_URLS[model_name]
        print(f"Downloading pose model ({model_name}) from {url} ...")
        urllib.request.urlretrieve(url, model_path)
        print(f"Saved model to {model_path}")
    return model_path


def landmarks_to_feature_vector(pose_landmarks):
    """[EN] Flatten one detected person's PoseLandmarker landmark list into
    a flat list of 99 floats: [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32].
    x/y are normalized to [0, 1] relative to image width/height; z is a
    rough depth relative to the hips, in the same normalized scale.

    [中] 把 PoseLandmarker 返回的（某一个人的）姿态关键点列表，展平成
    一个长度为 99 的浮点数列表：[x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]。
    x、y 坐标已经被归一化到 [0, 1] 区间（相对于图像的宽高）；z 坐标是一个
    相对于髋部（hips）的粗略深度值，使用同样的归一化比例尺。
    """
    values = []
    for landmark in pose_landmarks:
        values.extend([landmark.x, landmark.y, landmark.z])
    return values


def main():
    """[EN] Entry point: open the video file, run BlazePose frame-by-frame,
    and stream the resulting keypoint vector over OSC, looping the video
    by default.
    [中] 主入口函数：打开视频文件，逐帧运行 BlazePose 姿态检测，把每一帧
    得到的关键点向量通过 OSC 发送出去；默认会循环播放视频。
    """
    args = parse_args()

    osc_client = SimpleUDPClient(args.ip, args.port)
    print(f"Sending OSC to {args.ip}:{args.port}{args.address}")

    model_path = ensure_model_downloaded(args.model)

    # PoseLandmarker 是 MediaPipe 目前的 Tasks API 里负责姿态检测的类。
    # running_mode=VIDEO 表示按视频帧处理，内部会利用帧间时间戳做跟踪，
    # 比逐帧当独立图片处理（IMAGE 模式）更稳定、更快。
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video file: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS_FALLBACK
    frame_interval = 1.0 / fps
    # 每一帧之间应该间隔多久（秒），用来把处理速度限制在视频原本的帧率上，
    # 这样才是"实时"播放而不是把视频尽可能快地一次性处理完。
    print(f"Video opened: {args.video} ({fps:.1f} fps). Press Ctrl+C to stop.")
    if not args.no_loop:
        print("Looping enabled: video will restart automatically when it ends.")

    frame_count = 0  # 已经喂给 PoseLandmarker 的总帧数（跨越多次循环持续累加）。
    # PoseLandmarker 在 VIDEO 模式下要求时间戳必须单调递增，所以即使视频
    # 循环重播，也不能把这个计数器重置回 0——必须一直往上累加。
    no_pose_count = 0
    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()

            if not ok:
                # 视频读到结尾了（或读取失败）。
                if args.no_loop:
                    print("\nVideo finished (looping disabled). Stopping.")
                    break
                # 循环模式：把读取位置重置回视频开头，继续播放。
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # MediaPipe 需要 RGB 格式的图像，而 OpenCV 读取到的默认是 BGR，
            # 所以这里要做一次颜色空间转换，再包装成 mp.Image 对象。
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_count * frame_interval * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_count += 1

            if result.pose_landmarks:
                # 检测到人体姿态：取第一个人（num_poses=1），展平成 99 维
                # 向量并通过 OSC 发送。
                features = landmarks_to_feature_vector(result.pose_landmarks[0])
                osc_client.send_message(args.address, features)
                print(f"\rframe {frame_count}: sent 99-value pose vector   ", end="", flush=True)
            else:
                # 这一帧没有检测到人（比如人物暂时走出画面），跳过发送，
                # 但打印提醒方便排查问题（比如视频里根本没有人）。
                no_pose_count += 1
                print(f"\rframe {frame_count}: no pose detected ({no_pose_count} total)   ", end="", flush=True)

            if args.show:
                # 调试预览：把检测到的骨架关键点叠加画在原始画面上再显示出来。
                if result.pose_landmarks:
                    h, w = frame.shape[:2]
                    for landmark in result.pose_landmarks[0]:
                        cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 3, (0, 255, 0), -1)
                cv2.imshow("motion_extractor preview (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 按视频原本的帧率来节流，避免比真实播放速度快太多地把 OSC
            # 消息一股脑全部发出去。
            elapsed = time.time() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        landmarker.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
