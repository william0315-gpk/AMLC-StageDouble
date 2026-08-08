"""
StageDouble - Dance Motion Style Extractor & Style Library Builder
StageDouble - 舞蹈动作风格提取器与风格库构建工具

[EN] What this file does:
This module is the "offline analysis" companion to motion_extractor.py. Where
motion_extractor.py streams a live 99-value pose vector over OSC in real time,
this module instead reads a whole dance video offline, runs the same MediaPipe
PoseLandmarker (BlazePose) on every frame, and distills the resulting pose
sequence into two kinds of higher-level representations that the StageDouble
project can reuse later:

  1. A 6-dim param time series [mouth, expr, head, body, arm, speed] -- the
     same 6 output dimensions that ml_trainer_v2.py predicts at runtime. Here
     mouth/expr are left at 0.0 (they are driven by audio at runtime) while
     head/body/arm/speed are derived purely from the skeleton.
  2. A small set of statistical style features (avg_speed, avg_energy,
     spatial_extent, rhythm) that summarize the whole clip, plus an auto-
     assigned emotion position [arousal, valence].

A directory of dance videos can be batch-processed into a single pickle
"style library" via build_style_library(), so downstream components can pick a
motion style by name/emotion without re-running pose detection.

[中] 这个文件做什么：
这个模块是 motion_extractor.py 的"离线分析"配套工具。motion_extractor.py
负责实时地把 99 维姿态向量通过 OSC 流式发送出去；而本模块则是离线地读取
整段舞蹈视频，对每一帧运行同样的 MediaPipe PoseLandmarker（BlazePose），
再把得到的姿态序列提炼成两类更高层的表示，供 StageDouble 项目后续复用：

  1. 6 维参数时间序列 [mouth, expr, head, body, arm, speed]--和
     ml_trainer_v2.py 在运行时预测的 6 个输出维度完全对应。这里 mouth/expr
     固定为 0.0（运行时由音频驱动），而 head/body/arm/speed 完全由骨架
     推导得到。
  2. 一组统计风格特征（avg_speed、avg_energy、spatial_extent、rhythm），
     用来概括整段视频，并自动给出一个情绪坐标 [arousal, valence]。

通过 build_style_library() 可以把一个目录下的所有舞蹈视频批量处理成一个
pickle "风格库"，这样下游组件就可以直接按名称/情绪选用某种动作风格，而不
必每次都重新跑一遍姿态检测。

[EN] A note on the MediaPipe API used here:
This reuses the exact same Tasks-API PoseLandmarker pattern as
motion_extractor.py (`mediapipe.tasks.python.vision.PoseLandmarker`, VIDEO
mode). The official Google-hosted model bundle is downloaded to `models/` on
first use and cached afterwards. See motion_extractor.py for more background
on why the legacy `mediapipe.solutions.pose` API is not used.

[中] 关于这里使用的 MediaPipe API 版本说明：
本模块复用了和 motion_extractor.py 完全一致的 Tasks API PoseLandmarker
用法（`mediapipe.tasks.python.vision.PoseLandmarker`，VIDEO 模式）。首次
使用时会自动把 Google 官方托管的模型文件下载到 `models/` 目录，之后直接
复用缓存。关于为什么不使用旧版 `mediapipe.solutions.pose` API，详见
motion_extractor.py 里的说明。

[EN] How to run (build a style library from a folder of videos):
    python style_extractor.py --videos-dir ./dance_clips --output style_library.pkl
    python style_extractor.py --videos-dir ./dance_clips --output style_library.pkl \\
        --labels labels.json

[中] 如何运行（从一个视频文件夹构建风格库）：
    python style_extractor.py --videos-dir ./dance_clips --output style_library.pkl
    python style_extractor.py --videos-dir ./dance_clips --output style_library.pkl \\
        --labels labels.json

[EN] What this connects to:
This is an offline preprocessing step for the StageDouble pipeline. The pickle
it writes can be loaded by any downstream component that needs to pick a motion
style (for example, to drive a digital human with a precomputed dance style
instead of live pose capture). It does not talk over OSC itself.

[中] 这个文件如何与其他文件连接：
这是 StageDouble 流水线的离线预处理步骤。它写出的 pickle 文件可以被任何
需要选用动作风格的下游组件加载（例如：用预先计算好的舞蹈风格来驱动数字
人，而不是依赖实时姿态采集）。本模块自身不收发 OSC。
"""

import argparse
import glob
import json
import os
import pickle
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuration / 配置参数
# ---------------------------------------------------------------------------

N_LANDMARKS = 33  # BlazePose 输出的人体关键点数量（固定值，由模型本身决定）。
N_VALUES_PER_LANDMARK = 3  # 每个关键点包含 x, y, z 三个坐标值。
N_VALUES = N_LANDMARKS * N_VALUES_PER_LANDMARK  # 99：单帧姿态向量的总长度。

# BlazePose 关键点索引（仅列出本模块用到的几个）。
# BlazePose landmark indices (only the ones used by this module).
LM_NOSE = 0              # 鼻子
LM_LEFT_SHOULDER = 11    # 左肩
LM_RIGHT_SHOULDER = 12   # 右肩
LM_LEFT_WRIST = 15       # 左手腕
LM_RIGHT_WRIST = 16      # 右手腕
LM_LEFT_HIP = 23         # 左髋
LM_RIGHT_HIP = 24        # 右髋

# 降采样后最多保留多少个关键帧。视频帧数很多时，每隔 N 帧取一帧，把序列
# 压缩到这个长度以内，既保留整体动作风格，又避免风格库里存太多冗余帧。
MAX_KEYFRAMES = 300

DEFAULT_FPS_FALLBACK = 30.0  # 读不到视频帧率时退回使用的默认值。

# Official Google-hosted PoseLandmarker model bundles. 与 motion_extractor.py
# 保持一致：lite 最快、体积最小，适合批量离线处理。
#
# Google 官方托管的 PoseLandmarker 模型文件。和 motion_extractor.py 保持
# 一致：lite 速度最快、体积最小，适合批量离线处理。
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
# 模型文件的本地缓存目录：项目根目录下的 models/

DEFAULT_MODEL_NAME = "lite"  # 默认使用的模型规格。


# ---------------------------------------------------------------------------
# Model download / 模型下载（复用 motion_extractor.py 的模式）
# ---------------------------------------------------------------------------

def ensure_model_downloaded(model_name=DEFAULT_MODEL_NAME):
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

    [中] 把 PoseLandmarker 返回的（某一个人的）姿态关键点列表，展平成
    一个长度为 99 的浮点数列表：[x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]。
    """
    values = []
    for landmark in pose_landmarks:
        values.extend([landmark.x, landmark.y, landmark.z])
    return values


# ---------------------------------------------------------------------------
# Pose -> param conversion / 姿态序列 -> 参数时间序列
# ---------------------------------------------------------------------------

def poses_to_params(pose_sequence):
    """[EN] Convert a sequence of 99-dim pose frames into a 6-dim param time
    series, one row per frame: [mouth, expr, head, body, arm, speed].

    Each pose frame is laid out as
        [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]   (33 landmarks x 3)
    with x/y normalized to [0, 1] by MediaPipe (y points DOWN, as is standard
    for image coordinates).

    The 6 outputs mirror ml_trainer_v2.py's 6 output dimensions:
      - mouth : always 0.0 (driven by audio at runtime)
      - expr  : always 0.0 (driven by audio at runtime)
      - head  : nose (landmark 0) vertical position relative to the body
                center (mean of shoulders 11/12 + hips 23/24), normalized so
                0.5 = nose level with body center; higher = head raised.
      - body  : frame-to-frame displacement of the body center, normalized by
                torso length; 0 for the first frame.
      - arm   : average wrist height (landmarks 15/16) relative to shoulder
                height (landmarks 11/12), normalized so 0.5 = wrists at
                shoulder level; higher = arms raised.
      - speed : average frame-to-frame Euclidean distance across all 33
                landmarks, normalized by torso length; 0 for the first frame.

    All outputs are clipped to [0, 1]. Torso length (|hip_y - shoulder_y|) is
    used as a per-frame scale reference so the features are roughly invariant
    to how large the dancer appears in frame.

    [中] 把一串 99 维的姿态帧转换成 6 维的参数时间序列，每帧一行：
    [mouth, expr, head, body, arm, speed]。

    每一帧姿态的排布是
        [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]   （33 个关键点 x 3）
    其中 x/y 已被 MediaPipe 归一化到 [0, 1]（y 轴朝下，和图像坐标一致）。

    6 个输出与 ml_trainer_v2.py 的 6 个输出维度一一对应：
      - mouth : 恒为 0.0（运行时由音频驱动）
      - expr  : 恒为 0.0（运行时由音频驱动）
      - head  : 鼻子（关键点 0）相对于身体中心（肩 11/12 + 髋 23/24 的均值）
                的垂直位置，归一化后 0.5 = 鼻子与身体中心齐平；值越大表示
                头抬得越高。
      - body  : 身体中心的逐帧位移，用躯干长度归一化；第一帧为 0。
      - arm   : 手腕平均高度（关键点 15/16）相对于肩膀高度（关键点 11/12）
                的位置，归一化后 0.5 = 手腕与肩膀齐平；值越大表示手臂抬得
                越高。
      - speed : 全部 33 个关键点的逐帧欧氏距离均值，用躯干长度归一化；
                第一帧为 0。

    所有输出都截断到 [0, 1]。使用躯干长度（|髋 y - 肩 y|）作为逐帧的尺度
    参考，使特征大致不受舞者在画面中所占大小的影响。
    """
    poses = np.asarray(pose_sequence, dtype=float)
    if poses.ndim == 1:
        # 单帧输入也兼容：变成 (1, 99)。
        poses = poses.reshape(1, -1)
    n_frames = poses.shape[0]

    params = []
    prev_body_center = None  # 上一帧的身体中心 (cx, cy)，用于计算 body 位移。
    prev_pose = None         # 上一帧的 99 维向量，用于计算 speed。

    for i in range(n_frames):
        pose = poses[i]

        # --- 躯干长度：作为本帧的尺度参考 ---
        shoulder_y = (pose[LM_LEFT_SHOULDER * 3 + 1] + pose[LM_RIGHT_SHOULDER * 3 + 1]) / 2.0
        hip_y = (pose[LM_LEFT_HIP * 3 + 1] + pose[LM_RIGHT_HIP * 3 + 1]) / 2.0
        torso = abs(hip_y - shoulder_y) + 1e-6  # 加小常数避免除零。

        # --- 身体中心：肩 11/12 + 髋 23/24 的 x/y 均值 ---
        body_cx = (pose[LM_LEFT_SHOULDER * 3] + pose[LM_RIGHT_SHOULDER * 3]
                   + pose[LM_LEFT_HIP * 3] + pose[LM_RIGHT_HIP * 3]) / 4.0
        body_cy = (pose[LM_LEFT_SHOULDER * 3 + 1] + pose[LM_RIGHT_SHOULDER * 3 + 1]
                   + pose[LM_LEFT_HIP * 3 + 1] + pose[LM_RIGHT_HIP * 3 + 1]) / 4.0

        # --- head：鼻子 y 相对于身体中心 y ---
        # y 轴朝下，鼻子在身体中心上方时 nose_y < body_cy，offset > 0。
        nose_y = pose[LM_NOSE * 3 + 1]
        head_offset = (body_cy - nose_y) / torso  # 头高于中心时为正。
        head = 0.5 + head_offset / 4.0             # 0.5 = 与中心齐平。

        # --- arm：手腕高度相对于肩膀高度 ---
        # 手腕抬到肩膀以上时 wrist_y < shoulder_y，offset > 0。
        wrist_y = (pose[LM_LEFT_WRIST * 3 + 1] + pose[LM_RIGHT_WRIST * 3 + 1]) / 2.0
        arm_offset = (shoulder_y - wrist_y) / torso  # 手腕高于肩膀时为正。
        arm = 0.5 + arm_offset / 4.0                  # 0.5 = 手腕与肩膀齐平。

        # --- body：身体中心的逐帧位移 ---
        if prev_body_center is None:
            body = 0.0
        else:
            dx = body_cx - prev_body_center[0]
            dy = body_cy - prev_body_center[1]
            body = float(np.sqrt(dx * dx + dy * dy)) / torso

        # --- speed：33 个关键点的逐帧欧氏距离均值 ---
        if prev_pose is None:
            speed = 0.0
        else:
            diff = pose.reshape(N_LANDMARKS, N_VALUES_PER_LANDMARK) \
                - prev_pose.reshape(N_LANDMARKS, N_VALUES_PER_LANDMARK)
            dists = np.sqrt(np.sum(diff * diff, axis=1))  # 每个关键点的位移。
            speed = float(np.mean(dists)) / torso

        # mouth / expr 运行时由音频驱动，这里固定为 0.0。
        params.append([
            0.0,  # mouth
            0.0,  # expr
            float(np.clip(head, 0.0, 1.0)),
            float(np.clip(body, 0.0, 1.0)),
            float(np.clip(arm, 0.0, 1.0)),
            float(np.clip(speed, 0.0, 1.0)),
        ])

        prev_body_center = (body_cx, body_cy)
        prev_pose = pose

    return params


# ---------------------------------------------------------------------------
# Statistical style features / 统计风格特征
# ---------------------------------------------------------------------------

def compute_style_features(pose_sequence):
    """[EN] Compute statistical style features from a pose sequence.

    Returns a dict with 4 keys, all normalized to [0, 1]:
      - avg_speed      : mean frame-to-frame movement (normalized by torso).
      - avg_energy     : mean (over time) of the variance of per-landmark
                         movement across the body -- how unevenly motion is
                         distributed over the skeleton (normalized by torso).
      - spatial_extent : largest bounding-box range (max of width/height) of
                         all landmarks across the whole sequence; already in
                         [0, 1] because MediaPipe coords are normalized.
      - rhythm         : regularity of movement = 1 - coefficient of variation
                         of inter-frame distances; higher = more regular.

    [中] 从姿态序列中计算统计风格特征。

    返回一个包含 4 个键的字典，全部归一化到 [0, 1]：
      - avg_speed      : 逐帧运动的均值（用躯干长度归一化）。
      - avg_energy     : 逐帧地，先计算"各关键点位移在身体上的方差"，再对
                         时间取均值--衡量动作在身体各部位间分布的不均匀
                         程度（用躯干长度归一化）。
      - spatial_extent : 整段序列中所有关键点的最大包围盒范围（宽/高的
                         较大者）；由于 MediaPipe 坐标已归一化，本身就在
                         [0, 1] 内。
      - rhythm         : 运动的规律性 = 1 - 逐帧距离的变异系数；值越大表示
                         越规律。
    """
    poses = np.asarray(pose_sequence, dtype=float)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)

    # 空序列或极短序列的兜底：返回全 0 特征。
    if poses.shape[0] == 0:
        return {"avg_speed": 0.0, "avg_energy": 0.0, "spatial_extent": 0.0, "rhythm": 0.0}

    n_frames = poses.shape[0]

    # --- 躯干长度（逐帧取均值，作为整体尺度参考）---
    shoulder_y = (poses[:, LM_LEFT_SHOULDER * 3 + 1] + poses[:, LM_RIGHT_SHOULDER * 3 + 1]) / 2.0
    hip_y = (poses[:, LM_LEFT_HIP * 3 + 1] + poses[:, LM_RIGHT_HIP * 3 + 1]) / 2.0
    torso = float(np.mean(np.abs(hip_y - shoulder_y))) + 1e-6

    # --- 逐帧运动：相邻两帧之间 33 个关键点的欧氏距离 ---
    if n_frames < 2:
        movements = np.array([0.0])           # 只有一帧，无法算逐帧运动。
        per_landmark_dists = np.zeros((1, N_LANDMARKS))
    else:
        prev = poses[:-1].reshape(-1, N_LANDMARKS, N_VALUES_PER_LANDMARK)
        curr = poses[1:].reshape(-1, N_LANDMARKS, N_VALUES_PER_LANDMARK)
        diffs = curr - prev                               # (T-1, 33, 3)
        per_landmark_dists = np.sqrt(np.sum(diffs * diffs, axis=2))  # (T-1, 33)
        movements = np.mean(per_landmark_dists, axis=1)   # (T-1,) 每帧的均值位移

    mean_mov = float(np.mean(movements))
    std_mov = float(np.std(movements))

    # avg_speed：逐帧均值位移，用躯干长度归一化。
    avg_speed = mean_mov / torso

    # avg_energy：每帧"各关键点位移在身体上的方差"的时间均值，用躯干长度的
    # 平方归一化，使其与尺度无关。
    energy_raw = float(np.mean(np.var(per_landmark_dists, axis=1)))
    avg_energy = energy_raw / (torso * torso)

    # spatial_extent：整段序列中所有关键点的最大包围盒范围（宽/高的较大者）。
    xs = poses[:, 0::3]  # 所有关键点的 x 坐标 (T, 33)
    ys = poses[:, 1::3]  # 所有关键点的 y 坐标 (T, 33)
    widths = np.max(xs, axis=1) - np.min(xs, axis=1)
    heights = np.max(ys, axis=1) - np.min(ys, axis=1)
    extents = np.maximum(widths, heights)
    spatial_extent = float(np.max(extents))  # 坐标本身在 [0,1]，故也在 [0,1] 内。

    # rhythm：规律性 = 1 - 变异系数（std/mean）。变异系数越小越规律。
    cv = std_mov / (mean_mov + 1e-6)
    rhythm = 1.0 - cv

    return {
        "avg_speed": float(np.clip(avg_speed, 0.0, 1.0)),
        "avg_energy": float(np.clip(avg_energy, 0.0, 1.0)),
        "spatial_extent": float(np.clip(spatial_extent, 0.0, 1.0)),
        "rhythm": float(np.clip(rhythm, 0.0, 1.0)),
    }


# ---------------------------------------------------------------------------
# Single-video extraction / 单视频风格提取
# ---------------------------------------------------------------------------

def extract_style_from_video(video_path, model_path=None):
    """[EN] Extract motion style from a single dance video.

    Runs MediaPipe PoseLandmarker (VIDEO mode, same setup as
    motion_extractor.py) over every frame, collects the 99-dim pose sequence,
    downsamples to at most MAX_KEYFRAMES keyframes, then computes both the
    param time series and the statistical style features. An emotion position
    [arousal, valence] is auto-assigned from the style features:
        arousal = avg_speed
        valence = clip(spatial_extent - avg_energy, 0, 1)

    Args:
        video_path : path to the dance video file.
        model_path : optional explicit path to a PoseLandmarker .task bundle.
                     If None, the default "lite" model is auto-downloaded.

    Returns:
        dict with keys: name, pose_sequence, param_sequence, style_features,
        emotion_pos, emotion_label (None).

    [中] 从单段舞蹈视频中提取动作风格。

    对每一帧运行 MediaPipe PoseLandmarker（VIDEO 模式，配置与
    motion_extractor.py 一致），收集 99 维姿态序列，降采样到最多
    MAX_KEYFRAMES 个关键帧，然后同时计算参数时间序列和统计风格特征。
    情绪坐标 [arousal, valence] 由风格特征自动给出：
        arousal = avg_speed
        valence = clip(spatial_extent - avg_energy, 0, 1)

    参数：
        video_path : 舞蹈视频文件路径。
        model_path : 可选，显式指定 PoseLandmarker 的 .task 模型文件路径。
                     为 None 时自动下载默认的 "lite" 模型。

    返回：
        字典，包含键：name、pose_sequence、param_sequence、style_features、
        emotion_pos、emotion_label（为 None）。
    """
    if model_path is None:
        model_path = ensure_model_downloaded(DEFAULT_MODEL_NAME)

    # PoseLandmarker 配置与 motion_extractor.py 完全一致：VIDEO 模式、单人。
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        landmarker.close()
        raise SystemExit(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS_FALLBACK
    frame_interval = 1.0 / fps  # 每帧的时间间隔（秒），用于生成单调递增的时间戳。

    pose_sequence = []
    frame_idx = 0       # 已经喂给 PoseLandmarker 的帧计数（时间戳必须单调递增）。
    no_pose_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # 视频读完。

            # MediaPipe 需要 RGB，而 OpenCV 默认读到的是 BGR，做一次转换。
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * frame_interval * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if result.pose_landmarks:
                # 检测到人体：取第一个人，展平成 99 维向量存下来。
                pose_sequence.append(landmarks_to_feature_vector(result.pose_landmarks[0]))
            else:
                no_pose_count += 1
    finally:
        cap.release()
        landmarker.close()

    print(f"  extracted {len(pose_sequence)} pose frame(s) "
          f"({no_pose_count} frame(s) without pose)")

    # 整段视频都没检测到人：返回空风格条目，避免后续计算崩溃。
    if not pose_sequence:
        return {
            "name": os.path.splitext(os.path.basename(video_path))[0],
            "pose_sequence": [],
            "param_sequence": [],
            "style_features": {"avg_speed": 0.0, "avg_energy": 0.0,
                               "spatial_extent": 0.0, "rhythm": 0.0},
            "emotion_pos": [0.0, 0.0],
            "emotion_label": None,
        }

    # 降采样：帧数过多时每隔 N 帧取一帧，限制到约 MAX_KEYFRAMES 个关键帧。
    if len(pose_sequence) > MAX_KEYFRAMES:
        stride = int(np.ceil(len(pose_sequence) / MAX_KEYFRAMES))
        pose_sequence = pose_sequence[::stride]
        print(f"  downsampled to {len(pose_sequence)} keyframe(s) (stride={stride})")

    param_sequence = poses_to_params(pose_sequence)
    style_features = compute_style_features(pose_sequence)

    # 自动情绪坐标：arousal=avg_speed，valence=spatial_extent-avg_energy。
    arousal = style_features["avg_speed"]
    valence = style_features["spatial_extent"] - style_features["avg_energy"]
    emotion_pos = [
        float(np.clip(arousal, 0.0, 1.0)),
        float(np.clip(valence, 0.0, 1.0)),
    ]

    return {
        "name": os.path.splitext(os.path.basename(video_path))[0],
        "pose_sequence": pose_sequence,
        "param_sequence": param_sequence,
        "style_features": style_features,
        "emotion_pos": emotion_pos,
        "emotion_label": None,
    }


# ---------------------------------------------------------------------------
# Style library builder / 风格库构建
# ---------------------------------------------------------------------------

def build_style_library(videos_dir, output_path, emotion_labels=None):
    """[EN] Process all videos in a directory into a pickle style library.

    Finds every .mp4/.avi/.mov/.mkv file in `videos_dir`, extracts a style
    dict from each via extract_style_from_video(), optionally overrides the
    emotion position from `emotion_labels` (a dict mapping filename to
    [arousal, valence]), and pickles the resulting list to `output_path`.

    Args:
        videos_dir     : directory containing dance videos.
        output_path    : where to write the pickle file.
        emotion_labels : optional dict {filename: [arousal, valence]} to
                         override auto-assigned emotion positions.

    Returns:
        list of style dicts (also written to output_path as pickle).

    [中] 把一个目录下的所有视频处理成一个 pickle 风格库。

    找出 `videos_dir` 下所有 .mp4/.avi/.mov/.mkv 文件，对每个视频调用
    extract_style_from_video() 提取风格字典，可选择用 `emotion_labels`
    （一个 {文件名: [arousal, valence]} 的字典）覆盖自动给出的情绪坐标，
    最后把得到的列表 pickle 写入 `output_path`。

    参数：
        videos_dir     : 包含舞蹈视频的目录。
        output_path    : pickle 文件的输出路径。
        emotion_labels : 可选字典 {文件名: [arousal, valence]}，用来覆盖
                         自动计算的情绪坐标。

    返回：
        风格字典的列表（同时也会以 pickle 形式写入 output_path）。
    """
    # 收集目录下所有支持格式的视频文件（大小写都覆盖，去重并排序）。
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(videos_dir, ext)))
        video_files.extend(glob.glob(os.path.join(videos_dir, ext.upper())))
    video_files = sorted(set(video_files))

    print(f"Found {len(video_files)} video file(s) in {videos_dir}")

    style_library = []
    for i, vpath in enumerate(video_files, 1):
        fname = os.path.basename(vpath)
        print(f"[{i}/{len(video_files)}] Processing {fname} ...")
        try:
            style_dict = extract_style_from_video(vpath)
        except Exception as exc:  # 某个视频失败时不中断整个批次。
            print(f"  ERROR processing {fname}: {exc}")
            continue

        # 如果提供了人工情绪标签，覆盖自动计算的情绪坐标。
        if emotion_labels and fname in emotion_labels:
            style_dict["emotion_pos"] = [float(v) for v in emotion_labels[fname]]
            print(f"  emotion_pos overridden by label -> {style_dict['emotion_pos']}")

        style_library.append(style_dict)
        print(f"  done: emotion_pos={style_dict['emotion_pos']}")

    # 把整个风格库序列化成 pickle 文件。
    with open(output_path, "wb") as f:
        pickle.dump(style_library, f)
    print(f"Saved style library ({len(style_library)} entr(y)) to {output_path}")
    return style_library


# ---------------------------------------------------------------------------
# CLI entry point / 命令行入口
# ---------------------------------------------------------------------------

def parse_args():
    """[中] 解析命令行参数：视频目录、输出路径、可选的情绪标签文件。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--videos-dir", required=True, help="directory containing dance videos")
    parser.add_argument("--output", required=True, help="output pickle path for the style library")
    parser.add_argument(
        "--labels",
        default=None,
        help="optional JSON file: {filename: [arousal, valence]} to override emotion positions",
    )
    return parser.parse_args()


def main():
    """[EN] Entry point: build a style library from a folder of videos.
    [中] 主入口函数：从一个视频文件夹构建风格库。
    """
    args = parse_args()

    emotion_labels = None
    if args.labels:
        # 从 JSON 文件读取人工情绪标签：{filename: [arousal, valence]}。
        with open(args.labels, "r", encoding="utf-8") as f:
            emotion_labels = json.load(f)
        print(f"Loaded emotion labels for {len(emotion_labels)} video(s) from {args.labels}")

    build_style_library(args.videos_dir, args.output, emotion_labels=emotion_labels)


if __name__ == "__main__":
    main()
