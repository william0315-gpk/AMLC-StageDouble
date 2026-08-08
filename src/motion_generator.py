"""
StageDouble - Motion Generator (Style Weights + Audio -> 6-dim Output)
StageDouble - 动作生成器（风格权重 + 音频特征 -> 6 维输出）

[EN] What this file does:
Takes style weights (from StyleMatcher) and audio features (from
audio_extractor.py) and produces one frame of the 6-value output vector that
drives the MetaHuman / 2D preview:
    [mouth, expression, head_angle, body_amplitude, arm_height, movement_speed]

The six outputs are split by causal source:
  - Face params (mouth, expr)   <- driven directly by AUDIO features
      (volume -> mouth, pitch variation + MFCC energy -> expr)
  - Body params (head, body, arm, speed) <- driven by STYLE weights
      (weighted interpolation of each style's precomputed param_sequence)

[中] 这个文件做什么：
接收风格权重（来自 StyleMatcher）和音频特征（来自 audio_extractor.py），
生成一帧 6 维输出向量，用于驱动 MetaHuman / 2D 预览小人：
    [嘴巴, 表情, 头部角度, 身体幅度, 手臂高度, 动作速度]

六个输出按因果来源分成两组：
  - 面部参数（mouth, expr） <- 由音频特征直接驱动
      （音量 -> 嘴巴，音高变化率 + MFCC 能量 -> 表情）
  - 身体参数（head, body, arm, speed） <- 由风格权重驱动
      （对各风格预计算的 param_sequence 做加权插值）

[EN] What this connects to:
Pipeline (direction 2):
    audio_extractor.py --(OSC, 16 values)--> emotion_detector.py
        --> style_matcher.py --(style weights)--> THIS MODULE
    audio_extractor.py --(OSC, 16 values)--------------------------> THIS MODULE
    THIS MODULE --(6 values, OSC port 12000)--> MetaHuman / GUI

[中] 这个文件如何与其他文件连接：
数据流向（方向二）：
    audio_extractor.py --(OSC，16 维)--> emotion_detector.py
        --> style_matcher.py --(风格权重)--> 本模块
    audio_extractor.py --(OSC，16 维)-----------------------------> 本模块
    本模块 --(6 维，OSC 端口 12000)--> MetaHuman / GUI
"""

import numpy as np

# ---------------------------------------------------------------------------
# Output layout / 输出向量布局
# ---------------------------------------------------------------------------
# The 6 output params, in fixed order. Indices matter because both the body
# params (driven by style) and the face params (driven by audio) get written
# into specific slots of this same vector.
#
# 6 个输出参数的固定顺序。索引很重要，因为身体参数（风格驱动）和面部参数
# （音频驱动）会被写进同一个向量的不同位置。
#   0: mouth            嘴巴开合
#   1: expression       表情强度
#   2: head_angle       头部角度
#   3: body_amplitude   身体幅度
#   4: arm_height       手臂高度
#   5: movement_speed   动作速度

# Body param slots inside a param_sequence frame (which is itself 6-dim).
# We only read indices 2-5 from the style library; indices 0-1 (mouth/expr)
# are overwritten by audio-driven values at output time.
#
# 在 param_sequence 的每一帧（同样是 6 维）里，身体参数占用的索引位置。
# 我们只从风格库读取索引 2~5；索引 0~1（mouth/expr）在输出时会被音频驱动
# 的值覆盖掉。
BODY_PARAM_INDICES = [2, 3, 4, 5]  # head, body, arm, speed

# Audio feature layout (matches audio_extractor.py's 16-dim vector).
# 音频特征向量布局（与 audio_extractor.py 的 16 维向量一致）。
#   [0]      pitch_hz
#   [1]      volume (RMS)
#   [2..14]  mfcc_1 .. mfcc_13
#   [15]     tempo_bpm
PITCH_IDX = 0
VOLUME_IDX = 1
MFCC_START = 2   # mfcc_1
MFCC_END = 15    # exclusive upper bound -> mfcc_1..mfcc_13 = indices 2..14

# Normalization constants. These are rough empirical scales, not learned:
# audio_extractor.py outputs RMS volume in the ~0.0-0.3 range for normal
# speech/singing, so dividing by 0.3 maps "loud" to ~1.0. Pitch deltas of
# ~500 Hz across frames are "big", and MFCC std of ~10 is "expressive".
#
# 这些是经验性的归一化常数，不是学出来的：audio_extractor.py 输出的 RMS
# 音量在正常说话/唱歌时大约在 0.0~0.3 范围，所以除以 0.3 能把"大声"映射
# 到 ~1.0。帧间音高变化 ~500 Hz 算"很大"，MFCC 标准差 ~10 算"表现力强"。
VOLUME_SCALE = 0.3
PITCH_DELTA_SCALE = 500.0
MFCC_ENERGY_SCALE = 10.0

# A style only advances its playback pointer if its weight exceeds this
# threshold. Tiny weights would produce negligible motion but still cycle
# the pointer, wasting the effect of "resuming where you left off" when the
# style becomes active again.
#
# 只有权重超过这个阈值的风格才会推进回放帧指针。权重很小的风格产生的动作
# 几乎可以忽略，如果还推进指针，等它再次变活跃时就失去了"从上次位置接续"
# 的效果。
WEIGHT_ADVANCE_THRESHOLD = 0.01


class MotionGenerator:
    """Generates 6-dim output params from style weights + audio features.
    风格权重驱动身体参数，音频特征驱动面部参数。

    [EN] Usage:
        gen = MotionGenerator(style_library)
        out = gen.generate(style_weights, audio_features)  # one frame
        # out == [mouth, expr, head, body, arm, speed], all in [0, 1]

    [中] 用法：
        gen = MotionGenerator(style_library)
        out = gen.generate(style_weights, audio_features)  # 生成一帧
        # out == [mouth, expr, head, body, arm, speed]，均在 [0, 1]
    """

    def __init__(self, style_library):
        """[EN] Store the style library and init per-style playback state.
        [中] 保存风格库，并初始化每个风格的回放状态。

        Args:
            style_library: list of style dicts, each containing at least a
                "param_sequence": a list of [mouth, expr, head, body, arm,
                speed] frames (all values in [0, 1]).
                风格库列表，每个风格字典至少包含 "param_sequence"：
                一个 [mouth, expr, head, body, arm, speed] 帧列表（值均在 [0,1]）。
        """
        self.styles = style_library
        # Playback frame pointer per style: which frame of param_sequence
        # each style is currently "on". Multiple styles play independently,
        # each cycling through its own sequence.
        #
        # 每个风格的回放帧指针：当前停在 param_sequence 的第几帧。各风格
        # 独立回放，各自在自己的序列里循环。
        self.frame_indices = [0] * len(style_library)
        # Previous pitch value, used to compute the frame-to-frame pitch
        # change rate that drives the expression param.
        #
        # 上一帧的音高值，用于计算驱动表情参数的帧间音高变化率。
        self._prev_pitch = 0.0

    def generate(self, style_weights, audio_features):
        """Generate one frame of 6-dim output.
        生成一帧 6 维输出。

        [EN] Args:
            style_weights: list of floats summing to ~1.0 (from StyleMatcher).
                One weight per style, same order as the style library.
            audio_features: 16-dim list
                [pitch_hz, volume, mfcc_1..13, tempo_bpm] (from
                audio_extractor.py).

        [EN] Body params (head, body, arm, speed):
            - For each style i, read param_sequence[frame_indices[i]] and
              take the 4 body params at indices 2,3,4,5.
            - Weighted-average those 4 params across all styles.
            - Advance frame_indices[i] (with modulo wrap-around) for styles
              whose weight > 0.01.

        [EN] Face params (mouth, expr):
            - mouth: volume / VOLUME_SCALE, clipped to [0, 1].
            - expr: 0.5 * pitch_delta + 0.5 * mfcc_energy, where
                pitch_delta = |pitch_hz - prev_pitch| / PITCH_DELTA_SCALE
                mfcc_energy = std(mfcc_1..13) / MFCC_ENERGY_SCALE
              both clipped to [0, 1] before blending. prev_pitch is updated
              afterwards.

        [EN] Returns:
            [mouth, expr, head, body, arm, speed], all clipped to [0, 1].

        [中] 参数：
            style_weights：浮点数列表，和约为 1.0（来自 StyleMatcher）。
                每个风格一个权重，顺序与风格库一致。
            audio_features：16 维列表
                [pitch_hz, volume, mfcc_1..13, tempo_bpm]（来自
                audio_extractor.py）。

        [中] 身体参数（head, body, arm, speed）：
            - 对每个风格 i，读取 param_sequence[frame_indices[i]]，取索引
              2,3,4,5 处的 4 个身体参数。
            - 跨所有风格做加权平均。
            - 对权重大于 0.01 的风格，推进 frame_indices[i]（取模循环）。

        [中] 面部参数（mouth, expr）：
            - mouth：volume / VOLUME_SCALE，截断到 [0, 1]。
            - expr：0.5 * pitch_delta + 0.5 * mfcc_energy，其中
                pitch_delta = |pitch_hz - prev_pitch| / PITCH_DELTA_SCALE
                mfcc_energy = std(mfcc_1..13) / MFCC_ENERGY_SCALE
              两者先各自截断到 [0, 1] 再混合。之后更新 prev_pitch。

        [中] 返回：
            [mouth, expr, head, body, arm, speed]，均截断到 [0, 1]。
        """
        # ===================================================================
        # 1. Body params: weighted interpolation of style param_sequences.
        #    身体参数：对风格 param_sequence 做加权插值。
        # ===================================================================
        body_accum = np.zeros(len(BODY_PARAM_INDICES), dtype=np.float64)
        weight_sum = 0.0

        for i, style in enumerate(self.styles):
            w = style_weights[i] if i < len(style_weights) else 0.0
            if w <= 0.0:
                # Zero-weight style contributes nothing and doesn't advance.
                # 零权重风格不参与贡献，也不推进帧指针。
                continue

            seq = style.get("param_sequence", [])
            if not seq:
                # Style has no sequence data (shouldn't happen with a valid
                # library); skip it so we don't index into an empty list.
                #
                # 风格没有序列数据（正常的风格库不会这样）；跳过，避免对空
                # 列表取索引。
                continue

            frame_idx = self.frame_indices[i] % len(seq)
            frame = seq[frame_idx]
            # Read only the 4 body params (head, body, arm, speed) from
            # this frame. The mouth/expr slots in the sequence are ignored
            # here since audio drives those directly.
            #
            # 只取这一帧里的 4 个身体参数（head, body, arm, speed）。序列里
            # 的 mouth/expr 在这里忽略，因为它们由音频直接驱动。
            body_params = np.array(
                [frame[j] for j in BODY_PARAM_INDICES], dtype=np.float64
            )
            body_accum += w * body_params
            weight_sum += w

        if weight_sum > 0.0:
            body_out = body_accum / weight_sum
        else:
            # No active style (all weights ~0): hold a neutral pose.
            # 没有活跃风格（权重全为 ~0）：保持中性姿态。
            body_out = np.zeros(len(BODY_PARAM_INDICES), dtype=np.float64)

        # Advance playback pointers for styles with meaningful weight, so
        # each active style's sequence plays forward and loops. We do this
        # AFTER reading the current frame so all styles stay in sync within
        # a single generate() call.
        #
        # 推进有显著权重的风格的回放指针，让每个活跃风格的序列向前播放并
        # 循环。我们在读取当前帧之后才推进，这样同一次 generate() 调用内
        # 所有风格保持同步。
        for i, style in enumerate(self.styles):
            w = style_weights[i] if i < len(style_weights) else 0.0
            if w <= WEIGHT_ADVANCE_THRESHOLD:
                continue
            seq = style.get("param_sequence", [])
            if not seq:
                continue
            self.frame_indices[i] = (self.frame_indices[i] + 1) % len(seq)

        # ===================================================================
        # 2. Face params: driven directly by audio features.
        #    面部参数：由音频特征直接驱动。
        # ===================================================================
        pitch_hz = float(audio_features[PITCH_IDX])
        volume = float(audio_features[VOLUME_IDX])
        mfcc = np.array(
            audio_features[MFCC_START:MFCC_END], dtype=np.float64
        )

        # --- mouth: louder audio -> more open mouth.
        # --- 嘴巴：音量越大 -> 嘴巴张得越开。
        mouth = np.clip(volume / VOLUME_SCALE, 0.0, 1.0)

        # --- expr: blend of pitch change rate and MFCC spectral energy.
        #   Pitch delta captures melodic contour movement (big jumps = more
        #   expressive), MFCC std captures timbral variation (vibrato /
        #   tone shifts). Each half is normalized and clipped before the
        #   50/50 blend.
        #
        # --- 表情：音高变化率和 MFCC 频谱能量的混合。
        #   音高差捕捉旋律轮廓的运动（大跳 = 更有表现力），MFCC 标准差捕捉
        #   音色变化（颤音/音色切换）。两半各自归一化截断后做 50/50 混合。
        pitch_delta = abs(pitch_hz - self._prev_pitch) / PITCH_DELTA_SCALE
        pitch_delta = float(np.clip(pitch_delta, 0.0, 1.0))

        mfcc_energy = float(np.std(mfcc)) / MFCC_ENERGY_SCALE if len(mfcc) > 0 else 0.0
        mfcc_energy = float(np.clip(mfcc_energy, 0.0, 1.0))

        expr = 0.5 * pitch_delta + 0.5 * mfcc_energy

        # Update the previous pitch for next frame's delta computation.
        # 更新上一帧音高，供下一帧计算音高差使用。
        self._prev_pitch = pitch_hz

        # ===================================================================
        # 3. Assemble and clip the final 6-dim output.
        #    组装并截断最终的 6 维输出。
        # ===================================================================
        # body_out is [head, body, arm, speed] -> slots 2,3,4,5.
        # body_out 是 [head, body, arm, speed] -> 对应槽位 2,3,4,5。
        output = [
            float(mouth),
            float(expr),
            float(body_out[0]),  # head_angle
            float(body_out[1]),  # body_amplitude
            float(body_out[2]),  # arm_height
            float(body_out[3]),  # movement_speed
        ]
        return [float(np.clip(v, 0.0, 1.0)) for v in output]

    def reset(self):
        """Reset all frame indices to 0.
        将所有帧指针重置为 0。

        [EN] Call this when starting a fresh performance session so every
        style's param_sequence resumes from its first frame. The previous
        pitch tracker state is also reset.
        [中] 在开始一场新的表演时调用，让每个风格的 param_sequence 都从
        第一帧重新开始。同时重置音高跟踪状态。
        """
        self.frame_indices = [0] * len(self.styles)
        self._prev_pitch = 0.0
