"""
StageDouble - Emotion Detector
StageDouble - 情绪检测器

[EN] What this file does:
Maps the 16-dimensional audio feature vector from audio_extractor.py to a
2-dimensional emotion space [arousal, valence], both in [0, 1]. Arousal
represents energy / intensity; valence represents emotional positivity. The
default mapping is rule-based and needs no training data. It can optionally
be enhanced via Interactive Machine Learning (IML) by calling train() with
labeled examples.

[中] 这个文件做什么：
把 audio_extractor.py 输出的 16 维音频特征向量映射到一个 2 维情绪空间
[arousal, valence]（唤醒度、效价），两者都在 [0, 1] 范围内。arousal
代表能量/激烈程度；valence 代表情绪的正负倾向。默认使用规则式映射，
不需要任何训练数据即可工作；也可以通过调用 train() 传入带标签的样本，
用交互式机器学习（IML）来增强映射效果。

[EN] Feature vector layout (16 values, produced by audio_extractor.py):
    [0]     pitch_hz     - fundamental frequency (Hz), 0.0 when silent
    [1]     volume       - RMS amplitude (typically 0.0-0.5)
    [2:15]  mfcc_1..13   - 13 MFCC coefficient means
    [15]    tempo_bpm    - beats per minute (typically 60-200)

[中] 特征向量布局（16 个数值，由 audio_extractor.py 产生）：
    [0]     pitch_hz     - 基频（Hz），静音时为 0.0
    [1]     volume       - RMS 振幅（通常 0.0-0.5）
    [2:15]  mfcc_1..13   - 13 个 MFCC 系数的均值
    [15]    tempo_bpm    - 节拍速度（BPM，通常 60-200）

[EN] What this connects to:
Sits between audio_extractor.py and the digital-human rendering layer.
audio_extractor.py sends 16 values over OSC; this module turns them into
[arousal, valence] which downstream code can use to drive facial-expression
blendshapes, color grading, or any emotion-driven visual parameter.

[中] 这个文件如何与其他文件连接：
位于 audio_extractor.py 和数字人渲染层之间。audio_extractor.py 通过 OSC
发送 16 个数值，本模块将其转换为 [arousal, valence]，下游代码可以用它来
驱动面部表情混合变形（blendshapes）、色彩调色或任何由情绪驱动的视觉参数。
"""

import numpy as np


class EmotionDetector:
    """Detects emotion [arousal, valence] from 16-dim audio features.
    规则式映射初始版本，可通过 IML 训练增强。"""

    # --- Rule-based normalization constants / 规则式映射的归一化常数 ---
    # These are rough heuristics, not precise calibrations. They map the raw
    # audio feature ranges into [0, 1] using divide-then-clip. Tune them if
    # your particular mic / vocalist produces values in a different range.
    #
    # 这些是粗略的经验值，不是精确校准。用"先除后截断"的方式把原始音频
    # 特征的取值范围映射到 [0, 1]。如果你的麦克风/歌手产生的数值范围不同，
    # 可以调整这些常数。
    VOLUME_SCALE = 0.3    # RMS amplitude that counts as "full volume" / 算作"满音量"的 RMS 振幅
    TEMPO_MIN = 60.0      # BPM at the low end of the tempo range / 节奏归一化范围下限的 BPM
    TEMPO_RANGE = 140.0   # BPM span (60-200) for tempo normalization / 节奏归一化的 BPM 跨度（60-200）
    PITCH_MIN = 100.0     # Hz below which pitch is "low/neutral" / 低于此值的音高视为"低沉/中性"
    PITCH_RANGE = 500.0   # Hz span (100-600) for pitch normalization / 音高归一化的 Hz 跨度（100-600）
    MFCC_STD_SCALE = 50.0 # MFCC std that counts as "highly emotional" / 算作"高情绪"的 MFCC 标准差

    def __init__(self):
        self.smoothing = 0.85  # exponential smoothing factor / 指数平滑系数
        self._arousal = 0.5
        self._valence = 0.5
        self._trained = False
        self._model = None  # IML-trained model (None = use rules) / IML 训练的模型（None = 使用规则）

    def detect(self, audio_features):
        """Map 16-dim audio features to [arousal, valence], both in [0,1].

        [EN] If self._trained, use the trained model. Otherwise use the
        rule-based mapping:
        - arousal: 0.6 * volume_norm + 0.4 * tempo_norm
        - valence: 0.6 * pitch_norm + 0.4 * mfcc_energy_norm
        Exponential smoothing is applied to both outputs before returning.

        [中] 如果已训练（self._trained），则使用训练好的模型预测。
        否则使用规则式映射：
        - arousal（唤醒度）：0.6 * 音量归一化 + 0.4 * 节奏归一化
        - valence（效价）：0.6 * 音高归一化 + 0.4 * MFCC 能量归一化
        两个输出在返回前都会经过指数平滑处理。
        """
        if self._trained and self._model is not None:
            # --- Trained model path / 已训练模型路径 ---
            out = self._model.predict([audio_features])[0]
            new_arousal = float(np.clip(out[0], 0.0, 1.0))
            new_valence = float(np.clip(out[1], 0.0, 1.0))
        else:
            new_arousal, new_valence = self._rule_based_detect(audio_features)

        # --- Exponential smoothing: new = smoothing * old + (1 - smoothing) * new
        # --- 指数平滑：新值 = 平滑系数 * 旧值 + (1 - 平滑系数) * 新值
        # 让输出不会因为单帧特征的突变而剧烈跳变，保持情绪曲线的连续性。
        self._arousal = self.smoothing * self._arousal + (1 - self.smoothing) * new_arousal
        self._valence = self.smoothing * self._valence + (1 - self.smoothing) * new_valence

        return [self._arousal, self._valence]

    def _rule_based_detect(self, audio_features):
        """[EN] Rule-based mapping from 16-dim features to [arousal, valence].
        [中] 规则式映射：把 16 维特征映射到 [arousal, valence]。
        """
        pitch_hz = audio_features[0]
        volume = audio_features[1]
        mfcc = audio_features[2:15]  # mfcc_1 .. mfcc_13 (13 values) / mfcc_1 到 mfcc_13（共 13 个值）
        tempo_bpm = audio_features[15]

        # --- Arousal (唤醒度/能量): loud + fast = high arousal ---
        # --- 响亮 + 快节奏 = 高唤醒度 ---
        volume_norm = float(np.clip(volume / self.VOLUME_SCALE, 0.0, 1.0))
        tempo_norm = float(np.clip((tempo_bpm - self.TEMPO_MIN) / self.TEMPO_RANGE, 0.0, 1.0))
        arousal = 0.6 * volume_norm + 0.4 * tempo_norm

        # --- Valence (效价/情绪正负): high pitch + varied timbre = positive/excited ---
        # --- 高音高 + 丰富音色变化 = 正面/兴奋 ---
        if pitch_hz <= 0.0:
            # Silent: no pitch information, so keep the previous valence.
            # (After smoothing this is a no-op: new == old -> output unchanged.)
            # 静音时没有音高信息，保持上一次的 valence。
            # （经过指数平滑后这是一个空操作：新值 == 旧值 -> 输出不变。）
            valence = self._valence
        else:
            pitch_norm = float(np.clip((pitch_hz - self.PITCH_MIN) / self.PITCH_RANGE, 0.0, 1.0))
            # MFCC energy: high std across the 13 coefficients means more spectral
            # variation, which we interpret as greater emotional expressiveness.
            # MFCC 能量：13 个系数的标准差越大，说明频谱变化越丰富，
            # 我们将其解读为情绪表现力越强。
            mfcc_std = float(np.std(mfcc))
            mfcc_energy_norm = float(np.clip(mfcc_std / self.MFCC_STD_SCALE, 0.0, 1.0))
            valence = 0.6 * pitch_norm + 0.4 * mfcc_energy_norm

        return arousal, valence

    def train(self, X, y):
        """IML training: X is list of 16-dim audio features, y is list of [arousal, valence].
        Use RandomForestRegressor from sklearn. Set self._trained=True.

        [EN] Train a RandomForestRegressor on labeled examples. After training,
        detect() will use the model instead of the rule-based mapping. sklearn
        is imported lazily here so that the rule-based path works without it.

        [中] 用带标签的样本训练一个 RandomForestRegressor（随机森林回归器）。
        训练完成后，detect() 将使用模型预测而非规则式映射。sklearn 在此处
        延迟导入，这样规则式路径不需要安装 sklearn 也能正常工作。
        """
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(n_estimators=50, random_state=42)
        self._model.fit(X, y)
        self._trained = True

    @property
    def trained(self):
        """[中] 返回模型是否已经训练过。"""
        return self._trained
