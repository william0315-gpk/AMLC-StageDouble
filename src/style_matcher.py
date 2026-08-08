"""
StageDouble - Latent Style Matcher
StageDouble - 风格潜在空间匹配器

[EN] What this file does:
Navigates the latent space of motion styles by matching an emotion vector
(arousal, valence) to entries in a style library. Instead of picking a single
nearest style, it computes a soft weight distribution over ALL styles using a
Gaussian Radial Basis Function (RBF) kernel, so that several nearby styles can
blend together. The smoothness of this blending is controlled by sigma: a large
sigma produces a smooth interpolation across many styles, while a small sigma
sharpens the match toward the single closest style.

[中] 这个文件做什么：
通过把情绪向量（arousal 唤醒度, valence 效价）与风格库中的条目进行匹配，
在动作风格的潜在空间中导航。它不是只选一个最近邻风格，而是用高斯径向基
函数（RBF）核在**所有**风格上计算一个软权重分布，让多个相近的风格可以
混合在一起。这种混合的平滑程度由 sigma 控制：sigma 越大，权重在多个风格
之间越平滑地插值；sigma 越小，匹配越尖锐，越倾向于只选最近的那一个风格。
"""

import numpy as np


class StyleMatcher:
    """Matches emotion vectors to motion styles in the latent space.
    在情绪空间中导航，通过 RBF 距离计算各风格的权重。"""

    def __init__(self, style_library, sigma=0.25):
        """
        [EN] Args:
            style_library: list of style dicts, each with "emotion_pos" key
                           (a [arousal, valence] pair, both in [0, 1]).
            sigma: controls smoothness of matching. Larger = smoother
                   interpolation, smaller = sharper selection. Default 0.25
                   is good for [0, 1] space.

        [中] 参数：
            style_library: 风格字典列表，每个字典含 "emotion_pos" 键
                           （即 [唤醒度, 效价]，取值均在 [0, 1]）。
            sigma: 控制匹配的平滑度。越大越平滑（多风格插值），越小越尖锐
                   （倾向单风格选择）。默认 0.25 适合 [0, 1] 取值空间。
        """
        self.styles = style_library
        self.sigma = sigma
        if style_library:
            self.positions = np.array([s["emotion_pos"] for s in style_library])
        else:
            self.positions = np.empty((0, 2))

    def match(self, emotion_vector):
        """Compute style weights using RBF (Gaussian) distance.
        用 RBF（高斯）距离计算各风格的权重。

        [EN] Formula:
            weights[i] = exp(-||emotion - style_i||^2 / (2*sigma^2))
            Then normalize so weights sum to 1.0.
            If style library is empty, return empty list.

        [中] 公式：
            weights[i] = exp(-||emotion - style_i||^2 / (2*sigma^2))
            随后归一化，使权重之和为 1.0。
            若风格库为空，返回空列表。

        Returns:
            list of floats summing to 1.0
            归一化后的权重列表，和为 1.0
        """
        # Empty library -> nothing to match.
        # 风格库为空 -> 无可匹配项。
        if self.positions.shape[0] == 0:
            return []

        emotion = np.asarray(emotion_vector, dtype=float)

        # Vectorized squared Euclidean distance to every style position.
        # 向量化计算到每个风格位置的欧氏距离平方。
        diff = self.positions - emotion          # (N, 2)
        sq_dist = np.sum(diff ** 2, axis=1)      # (N,)

        # Gaussian RBF kernel -> all weights strictly > 0.
        # 高斯 RBF 核 -> 所有权重均严格大于 0。
        weights = np.exp(-sq_dist / (2.0 * self.sigma ** 2))

        # Normalize to a probability distribution.
        # 归一化为概率分布。
        total = weights.sum()
        if total == 0.0:
            # Degenerate case (e.g. sigma == 0): fall back to uniform.
            # 退化情形（如 sigma == 0）：退化为均匀分布。
            weights = np.full(self.positions.shape[0], 1.0 / self.positions.shape[0])
        else:
            weights = weights / total

        return weights.tolist()

    def get_active_styles(self, emotion_vector, threshold=0.1):
        """Return list of (index, weight, style_dict) for weights > threshold.
        返回权重 > threshold 的 (索引, 权重, 风格字典) 列表。

        [EN] Useful for GUI highlighting. Sorted by weight descending.

        [中] 适用于 GUI 高亮显示。按权重从大到小排序。

        Args:
            threshold: minimum weight for a style to be considered "active".
                       threshold: 风格被视为"激活"的最小权重。

        Returns:
            list of (index, weight, style_dict) tuples, sorted by weight desc.
            (索引, 权重, 风格字典) 元组列表，按权重降序排列。
        """
        weights = self.match(emotion_vector)

        # Collect active styles above the threshold.
        # 收集权重超过阈值的激活风格。
        active = [
            (i, w, self.styles[i])
            for i, w in enumerate(weights)
            if w > threshold
        ]

        # Sort by weight, heaviest first.
        # 按权重排序，最大者在前。
        active.sort(key=lambda x: x[1], reverse=True)
        return active

    def set_sigma(self, sigma):
        """Adjust sigma at runtime (e.g., via GUI slider).
        运行时调整 sigma（例如通过 GUI 滑块）。"""
        self.sigma = sigma
