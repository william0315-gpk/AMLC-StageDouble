# StageDouble 方向二：动作风格库 + 声音情绪匹配

## 方案概述

将当前的"视频实时输入 + 人工滑块标注"改为"预训练动作风格库 + 声音情绪驱动"。核心思路：

- **离线阶段**：从多个舞蹈视频中提取动作风格特征，建立风格库（每个风格在情绪空间中有一个位置）
- **运行阶段**：实时音频 -> 情绪检测 -> 在风格库中匹配/插值 -> 生成 6 维输出参数 -> MetaHuman
- **因果关系**：声音情绪驱动动作风格选择，音频特征直接驱动面部参数，动作风格驱动身体参数

## 数据流架构

```
┌───────────── 离线阶段（一次性）─────────────┐
│                                              │
│  data/videos/*.mp4                           │
│       │                                      │
│       ▼                                      │
│  style_extractor.py                          │
│  (复用 motion_extractor 的 MediaPipe 提取)    │
│       │                                      │
│       ▼                                      │
│  data/styles/style_library.pkl               │
│  [{name, emotion_label, emotion_pos,         │
│    param_sequence, style_features}, ...]     │
│                                              │
└──────────────────────────────────────────────┘

┌───────────── 在线阶段（实时表演）─────────────┐
│                                              │
│  麦克风 -> audio_extractor.py                │
│              │ 16维音频特征 (OSC 6448)        │
│              ▼                               │
│           emotion_detector.py                │
│              │ 情绪向量 [arousal, valence]    │
│              ▼                               │
│           style_matcher.py                   │
│              │ 风格权重 [w1, w2, ...]         │
│              ▼                               │
│           motion_generator.py                │
│              │ 6维输出 [mouth,expr,head,      │
│              │         body,arm,speed]       │
│              ▼                               │
│           OSC 12000 -> MetaHuman / GUI       │
│                                              │
│  因果分工：                                    │
│  · mouth, expr  <- 音频特征直接驱动（音量/音高）│
│  · head, body, arm, speed <- 动作风格驱动      │
│                                              │
└──────────────────────────────────────────────┘
```

## 文件夹结构

```
Pengkai-Gao-AMLC-Final-Project/
├── src/
│   ├── audio_extractor.py      # [保留] 实时音频特征提取，不变
│   ├── motion_extractor.py     # [保留] 视频姿态提取，style_extractor 复用其逻辑
│   ├── ml_trainer_v2.py        # [保留] 核心类（StreamReceiver 等）供复用
│   ├── app.py                  # [修改] 新增情绪显示、风格库面板、潜空间画布
│   ├── dashboard.py            # [保留] 输出柱状图，不变
│   ├── osc_listener.py         # [保留] 调试工具，不变
│   ├── style_extractor.py      # [新增] 离线：从视频提取动作风格，构建风格库
│   ├── emotion_detector.py     # [新增] 运行时：音频特征 -> 情绪向量
│   ├── style_matcher.py        # [新增] 运行时：情绪向量 -> 风格权重（潜空间导航）
│   └── motion_generator.py     # [新增] 运行时：风格权重 + 音频 -> 6维输出
├── data/
│   ├── videos/                 # [新增] 用户放入舞蹈视频（mp4），用户稍后提供
│   │   └── .gitkeep
│   └── styles/                 # [新增] style_extractor 生成的风格库
│       └── .gitkeep
├── ue/
│   ├── metahuman_osc.py        # [保留] UE OSC 接收，不变
│   └── setup_guide.md          # [保留] UE 配置指南，不变
├── models/                     # [保留] MediaPipe 模型缓存（.gitignore 忽略）
├── start.py                    # [修改] 新启动流程：只需 audio_extractor + app
├── requirements.txt            # [修改] 可能新增依赖
├── .gitignore                  # [保留]
└── 项目说明.md                  # [修改] 更新操作说明
```

## 各模块详细设计

### 1. `src/style_extractor.py`（新增 - 离线风格库构建）

**职责**：读取 `data/videos/` 下的所有视频，对每个视频提取姿态序列并计算风格特征，输出 `data/styles/style_library.pkl`。

**核心函数**：
```python
def extract_style_from_video(video_path, model_path) -> dict:
    """对单个视频提取风格特征。
    复用 motion_extractor.py 的 MediaPipe PoseLandmarker 逻辑，
    逐帧提取 99 维姿态，然后计算：
    - param_sequence: 从姿态序列推导的 6 维参数时间序列
    - style_features: 统计特征（avg_speed, avg_energy, spatial_extent, rhythm）
    返回: {name, pose_sequence, param_sequence, style_features}
    """

def build_style_library(videos_dir, output_path, emotion_labels=None):
    """遍历 videos_dir 下所有 mp4，提取风格，保存为 pkl。
    emotion_labels: 可选的手动标签 dict {filename: [arousal, valence]}
    如果不提供，用 style_features 自动聚类分配位置。
    """
```

**姿态 -> 6 维参数的映射规则**（自动推导，非人工标注）：
- `head`：头部关键点(landmark 0)的 y 坐标相对于身体中心的偏移，归一化到 [0, 1]
- `body`：躯干中心(landmark 11,12,23,24 的平均) 的帧间位移幅度
- `arm`：手腕(landmark 15,16) 相对于肩膀(landmark 11,12) 的高度差
- `speed`：所有关键点的帧间欧氏距离均值
- `mouth` / `expr`：**不从姿态推导**，运行时由音频直接驱动

**emotion_position 的确定**：
- 如果用户提供了情绪标签（如 "excited" -> [0.9, 0.8]），直接使用
- 如果没有标签，用 style_features 中的 avg_speed 和 avg_energy 映射到 2D 情绪空间
  - arousal = avg_speed（动作越快 -> 唤醒度越高）
  - valence = spatial_extent - avg_energy（空间幅度大但能量适中 -> 正向）

### 2. `src/emotion_detector.py`（新增 - 运行时情绪检测）

**职责**：接收 16 维音频特征，输出 2 维情绪向量 `[arousal, valence]`（均在 [0, 1] 范围）。

**核心类**：
```python
class EmotionDetector:
    def __init__(self):
        # 规则式映射的初始参数（可被 IML 训练替代）
        self.arousal_smoothing = 0.8  # 平滑系数
        self._arousal = 0.5
        self._valence = 0.5

    def detect(self, audio_features) -> list[float]:
        """从 16 维音频特征计算情绪向量。
        audio_features = [pitch_hz, volume, mfcc_1..13, tempo_bpm]

        arousal（唤醒度/能量）:
        - 主要由 volume (RMS) 和 tempo_bpm 驱动
        - volume 越大 -> arousal 越高
        - tempo 越快 -> arousal 越高

        valence（效价/情绪正负）:
        - 主要由 pitch_hz 和 MFCC 变化驱动
        - pitch 越高 -> valence 越高（欢快）
        - pitch 越低 -> valence 越低（低沉）
        - MFCC 变化大 -> 情绪波动大

        返回 [arousal, valence]，经过指数平滑避免抖动。
        """

    def train(self, X, y):
        """IML 训练：用户标注音频片段的情绪，训练一个回归器替代规则。
        X: 音频特征列表, y: 对应的 [arousal, valence] 标签
        """
```

**初始实现用规则映射**（无需训练数据即可工作），IML 训练为可选增强。

### 3. `src/style_matcher.py`（新增 - 潜空间导航）

**职责**：根据情绪向量在风格库中计算各风格的权重（软分配）。

**核心类**：
```python
class StyleMatcher:
    def __init__(self, style_library):
        self.styles = style_library  # 加载的风格库
        self.positions = np.array([s["emotion_pos"] for s in self.styles])

    def match(self, emotion_vector) -> list[float]:
        """计算情绪向量到各风格点的权重。
        使用 RBF（径向基函数）/ softmax 距离计算：
        weights[i] = exp(-dist(emotion, style_i)^2 / (2*sigma^2))
        归一化后返回权重列表（和为 1.0）。
        sigma 控制平滑度：大 sigma -> 更平滑的插值，小 sigma -> 更锐利的选择。
        """

    def get_active_styles(self, emotion_vector, threshold=0.1):
        """返回权重大于 threshold 的风格，用于 GUI 高亮显示。"""
```

### 4. `src/motion_generator.py`（新增 - 动作生成）

**职责**：根据风格权重和实时音频特征，生成 6 维输出参数。

**核心类**：
```python
class MotionGenerator:
    def __init__(self, style_library):
        self.styles = style_library
        self.frame_indices = [0] * len(self.styles)  # 各风格的回放帧指针

    def generate(self, style_weights, audio_features) -> list[float]:
        """生成 6 维输出参数。

        身体参数（head, body, arm, speed）：
        - 每个风格有一个预计算的 param_sequence（6维时间序列）
        - 按各自帧指针取当前帧的值
        - 加权平均得到合成参数
        - 推进帧指针（速度由 style_weights 中高权重风格的 rhythm 决定）

        面部参数（mouth, expr）：
        - mouth = 音频 volume 的实时映射（归一化）
        - expr = 音频 pitch 变化率 + MFCC 能量的组合映射

        返回 [mouth, expr, head, body, arm, speed]，均 clip 到 [0, 1]。
        """
```

**关键设计**：面部参数完全由音频驱动（因果关系直接），身体参数由风格库驱动（通过情绪间接关联），两者在输出层面自然融合。

### 5. `src/app.py`（修改 - 新增 GUI 面板）

在现有 GUI 基础上新增：

**左栏新增（在现有录制参数下方）**：
- "风格库" 面板：
  - "构建风格库" 按钮（调用 style_extractor 处理 data/videos/）
  - 风格列表（显示已加载的风格名称和情绪标签）
  - "加载风格库" 按钮
- "情绪检测" 面板：
  - 实时 arousal/valence 数值和进度条
  - IML 标注控件（录制音频片段 + 标注情绪 -> 训练 emotion_detector）

**右栏新增（在现有柱状图上方）**：
- "潜空间地图" Canvas（200x200）：
  - 各风格作为彩色圆点绘制在 2D 情绪空间中
  - 当前情绪位置作为高亮十字标记
  - 实时移动，可视化"导航"过程
  - 可选：用户可拖拽当前位置手动探索

**底部状态栏新增**：
- 风格库状态（已加载 X 个风格）
- 当前主导风格名称

**运行流程变化**：
- 旧流程：录制 -> 训练 -> 运行（需要 motion_extractor 实时运行）
- 新流程：构建风格库 -> 运行（只需 audio_extractor 实时运行）
- IML 增强：可选地标注音频情绪 -> 训练 emotion_detector

### 6. `start.py`（修改 - 新启动流程）

- 去掉 motion_extractor.py 的启动（性能阶段不再需要实时视频输入）
- 启动流程变为：audio_extractor.py -> app.py（2 个进程）
- 新增"构建风格库"入口：可选择 data/videos/ 下的视频运行 style_extractor

## 实现步骤

### Step 1: 创建文件夹结构
- 创建 `data/videos/.gitkeep` 和 `data/styles/.gitkeep`

### Step 2: 实现 `src/style_extractor.py`
- 复用 motion_extractor.py 的 PoseLandmarker 初始化和帧提取逻辑
- 实现 `extract_style_from_video()`：逐帧提取姿态 -> 计算 6 维参数序列 -> 计算风格统计特征
- 实现 `build_style_library()`：遍历视频目录，汇总保存为 pkl
- 姿态 -> 参数映射函数 `poses_to_params()`

### Step 3: 实现 `src/emotion_detector.py`
- 规则式 `detect()`：volume -> arousal, pitch -> valence，指数平滑
- `train()` 接口预留（IML 训练用 RandomForest 或 MLP）

### Step 4: 实现 `src/style_matcher.py`
- 加载风格库 pkl
- RBF 距离计算 + softmax 归一化
- `match()` 和 `get_active_styles()`

### Step 5: 实现 `src/motion_generator.py`
- 加载风格库 pkl
- `generate()`：风格加权插值身体参数 + 音频驱动面部参数
- 帧指针管理（循环回放各风格的参数序列）

### Step 6: 修改 `src/app.py`
- 新增风格库面板、情绪显示、潜空间 Canvas
- 修改运行逻辑：用 emotion_detector -> style_matcher -> motion_generator 替代直接 ML 预测
- 保留原有 IML 录制/训练/运行作为"经典模式"（可切换）

### Step 7: 修改 `start.py`
- 简化启动流程（去掉 motion_extractor）
- 新增"构建风格库"按钮

### Step 8: 更新 `requirements.txt` 和 `项目说明.md`
- 如有新依赖则添加
- 更新操作说明

## 假设与决策

1. **情绪空间为 2D**（arousal × valence），这是情感计算领域最常用的模型（Russell circumplex model），论文中容易引用。
2. **初始情绪检测用规则映射**，无需训练数据即可工作。IML 训练为可选增强，用户后续可标注数据提升精度。
3. **面部参数由音频直接驱动**，身体参数由风格库驱动。这种分工建立了清晰的因果关系，且在论文中可明确论述。
4. **风格库的 param_sequence 是预计算的**，运行时只需加权插值，不需要实时 MediaPipe 推理，性能开销低。
5. **保留原有 IML 模式**作为"经典模式"，用户可在 GUI 中切换"风格库模式"和"经典模式"。
6. **用户稍后提供视频**，框架先搭好，视频放入 `data/videos/` 即可使用。

## 验证步骤

1. 无视频时：`app.py` 能启动，显示空风格库，音频采集正常，情绪检测正常
2. 放入视频后：点"构建风格库"，`style_library.pkl` 生成，GUI 显示风格列表和潜空间地图
3. 运行模式：音频输入 -> 情绪变化 -> 潜空间位置移动 -> 风格权重变化 -> 6 维输出变化 -> 2D 小人动作变化
4. OSC 输出：6 维参数通过端口 12000 发出，dashboard.py 和 osc_listener.py 能正常接收
5. IML 增强：标注音频情绪 -> 训练 emotion_detector -> 情绪检测精度提升
