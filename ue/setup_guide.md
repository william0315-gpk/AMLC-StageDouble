# StageDouble -- Unreal Engine MetaHuman 对接指南

## 一、你需要安装什么

| 软件 | 大小 | 用途 | 下载方式 |
|------|------|------|---------|
| Epic Games Launcher | ~200MB | 下载 UE 的工具 | epicgames.com |
| Unreal Engine 5.3+ | ~50GB | 3D 渲染引擎 | 通过 Launcher 下载 |
| MetaHuman | 免费 | 生成逼真数字人 | Launcher 内安装 |

## 二、安装步骤

### 2.1 安装 Epic Games Launcher
1. 打开 https://www.epicgames.com/download 下载并安装
2. 注册/登录账号

### 2.2 安装 Unreal Engine
1. 在 Launcher 左侧点"Unreal Engine"
2. 点"Library" -> "+"号 -> 选 5.3 或更新版本
3. 等待下载完成（约 50GB，需要较长时间）

### 2.3 安装 MetaHuman 插件
1. Launcher 中点"Marketplace" -> 搜索 "MetaHuman"
2. 点"Add to Cart" -> "Checkout"（免费的）
3. 在 UE 中创建新项目（Games -> Blank -> 创建）

### 2.4 在 UE 项目中启用插件
1. 打开 UE 项目
2. 菜单：Edit -> Plugins
3. 搜索并启用以下三个插件：
   - **MetaHuman** （数字人支持）
   - **Python Editor Script Plugin** （Python 脚本支持）
   - **OSC** （OSC 协议支持）
4. 重启 UE

### 2.5 创建 MetaHuman 角色
1. 菜单：Tools -> Quixel Bridge
2. 左侧选"MetaHumans" -> "Ready"
3. 选一个角色 -> 点"Download"
4. 下载完成后点"Add to Project"
5. 角色会出现在 Content Browser 里

### 2.6 把 MetaHuman 放到场景里
1. 从 Content Browser 拖拽 MetaHuman 蓝图到场景中
2. 确保角色面朝摄像机

## 三、导入 StageDouble 脚本

### 3.1 放置脚本文件
1. 在 UE 项目目录下找到 `Content/Python/` 文件夹（没有就新建）
2. 把 `ue/metahuman_osc.py` 复制到这个文件夹里
3. 或者直接在 UE 里打开 Python 控制台，复制脚本内容粘贴运行

### 3.2 启动 OSC 接收
1. 菜单：Tools -> Python
2. 在 Python 控制台输入：
   ```python
   import metahuman_osc
   start_stagedouble()
   ```
3. 看到以下输出表示成功：
   ```
   [StageDouble] Found MetaHuman: 你的角色名
   [StageDouble] Found SkeletalMeshComponent: ...
   [StageDouble] OSC receiver started on port 12000
   [StageDouble] Started. Receiving OSC on port 12000.
   ```

### 3.3 测试连接
在 Python 控制台输入：
```python
test_connection()
```
MetaHuman 应该会有轻微的动作变化（嘴巴张开、头部微动）。

## 四、参数到 MetaHuman 的映射

| OSC 收到的值 | 范围 | 映射到 MetaHuman | 效果 |
|-------------|------|-----------------|------|
| 第 1 个 (mouth) | 0~1 | 下颌骨骼 Z 轴旋转 | 嘴巴从闭合到张大 |
| 第 2 个 (express) | 0~1 | 面部 Pose Asset 权重 | 表情从平淡到夸张 |
| 第 3 个 (head) | 0~1 | 头部骨骼 Pitch 旋转 | 低头(-0.3)到抬头(+0.3) |
| 第 4 个 (body) | 0~1 | 脊柱骨骼缩放 | 身体轻微缩放表示动作幅度 |
| 第 5 个 (arm) | 0~1 | 双侧上臂骨骼旋转 | 手臂从下垂到举起 |
| 第 6 个 (speed) | 0~1 | 动画播放速率 | 0.5x 慢到 2.0x 快 |

## 五、调整骨骼名称

如果你的 MetaHuman 角色骨骼名称和脚本里写的不一样（比如命名规则不同），需要修改 `metahuman_osc.py` 顶部的配置：

```python
BONE_HEAD = "head"          # 头部骨骼名
BONE_JAW = "jaw"            # 下颌骨骼名
BONE_SPINE_01 = "spine_01"  # 脊柱骨骼名
BONE_UPPERARM_L = "upperarm_l"  # 左上臂骨骼名
BONE_UPPERARM_R = "upperarm_r"  # 右上臂骨骼名
```

查看骨骼名称的方法：
1. 双击 MetaHuman 蓝图打开
2. 切换到"骨骼"视图
3. 在骨骼树中找到对应骨骼的名称

## 六、完整运行流程

```
第 1 步：打开 UE 项目，把 MetaHuman 拖到场景里
第 2 步：在 UE Python 控制台运行 start_stagedouble()
第 3 步：打开终端 1，运行 python audio_extractor.py
第 4 步：打开终端 2，运行 python motion_extractor.py --video 你的视频.mp4
第 5 步：打开终端 3，运行 python app.py
第 6 步：在 app.py 窗口里录制、训练、运行
第 7 步：UE 里的 MetaHuman 开始跟着你的声音和视频动作
```

## 七、常见问题

| 问题 | 解决方法 |
|------|---------|
| Python 控制台找不到 | 确认 Python Editor Script Plugin 已启用并重启 UE |
| 找不到 MetaHuman | 确认 MetaHuman 插件已启用，角色已下载并添加到项目 |
| OSC 收不到数据 | 确认 app.py 或 ml_trainer_v2.py 已经点了"运行"且模型已训练 |
| 骨骼名不对 | 用蓝图编辑器查看实际骨骼名，修改脚本顶部的 BONE_* 变量 |
| 角色不动 | 确认 start_stagedouble() 已运行，Python 控制台有"OSC receiver started"输出 |
| 嘴巴动但手臂不动 | MetaHuman 的手臂骨骼可能需要通过 Control Rig 而非直接骨骼变换控制 |

## 八、关于 Control Rig（进阶）

脚本中直接修改骨骼变换的方式适用于简单测试。如果需要更精确的控制（特别是面部表情），建议使用 Control Rig：

1. 在 MetaHuman 蓝图中添加 Control Rig 节点
2. 用 OSC Dispatch 节点接收 6 个值
3. 分别连接到 Control Rig 的控制点（如 jaw_open、head_tilt 等）

这需要一定的 UE Blueprint 基础，可以参考 UE 官方的 MetaHuman 文档：
https://dev.epicgames.com/documentation/en-us/metahuman
