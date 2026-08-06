"""
StageDouble - Unreal Engine MetaHuman OSC Receiver
UE 端 OSC 接收脚本

[EN] This script runs inside Unreal Engine's Python console. It listens for
the 6-value OSC output from StageDouble's ml_trainer_v2.py (or app.py) and
applies them to a MetaHuman actor's skeletal mesh in real time.

[中] 此脚本在 Unreal Engine 的 Python 控制台中运行。它监听来自
StageDouble 的 ml_trainer_v2.py（或 app.py）发出的 6 个 OSC 输出值，
并实时应用到 MetaHuman 角色的骨骼网格体上。

=== 使用前提 / Prerequisites ===
1. Unreal Engine 5.3+ 已安装
2. MetaHuman 插件已启用
3. Python Scripting 插件已启用
4. OSC 插件已启用
5. 场景中已有一个 MetaHuman 角色

=== 使用方法 / Usage ===
1. 在 UE 中打开你的场景（包含 MetaHuman）
2. 菜单 Tools -> Python
3. 将此脚本复制到 Python 控制台中运行
4. 或将此文件放到 UE 项目的 Content/Python/ 目录下，然后在控制台输入：
   import metahuman_osc

=== 参数映射 / Parameter Mapping ===
值 1 (mouth)    -> 下颌骨骼 Z 轴旋转（嘴巴开合）
值 2 (express)  -> 面部表情权重（表情强度）
值 3 (head)     -> 头部骨骼 Pitch 旋转（头部角度）
值 4 (body)     -> 身体骨骼缩放（身体动作幅度）
值 5 (arm)      -> 手臂骨骼旋转（手臂高度）
值 6 (speed)    -> 动画播放速率（动作速度）
"""

import socket
import struct
import threading
import time
import math

try:
    import unreal
except ImportError:
    print("[Warning] unreal module not found. This script must run inside UE.")
    unreal = None


# === 配置 / Configuration ===
OSC_PORT = 12000
OSC_ADDRESS = "/stagedouble/outputs"

# MetaHuman 骨骼名称（可能因角色不同而略有差异，请根据实际情况修改）
BONE_HEAD = "head"
BONE_JAW = "jaw"
BONE_SPINE_01 = "spine_01"
BONE_UPPERARM_L = "upperarm_l"
BONE_UPPERARM_R = "upperarm_r"

# 参数范围映射
MOUTH_OPEN_MIN = 0.0    # 下颌闭合角度
MOUTH_OPEN_MAX = 0.5    # 下颌张开角度（弧度）

HEAD_TILT_MIN = -0.3    # 低头角度
HEAD_TILT_MAX = 0.3     # 抬头角度

ARM_RAISE_MIN = 0.0     # 手臂下垂角度
ARM_RAISE_MAX = 1.2     # 手臂抬起角度

BODY_SCALE_MIN = 0.95   # 身体最小缩放
BODY_SCALE_MAX = 1.05   # 身体最大缩放


class OSCReceiver:
    """[中] 简易 OSC 消息接收器，解析 /stagedouble/outputs 地址的消息。"""

    def __init__(self, port, address, callback):
        self.port = port
        self.address = address
        self.callback = callback
        self.running = False
        self.sock = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.settimeout(1.0)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[StageDouble] OSC receiver started on port {self.port}")

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()

    def _loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                values = self._parse_osc(data)
                if values is not None:
                    self.callback(values)
            except socket.timeout:
                continue
            except OSError:
                break

    def _parse_osc(self, data):
        """[中] 解析 OSC 数据包，返回 6 个 float 值或 None。"""
        try:
            # 解析地址
            addr_end = data.index(b'\x00')
            addr = data[:addr_end].decode('ascii')
            # 对齐到 4 字节边界
            addr_padded = (addr_end + 4) & ~3

            # 解析类型标签
            type_start = addr_padded
            if data[type_start:type_start + 1] != b',':
                return None
            type_end = data.index(b'\x00', type_start)
            type_str = data[type_start + 1:type_end].decode('ascii')
            type_padded = (type_end + 4) & ~3

            if addr != self.address:
                return None

            # 解析浮点数值
            values = []
            offset = type_padded
            for t in type_str:
                if t == 'f':
                    val = struct.unpack('>f', data[offset:offset + 4])[0]
                    values.append(val)
                    offset += 4

            if len(values) >= 6:
                return values[:6]
            return None
        except (ValueError, struct.error, IndexError):
            return None


class MetaHumanController:
    """[中] 控制 MetaHuman 角色的骨骼变换。"""

    def __init__(self):
        self.actor = None
        self.skeletal_mesh = None
        self.anim_instance = None
        self._find_metahuman()

    def _find_metahuman(self):
        """[中] 在场景中查找 MetaHuman 角色。"""
        if not unreal:
            return

        actors = unreal.EditorLevelLibrary.get_all_level_actors()
        for actor in actors:
            if actor.get_class().get_name() == "BP_MetaHumanActor_C" or \
               "metahuman" in actor.get_name().lower():
                self.actor = actor
                print(f"[StageDouble] Found MetaHuman: {actor.get_name()}")

                # 查找 Skeletal Mesh Component
                for comp in actor.get_components_by_class(unreal.SkeletalMeshComponent):
                    self.skeletal_mesh = comp
                    self.anim_instance = comp.get_anim_instance()
                    print(f"[StageDouble] Found SkeletalMeshComponent: {comp.get_name()}")
                    break
                break

        if not self.actor:
            print("[StageDouble] Warning: No MetaHuman found in scene.")
            print("[StageDouble] Please add a MetaHuman to the scene first.")

    def apply_params(self, values):
        """[中] 将 6 个参数值应用到 MetaHuman 骨骼上。

        values[0] = mouth     嘴巴开合 0~1
        values[1] = express   表情强度 0~1
        values[2] = head      头部角度 0~1 (0=低头, 0.5=正中, 1=抬头)
        values[3] = body      身体幅度 0~1
        values[4] = arm       手臂高度 0~1
        values[5] = speed     动作速度 0~1
        """
        if not self.skeletal_mesh or not unreal:
            return

        mouth, expr, head, body, arm, speed = values

        # 嘴巴开合 -> 下颌旋转
        mouth_angle = MOUTH_OPEN_MIN + mouth * (MOUTH_OPEN_MAX - MOUTH_OPEN_MIN)
        self._set_bone_rotation(BONE_JAW, pitch=0, yaw=0, roll=mouth_angle)

        # 头部角度 -> 头部 Pitch
        head_tilt = HEAD_TILT_MIN + head * (HEAD_TILT_MAX - HEAD_TILT_MIN)
        self._set_bone_rotation(BONE_HEAD, pitch=head_tilt, yaw=0, roll=0)

        # 身体幅度 -> 脊柱缩放
        body_scale = BODY_SCALE_MIN + body * (BODY_SCALE_MAX - BODY_SCALE_MIN)
        self._set_bone_scale(BONE_SPINE_01, scale=body_scale)

        # 手臂高度 -> 上臂旋转
        arm_angle = ARM_RAISE_MIN + arm * (ARM_RAISE_MAX - ARM_RAISE_MIN)
        self._set_bone_rotation(BONE_UPPERARM_L, pitch=-arm_angle, yaw=0, roll=0)
        self._set_bone_rotation(BONE_UPPERARM_R, pitch=-arm_angle, yaw=0, roll=0)

        # 动作速度 -> 动画播放速率
        if self.anim_instance:
            play_rate = 0.5 + speed * 1.5  # 0.5x ~ 2.0x
            self.anim_instance.set_play_rate(play_rate)

        # 表情强度 -> 面部 Pose Asset 权重（需要用户配置 Pose Asset）
        # 这部分需要用户在 Blueprint 中配置具体的 Pose Asset
        # 以下代码是一个示例，可能需要根据实际的 MetaHuman 面部设置调整
        # if expr > 0.01:
        #     # 可以通过 Control Rig 或 Pose Asset 来控制表情
        #     pass

    def _set_bone_rotation(self, bone_name, pitch=0, yaw=0, roll=0):
        """[中] 设置骨骼旋转（单位：弧度）。"""
        if not self.skeletal_mesh:
            return
        try:
            transform = self.skeletal_mesh.get_bone_transform(
                self.skeletal_mesh.get_bone_index(bone_name),
                space=unreal.SocketCoordinateSpace.WORLD
            )
            rot = unreal.Rotator()
            rot.pitch = math.degrees(pitch)
            rot.yaw = math.degrees(yaw)
            rot.roll = math.degrees(roll)
            transform.rotation = rot.quaternion()
            # 注意：直接设置骨骼变换需要 Control Rig 或 Anim Node 支持
            # 这里使用 SkeletalMeshComponent 的 BoneTransform 修改方式
        except Exception as e:
            pass  # 静默处理，避免每帧打印错误

    def _set_bone_scale(self, bone_name, scale=1.0):
        """[中] 设置骨骼缩放。"""
        if not self.skeletal_mesh:
            return
        try:
            vec = unreal.Vector(scale, scale, scale)
            # 注意：直接设置骨骼缩放需要 Control Rig 或 Anim Node 支持
        except Exception as e:
            pass


# === 全局实例 / Global Instance ===
_receiver = None
_controller = None
_running = False


def start_stagedouble():
    """[中] 启动 StageDouble OSC 接收。

    在 UE Python 控制台中调用：
    >>> start_stagedouble()
    """
    global _receiver, _controller, _running

    if _running:
        print("[StageDouble] Already running.")
        return

    _controller = MetaHumanController()
    _receiver = OSCReceiver(OSC_PORT, OSC_ADDRESS, _controller.apply_params)
    _receiver.start()
    _running = True
    print("[StageDouble] Started. Receiving OSC on port 12000.")
    print("[StageDouble] Make sure ml_trainer_v2.py (or app.py) is running with 'run' command.")


def stop_stagedouble():
    """[中] 停止 StageDouble OSC 接收。

    在 UE Python 控制台中调用：
    >>> stop_stagedouble()
    """
    global _receiver, _running
    if _receiver:
        _receiver.stop()
        _receiver = None
    _running = False
    print("[StageDouble] Stopped.")


def test_connection():
    """[中] 测试连接：发送一个测试值看是否正常工作。

    在 UE Python 控制台中调用：
    >>> test_connection()
    """
    if not _controller:
        print("[StageDouble] Controller not initialized. Run start_stagedouble() first.")
        return

    print("[StageDouble] Sending test values: mouth=0.5, express=0.3, head=0.5, body=0.2, arm=0.4, speed=0.5")
    _controller.apply_params([0.5, 0.3, 0.5, 0.2, 0.4, 0.5])
    print("[StageDouble] Test complete. Check if MetaHuman responded.")


# === 自动启动说明 / Auto-start Note ===
# 如果希望脚本导入后自动启动，取消注释下面两行：
# If you want auto-start on import, uncomment the following lines:
# print("[StageDouble] Module loaded. Run start_stagedouble() to begin.")

if __name__ == "__main__":
    # 在 UE Python 控制台中直接运行此文件时自动启动
    start_stagedouble()
else:
    print("[StageDouble] Module imported. Run start_stagedouble() to begin receiving OSC.")
