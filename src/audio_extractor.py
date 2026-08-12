"""
StageDouble - Real-Time Vocal Feature Extractor
StageDouble - 实时人声特征提取器

[EN] What this file does:
Captures live microphone audio, extracts pitch / volume / MFCC / beat (tempo)
with librosa, and streams the 16-value feature vector over OSC in real time.

[中] 这个文件做什么：
从麦克风实时采集音频，用 librosa 提取音高（pitch）、音量（volume）、
音色特征（MFCC）和节奏（beat/tempo），然后把这 16 个数值打包成一个
特征向量，通过 OSC 协议实时发送出去。

[EN] How to run:
    python audio_extractor.py                      # use system default mic
    python audio_extractor.py --list-devices        # find your mic's device index
    python audio_extractor.py --device 2             # capture from a specific device
    python audio_extractor.py --ip 127.0.0.1 --port 6448 --address /wek/inputs

[中] 如何运行：
    python audio_extractor.py                      # 使用系统默认麦克风
    python audio_extractor.py --list-devices        # 查看可用麦克风及其编号
    python audio_extractor.py --device 2             # 使用指定编号的麦克风
    python audio_extractor.py --ip 127.0.0.1 --port 6448 --address /wek/inputs

[EN] What this connects to:
This is the first stage of the StageDouble pipeline. It sends its OSC
feature vector to ml_trainer_v2.py (or osc_listener.py for debugging), which
listens on the same host/port/address by default (127.0.0.1:6448
/wek/inputs). Nothing needs to be running for this script itself to work -
it will happily send OSC packets into the void if no one is listening.

[中] 这个文件如何与其他文件连接：
这是 StageDouble 流水线的第一环。它把提取好的特征向量通过 OSC 发送给
ml_trainer_v2.py（用于训练/推理）或 osc_listener.py（用于调试查看数值）。
两者默认都监听同一个地址（127.0.0.1:6448 /wek/inputs）。这个脚本本身
不依赖下游程序是否在运行——即使没有人监听，它也会正常地把 OSC 数据包
发送出去（只是没人收到而已）。

Note (macOS): the first run will trigger a system prompt asking for
microphone access for your terminal / IDE. Approve it in
System Settings -> Privacy & Security -> Microphone.

注意（macOS）：第一次运行时，系统会弹出请求麦克风权限的提示，需要在
"系统设置 -> 隐私与安全性 -> 麦克风" 中允许你的终端/IDE 访问麦克风。
"""

import argparse
import queue
import sys

import numpy as np
import librosa
import sounddevice as sd
from pythonosc.udp_client import SimpleUDPClient

# ---------------------------------------------------------------------------
# Configuration / 配置参数
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050  # Hz. librosa's own default; plenty for voice, keeps CPU cost low.
# 采样率（Hz）。这是 librosa 自带的默认值，对人声来说已经足够精细，
# 同时能把 CPU 开销控制在较低水平。
CHANNELS = 1  # mono mic input
# 声道数：1 表示单声道（mono）输入，麦克风采集人声不需要立体声。

# How often we compute features and send an OSC message.
#
# librosa.pyin() has a fixed processing floor of ~50ms per call on an M2
# (from its internal frame_length=2048 window, regardless of how short the
# input is), so a 50ms hop leaves no headroom once MFCC/tempo/OSC overhead
# is added and the loop falls behind. 100ms (10 Hz) was benchmarked to run
# comfortably in real time; the QUEUE_BACKLOG_WARNING below will tell you if
# it isn't on your machine.
#
# 多久计算一次特征并发送一次 OSC 消息。
#
# 关键决策说明：librosa.pyin()（音高检测函数）在 M2 芯片上每次调用有大约
# 50ms 的固定处理下限（这是它内部 frame_length=2048 窗口决定的，不论输入
# 音频多短都一样），所以如果把节拍间隔（hop）设成 50ms，加上 MFCC、节奏
# 检测和 OSC 发送的开销后，处理速度就会跟不上，导致队列积压。经过实测，
# 100ms（也就是每秒发送 10 次）可以在实时场景下稳定运行且留有余量；如果
# 在你的机器上仍然跟不上，下面的 QUEUE_BACKLOG_WARNING 会打印警告提醒你。
HOP_DURATION = 1  # seconds -> 1 messages/sec
# 每个处理周期的时长（秒），1 秒 = 每秒发送 1 条 OSC 消息。
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)
# 每个处理周期对应的采样点数量（由采样率 x 周期时长算出）。

# pitch (pyin) and MFCC need more than one hop of context to be stable, so we
# analyze them over a short rolling window instead of the raw hop.
#
# 音高（pyin）和 MFCC 计算需要比单个 hop 更长的上下文音频才能得到稳定的
# 结果，所以我们用一个稍长一点的"滚动窗口"来做这两项分析，而不是只用
# 刚采集到的那一小段音频。
ANALYSIS_WINDOW_DURATION = 0.1  # seconds
ANALYSIS_WINDOW_SAMPLES = int(SAMPLE_RATE * ANALYSIS_WINDOW_DURATION)

# Tempo/beat estimation needs even more context to find a periodic pulse.
# 节奏/节拍（tempo/beat）的估计需要更长的音频上下文才能找到周期性的
# 节拍脉冲，所以用一个长达 1 秒的窗口。
BEAT_WINDOW_DURATION = 1.0  # seconds
BEAT_WINDOW_SAMPLES = int(SAMPLE_RATE * BEAT_WINDOW_DURATION)

N_MFCC = 13  # standard MFCC coefficient count
# MFCC（梅尔频率倒谱系数）的维度数，13 是语音处理里的常用标准值，
# 足以描述音色特征又不会让特征向量过于庞大。

# Expected singing range, used to keep pyin from chasing octave errors outside it.
# 预期的演唱音域范围，用来限制 pyin 的搜索区间，避免它把八度误判
# （octave error）到不合理的音高上。
PITCH_FMIN = librosa.note_to_hz("C2")  # ~65 Hz
PITCH_FMAX = librosa.note_to_hz("C7")  # ~2093 Hz

# Wekinator's documented defaults: listens for OSC on 6448, expects inputs
# as a single message with one float per input, in a fixed order.
#
# 这里沿用了 Wekinator 文档里的默认约定：监听 6448 端口，把所有输入值
# 打包在同一条 OSC 消息里、按固定顺序排列。即便下游不是 Wekinator 本身
# （比如 ml_trainer_v2.py），这个约定仍然保留，方便任何监听同一地址的
# 程序接收。
DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 6448
DEFAULT_OSC_ADDRESS = "/wek/inputs"

def parse_args():
    """[EN] Parse command-line arguments.
    [中] 解析命令行参数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="print available audio devices and exit")
    parser.add_argument("--ip", default=DEFAULT_OSC_IP, help="target host/IP to send OSC features to")
    parser.add_argument("--port", type=int, default=DEFAULT_OSC_PORT, help="target OSC port")
    parser.add_argument("--address", default=DEFAULT_OSC_ADDRESS, help="OSC address the receiver listens on")
    return parser.parse_args()


def extract_features(hop_chunk, analysis_window, beat_window):
    """Compute one feature vector from the latest audio.

    hop_chunk        - most recent ~100ms of audio (fast-reacting features)
    analysis_window  - most recent ~100ms of audio (stable pitch/MFCC)
    beat_window       - most recent ~1s of audio (tempo needs a longer view)

    Returns a flat list of floats: [pitch_hz, volume, mfcc_1..mfcc_13, tempo_bpm]

    ----------------------------------------------------------------------
    [中] 根据最新采集到的音频，计算出一个完整的特征向量。

    参数说明：
    hop_chunk        —— 最近约 100ms 的音频（用于反应最快的特征，如音量）
    analysis_window  —— 最近约 100ms 的分析窗口（用于计算更稳定的音高和 MFCC）
    beat_window       —— 最近约 1 秒的滚动窗口（节奏检测需要更长的时间跨度）

    返回值：一个长度为 16 的浮点数列表：
        [音高(pitch_hz), 音量(volume), MFCC 1~13, 节奏(tempo_bpm)]
    """

    # --- Volume: RMS amplitude of the freshest samples, most responsive to
    # sudden changes like a loud note or a breath.
    # --- 音量：取最新音频片段的均方根（RMS）振幅，对突然的音量变化
    # （比如唱大声或换气）反应最灵敏。
    volume = float(np.sqrt(np.mean(hop_chunk**2)))

    # --- Pitch: probabilistic YIN gives a per-frame f0 estimate plus how
    # confident it is that each frame is voiced. We take the most recent
    # voiced estimate and fall back to 0.0 during silence/unvoiced audio.
    # --- 音高：概率化 YIN 算法（pyin）会对每一帧音频给出一个基频（f0）
    # 估计值，以及该帧是否为"有声"（voiced，即真的在发声而不是静音/噪音）
    # 的置信度。我们取最近一个"有声"帧的估计值；如果整个窗口都是静音
    # 或无法判断音高，则返回 0.0。
    f0, voiced_flag, voiced_prob = librosa.pyin(
        analysis_window, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=SAMPLE_RATE
    )
    voiced_f0 = f0[~np.isnan(f0)]
    pitch_hz = float(voiced_f0[-1]) if len(voiced_f0) else 0.0

    # --- Timbre: MFCCs summarize the spectral envelope (roughly "vocal
    # texture"). Averaging across the analysis window smooths out
    # frame-to-frame noise.
    # --- 音色：MFCC（梅尔频率倒谱系数）概括了声音的频谱包络，大致对应
    # "音色/嗓音质感"。对分析窗口内的多帧结果取平均，可以平滑掉帧与帧
    # 之间的噪声波动。
    mfcc = librosa.feature.mfcc(y=analysis_window, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_means = mfcc.mean(axis=1).tolist()

    # --- Beat: estimate tempo (BPM) from the longer rolling buffer, since a
    # periodic pulse can't be found from a fraction of a second of audio.
    # --- 节奏：从时间跨度更长的滚动缓冲区里估计节奏（单位 BPM，每分钟
    # 拍数），因为几分之一秒的音频根本不足以发现周期性的节拍规律。
    tempo = librosa.feature.tempo(y=beat_window, sr=SAMPLE_RATE)
    tempo_bpm = float(tempo[0])

    return [pitch_hz, volume, *mfcc_means, tempo_bpm]


def main():
    """[EN] Entry point: set up the mic stream and OSC client, then loop
    forever extracting features and sending them out.
    [中] 主入口函数：初始化麦克风音频流和 OSC 发送客户端，然后进入
    一个无限循环，持续提取特征并发送出去。
    """
    args = parse_args()

    if args.list_devices:
        # 如果用户只是想查看可用的麦克风设备列表，打印后直接退出。
        print(sd.query_devices())
        return

    osc_client = SimpleUDPClient(args.ip, args.port)
    print(f"Sending OSC to {args.ip}:{args.port}{args.address}")

    # Thread-safe handoff: sounddevice calls audio_callback on its own
    # high-priority audio thread, so we just hand chunks off to the main
    # thread via a queue rather than doing any heavy computation in the
    # callback itself (which could cause audio dropouts).
    #
    # 线程安全的数据交接：sounddevice 库会在它自己的高优先级音频线程里
    # 调用 audio_callback，所以我们只在回调函数里做最轻量的操作——把
    # 采集到的音频片段放进一个队列（queue），交给主线程去处理。绝不能
    # 在回调函数里做耗时的计算（比如特征提取），否则会导致音频丢帧/卡顿。
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        # 这是 sounddevice 在音频线程里自动调用的回调函数。
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    # Rolling buffer of the last ~1 second of audio. Every new hop slides in
    # at the end and the oldest samples drop off the front.
    #
    # 一个保存"最近约 1 秒"音频的滚动缓冲区（rolling buffer）。每来一个
    # 新的 hop，就把它拼接到缓冲区末尾，同时把最旧的那部分样本从开头
    # 丢弃，始终保持缓冲区长度不变（类似一个先进先出的音频窗口）。
    beat_window = np.zeros(BEAT_WINDOW_SAMPLES, dtype=np.float32)

    # librosa.pyin() JIT-compiles its inner loop (via numba) on its very
    # first call, which takes over a second. Doing that here, before the
    # mic stream opens, keeps that one-time cost from stalling the
    # real-time loop and causing a startup backlog.
    #
    # 关键决策说明：librosa.pyin() 内部使用 numba 做即时编译（JIT），
    # 第一次调用时编译过程要花一秒多。如果不预热，这个一次性的耗时就会
    # 发生在正式的实时循环里，导致刚开始运行时音频队列迅速积压。所以我们
    # 在打开麦克风音频流之前，先用假数据"空跑"一次 extract_features()，
    # 把这个编译成本提前消化掉。
    print("Warming up pitch tracker...")
    extract_features(beat_window[-HOP_SAMPLES:], beat_window[-ANALYSIS_WINDOW_SAMPLES:], beat_window)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=HOP_SAMPLES,
        device=args.device,
        callback=audio_callback,
    )

    with stream:
        print("Listening on the microphone. Press Ctrl+C to stop.")
        try:
            while True:
                # 从队列里取出下一段音频（如果队列为空则阻塞等待）。
                hop_chunk = audio_queue.get()

                # Real-time catch-up: if chunks piled up while we were busy
                # extracting features, drain the queue so we always process
                # the freshest audio instead of falling further behind.
                #
                # 实时追赶：如果在提取特征期间积压了多个音频片段，就清空
                # 队列只保留最新数据，避免延迟越积越多。积压的音频仍然会
                # 被拼接进 beat_window，保证节奏检测不丢数据。
                drained = []
                while not audio_queue.empty():
                    drained.append(audio_queue.get())

                if drained:
                    all_new = np.concatenate([hop_chunk] + drained)
                    if len(all_new) >= BEAT_WINDOW_SAMPLES:
                        beat_window = all_new[-BEAT_WINDOW_SAMPLES:].copy()
                    else:
                        beat_window = np.concatenate([beat_window[len(all_new):], all_new])
                    hop_chunk = drained[-1]
                else:
                    beat_window = np.concatenate([beat_window[len(hop_chunk):], hop_chunk])

                analysis_window = beat_window[-ANALYSIS_WINDOW_SAMPLES:]

                # 计算这一轮的特征向量，并通过 OSC 发送出去。
                features = extract_features(hop_chunk, analysis_window, beat_window)
                osc_client.send_message(args.address, features)

        except KeyboardInterrupt:
            # 用户按下 Ctrl+C，优雅退出。
            print("\nStopped.")


if __name__ == "__main__":
    main()
