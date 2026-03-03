import struct
import lzma
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.ticker import MultipleLocator
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# 配置 Matplotlib 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def get_base_path():
    """
    获取程序运行的基准路径。
    解决 PyInstaller --onefile 模式下路径指向临时文件夹的问题。
    """
    if getattr(sys, 'frozen', False):
        # 如果是打包后的可执行文件，返回 .exe 所在的目录
        return os.path.dirname(sys.executable)
    # 如果是脚本运行，返回脚本所在的目录
    return os.path.dirname(os.path.abspath(__file__))

def read_uleb128(f):
    """读取变长整数 ULEB128 格式数据"""
    result = 0
    shift = 0
    while True:
        byte = f.read(1)
        if not byte:
            break
        b = byte[0]
        result |= (b & 0x7f) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return result

def read_string(f):
    """读取 osu! 专用的变长字符串格式"""
    flag = f.read(1)
    if not flag:
        return ""
    if flag[0] == 0x0b:
        length = read_uleb128(f)
        return f.read(length).decode('utf-8')
    return ""

def lowpass_filter(data, cutoff, fs=1000, order=4):
    """应用低通滤波以平滑信号，如果 cutoff >= 500 则不执行滤波"""
    if cutoff >= fs / 2:
        return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def perform_fft_analysis(durations, cutoff_hz):
    """执行频域转换并估算 Hz (取频域最大值)"""
    if not durations: return None, None, 0
    
    # 构建信号序列，步长为 1ms，总长 512ms
    max_ms = 512
    signal = np.zeros(max_ms)
    counts = Counter([int(d) for d in durations if 0 < d < max_ms])
    for ms, count in counts.items():
        signal[ms] = count
    
    # 预处理：减去均值消除直流分量，并执行低通滤波
    signal = signal - np.mean(signal)
    signal = lowpass_filter(signal, cutoff_hz)

    # 快速傅里叶变换
    yf = fft(signal)
    xf = fftfreq(max_ms, 1/1000)[:max_ms//2]
    amplitude = 2.0/max_ms * np.abs(yf[0:max_ms//2])

    # 寻找 60Hz-1000Hz 之间的最大峰值
    mask = (xf >= 60) & (xf <= 1000)
    if not any(mask): return xf, amplitude, 0
    
    peak_idx = np.argmax(amplitude[mask])
    est_hz = xf[mask][peak_idx]
    return xf, amplitude, est_hz

def parse_osr_and_plot_lines(file_path: str, width: int, height: int, output_dir: str, cutoff_hz: float):
    """解析 osu! 回放文件并渲染按键持续时间分布图及其频域转换图"""
    if not os.path.exists(file_path):
        print("错误：找不到指定的文件，请确认文件路径是否正确。")
        return

    player_name = "Unknown"

    with open(file_path, 'rb') as f:
        # 解析头部数据
        game_mode = struct.unpack('<b', f.read(1))[0]
        if game_mode != 3:
            print("警告：该文件可能不是 osu!mania 的回放文件，程序将尝试继续解析。")

        version = struct.unpack('<i', f.read(4))[0]
        read_string(f)  # beatmap_hash
        player_name = read_string(f)
        read_string(f)  # replay_hash

        # 跳过固定长度的成绩数据区块
        f.read(19)

        struct.unpack('<i', f.read(4))[0]  # mods
        read_string(f)  # life_bar
        struct.unpack('<q', f.read(8))[0]  # timestamp

        # 读取压缩数据长度和数据本体
        replay_data_length = struct.unpack('<i', f.read(4))[0]
        compressed_data = f.read(replay_data_length)

    # 解压 LZMA 数据
    try:
        decompressed_data = lzma.decompress(compressed_data).decode('ascii')
    except Exception as e:
        print(f"错误：LZMA 数据解压失败。详细信息: {e}")
        return

    # 状态机处理按键数据
    frames = decompressed_data.split(',')
    current_time = 0
    pressed_keys = {}
    durations_by_key = {col: [] for col in range(18)}

    for frame in frames:
        if not frame: continue
        parts = frame.split('|')
        if len(parts) < 4: continue
        
        w = int(parts[0])
        x_val = float(parts[1])
        if w == -12345: continue

        current_time += w
        keys_bitmask = int(x_val)

        for col in range(18):
            is_pressed = (keys_bitmask & (1 << col)) != 0
            if is_pressed and col not in pressed_keys:
                pressed_keys[col] = current_time
            elif not is_pressed and col in pressed_keys:
                duration = current_time - pressed_keys[col]
                durations_by_key[col].append(duration)
                del pressed_keys[col]

    # --- 频域计算逻辑 ---
    all_durs = [d for durs in durations_by_key.values() for d in durs]
    xf, amp, est_hz = perform_fft_analysis(all_durs, cutoff_hz)

    # --- 绘图逻辑 ---
    dpi = 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / dpi, height / dpi), dpi=dpi)
    
    # [左子图：原有的时域折线图]
    active_keys = [col for col, durs in durations_by_key.items() if len(durs) > 0]
    x_vals = list(range(161))
    
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))

    for key in sorted(active_keys):
        valid_durations = [d for d in durations_by_key[key] if 0 <= d <= 160]
        if not valid_durations: continue
        duration_counts = Counter(valid_durations)
        y_vals = [duration_counts[x] for x in x_vals]
        ax1.plot(x_vals, y_vals, label=f'Key {key + 1}', alpha=0.85, 
                 linewidth=(2 if width < 2000 else 3))

    file_basename = os.path.basename(file_path)
    title_fontsize = 14 * (width / 1920)
    label_fontsize = 12 * (width / 1920)

    ax1.set_title(f"{file_basename}\nPlayer: {player_name} (时域分布)", fontsize=title_fontsize, fontweight='bold')
    ax1.set_xlabel("按键持续时间 Duration (ms)", fontsize=label_fontsize)
    ax1.set_ylabel("按键数量 Count", fontsize=label_fontsize)
    ax1.set_xlim(0, 160)
    ax1.grid(which='major', axis='both', linestyle='-', alpha=0.3)
    ax1.legend(title="Keys", loc='upper right', fontsize=label_fontsize * 0.7)

    # [右子图：新增的频域 FFT 分析图]
    if xf is not None:
        ax2.plot(xf, amp, color='crimson', linewidth=1.5)
        ax2.fill_between(xf, amp, alpha=0.15, color='crimson')
        ax2.set_title(f"频域分析 (估算键盘回报率: {est_hz:.1f} Hz)", fontsize=title_fontsize, fontweight='bold')
        ax2.set_xlabel("频率 Frequency (Hz)", fontsize=label_fontsize)
        ax2.set_ylabel("强度 Magnitude", fontsize=label_fontsize)
        ax2.set_xlim(0, 500) # 1ms采样下，Nyquist频率为500Hz
        ax2.xaxis.set_major_locator(MultipleLocator(125))
        ax2.grid(True, alpha=0.3)
        
        # 标注估算峰值
        max_v = np.max(amp[(xf >= 60)])
        ax2.annotate(f'Peak: {est_hz:.1f}Hz', xy=(est_hz, max_v), xytext=(est_hz+30, max_v),
                     arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()

    # --- 导出图片逻辑 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = f"{os.path.splitext(file_basename)[0]}_fft_analysis.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, format='png', dpi=dpi)
    plt.close()

    print(f"\n[解析成功]")
    print(f"检测到主峰频率: {est_hz:.2f} Hz")
    print(f"图表已保存至：\n{output_path}")

def main():
    print("=== osu!mania Replay 深度解析工具 (时域 + 频域) ===")
    print("-" * 40)

    # 第一步：获取文件路径
    raw_path = input("[1] 请输入 .osr 文件路径：\n> ")
    file_path = raw_path.strip().strip('\"').strip('\'')

    if not os.path.isfile(file_path):
        print("错误：文件不存在。")
        return

    # 新增：设置低通滤波
    cutoff_in = input("\n[2] 设置低通滤波上限 (Hz) [默认 1500, 即不滤波]:\n> ").strip()
    cutoff_hz = float(cutoff_in) if cutoff_in else 1500.0

    # 第二步：选择分辨率
    print("\n[3] 选择输出分辨率 (1:720P, 2:1080P, 3:2K, 4:4K):")
    choice = input("> [默认: 2]: ").strip()
    resolutions = {"1": (1280, 720), "2": (1920, 1080), "3": (2560, 1440), "4": (3840, 2160)}
    width, height = resolutions.get(choice, (1920, 1080))

    # 第三步：指定输出目录
    base_dir = get_base_path()
    print(f"\n[4] 请输入输出目录路径 (直接回车默认保存至当前程序所在目录):")
    custom_output = input("> ").strip().strip('\"').strip('\'')
    output_dir = custom_output if custom_output else base_dir

    print(f"\n正在执行傅里叶变换与渲染，请稍候...")
    parse_osr_and_plot_lines(file_path, width, height, output_dir, cutoff_hz)
    input("\n处理完毕，请按 Enter 键退出...")

if __name__ == "__main__":
    main()
