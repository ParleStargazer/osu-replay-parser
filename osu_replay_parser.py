import struct
import lzma
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.ticker import MultipleLocator
from scipy.fft import fft, fftfreq

# 配置 Matplotlib 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def get_base_path():
    """
    获取程序运行的基准路径。
    解决 PyInstaller --onefile 模式下路径指向临时文件夹的问题。
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def read_uleb128(f):
    """读取变长整数 ULEB128 格式数据"""
    result, shift = 0, 0
    while True:
        byte = f.read(1)
        if not byte: break
        b = byte[0]
        result |= (b & 0x7f) << shift
        if (b & 0x80) == 0: break
        shift += 7
    return result

def read_string(f):
    """读取 osu! 专用的变长字符串格式"""
    flag = f.read(1)
    if not flag or flag[0] != 0x0b: return ""
    length = read_uleb128(f)
    return f.read(length).decode('utf-8')

def perform_fft_analysis(durations):
    """
    执行原始 FFT 分析：
    1. 不进行插值和平滑，维持原始 1ms 采样特征。
    2. 去除直流分量以解决低频误报。
    """
    if not durations: return None, None, 0, "无有效数据"
    
    fs = 1000  # 原始采样率 1ms = 1000Hz
    max_ms = 512
    
    # 1. 构建原始信号 (1ms 步长)
    signal = np.zeros(max_ms)
    counts = Counter([int(d) for d in durations if 0 < d < max_ms])
    for ms, count in counts.items():
        signal[ms] = count
    
    # 2. 核心优化：去直流分量 (减去均值)
    # 这一步能消除 0Hz 附近的巨大能量堆积，防止误抓极低频峰值
    signal = signal - np.mean(signal)

    # 3. 执行 FFT
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)[:N//2]
    amplitude = 2.0/N * np.abs(yf[0:N//2])

    # 4. 自动评级判断 (搜索范围设定为 10Hz - 1000Hz)
    mask = (xf >= 10) & (xf <= 1000)
    if not any(mask): return xf, amplitude, 0, "分析失败"
    
    search_amp = amplitude[mask]
    peak_idx = np.argmax(search_amp)
    est_hz = xf[mask][peak_idx]
    
    # 计算信噪比显著性
    avg_amp = np.mean(amplitude[(xf >= 10)])
    snr = amplitude[mask][peak_idx] / avg_amp if avg_amp > 0 else 0
    
    # 判定结论
    if snr < 3.5: 
        conclusion = ">=500Hz (表现接近记录上限)"
    else:
        conclusion = f"主峰: {est_hz:.1f}Hz"

    return xf, amplitude, est_hz, conclusion

def parse_osr_and_plot_lines(file_path: str, width: int, height: int, output_dir: str):
    """解析 osu! 回放文件并渲染图表"""
    if not os.path.exists(file_path):
        print("错误：文件不存在。")
        return

    player_name = "Unknown"
    with open(file_path, 'rb') as f:
        f.read(1) 
        struct.unpack('<i', f.read(4))[0] 
        read_string(f) 
        player_name = read_string(f)
        read_string(f) 
        f.read(19) 
        struct.unpack('<i', f.read(4))[0] 
        read_string(f) 
        struct.unpack('<q', f.read(8))[0] 
        replay_data_length = struct.unpack('<i', f.read(4))[0]
        compressed_data = f.read(replay_data_length)

    try:
        decompressed_data = lzma.decompress(compressed_data).decode('ascii')
    except Exception as e:
        print(f"解压失败: {e}")
        return

    frames = decompressed_data.split(',')
    current_time, pressed_keys = 0, {}
    durations_by_key = {col: [] for col in range(18)}

    for frame in frames:
        if not frame: continue
        parts = frame.split('|')
        if len(parts) < 4: continue
        w, x_val = int(parts[0]), float(parts[1])
        if w == -12345: continue
        current_time += w
        keys_bitmask = int(x_val)
        for col in range(18):
            is_pressed = (keys_bitmask & (1 << col)) != 0
            if is_pressed and col not in pressed_keys:
                pressed_keys[col] = current_time
            elif not is_pressed and col in pressed_keys:
                durations_by_key[col].append(current_time - pressed_keys[col])
                del pressed_keys[col]

    # 执行核心 FFT 分析
    all_durs = [d for sublist in durations_by_key.values() for d in sublist]
    xf, amp, est_hz, conclusion = perform_fft_analysis(all_durs)

    # --- 渲染逻辑 ---
    dpi = 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / dpi, height / dpi), dpi=dpi)
    
    # 左图：保留原有时域折线图渲染
    active_keys = [col for col, durs in durations_by_key.items() if len(durs) > 0]
    x_vals_t = np.arange(161)
    for key in sorted(active_keys):
        v_durs = [int(d) for d in durations_by_key[key] if 0 <= d <= 160]
        if not v_durs: continue
        d_counts = Counter(v_durs)
        y_vals = [d_counts[x] for x in x_vals_t]
        ax1.plot(x_vals_t, y_vals, label=f'Key {key+1}', alpha=0.8)
    
    ax1.set_title(f"时域分布 (Player: {player_name})", fontweight='bold')
    ax1.set_xlabel("按键时长 Duration (ms)")
    ax1.set_ylabel("数量 Count")
    ax1.set_xlim(0, 160)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize='x-small')

    # 右图：频域 FFT 分析图 (0-1000Hz)
    if xf is not None:
        ax2.plot(xf, amp, color='darkgreen')
        ax2.fill_between(xf, amp, alpha=0.15, color='green')
        ax2.set_title(f"频域特征 (自动判断: {conclusion})", fontweight='bold')
        ax2.set_xlabel("频率 Frequency (Hz)\n[注: 基于原始 1ms 采样数据分析]", fontsize=10, color='gray')
        ax2.set_ylabel("强度 Magnitude")
        ax2.set_xlim(0, 1000) 
        ax2.xaxis.set_major_locator(MultipleLocator(125))
        ax2.grid(True, alpha=0.3)
        
        # 标注真实特征峰
        if est_hz > 0:
            max_v = amp[np.argmin(np.abs(xf - est_hz))]
            ax2.annotate(f'Peak: {est_hz:.1f}Hz', xy=(est_hz, max_v), xytext=(est_hz+60, max_v),
                         arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    file_basename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(file_basename)[0]}_hz_analysis.png")
    plt.savefig(output_path)
    plt.close()

    print(f"\n[解析成功]")
    print(f"评估结果: {conclusion}")
    print(f"检测主峰频率: {est_hz:.2f} Hz")
    print(f"提示: osu! Replay 物理记录精度上限为 1000Hz")
    print(f"图表已保存至: {output_path}")

def main():
    print("=== osu!mania Replay 解析工具 ===")
    print("提示: 回放文件的时间戳精度为 1ms，等效最高采样率为 1000Hz。")
    print("-" * 50)

    path = input("[1] 请输入 .osr 文件路径:\n> ").strip().strip('\"').strip('\'')
    if not os.path.isfile(path):
        print("错误：文件不存在。")
        return

    print("\n[2] 选择输出分辨率 (1:720P, 2:1080P, 3:2K, 4:4K):")
    res_choice = input("> [默认 2]: ").strip()
    resolutions = {"1": (1280, 720), "2": (1920, 1080), "3": (2560, 1440), "4": (3840, 2160)}
    width, height = resolutions.get(res_choice, (1920, 1080))

    base_dir = get_base_path()
    print(f"\n[3] 正在生成频域分析图表...")
    parse_osr_and_plot_lines(path, width, height, base_dir)
    input("\n处理完毕，请按 Enter 键退出...")

if __name__ == "__main__":
    main()
