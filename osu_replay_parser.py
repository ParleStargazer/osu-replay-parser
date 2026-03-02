import struct
import lzma
import os
import sys
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MultipleLocator

# 配置 Matplotlib 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

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

def parse_osr_and_plot_lines(file_path: str, width: int, height: int):
    """解析 osu! 回放文件并渲染按键持续时间分布图"""
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
        beatmap_hash = read_string(f)
        player_name = read_string(f)
        replay_hash = read_string(f)

        # 跳过固定长度的成绩数据区块 (2 * 6 + 4 + 2 + 1)
        f.read(19)

        mods = struct.unpack('<i', f.read(4))[0]
        life_bar = read_string(f)
        timestamp = struct.unpack('<q', f.read(8))[0]

        # 读取压缩数据长度和数据本体
        replay_data_length = struct.unpack('<i', f.read(4))[0]
        compressed_data = f.read(replay_data_length)

    # 解压 LZMA 数据
    try:
        decompressed_data = lzma.decompress(compressed_data).decode('ascii')
    except Exception as e:
        print(f"错误：LZMA 数据解压失败。详细信息: {e}")
        return

    # 状态机：分类计算每个按键的持续时间
    frames = decompressed_data.split(',')
    current_time = 0
    pressed_keys = {}
    durations_by_key = {col: [] for col in range(18)}

    for frame in frames:
        if not frame:
            continue

        parts = frame.split('|')
        if len(parts) < 4:
            continue

        w = int(parts[0])
        x_val = float(parts[1])

        # 跳过 RNG 随机数种子帧
        if w == -12345:
            continue

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

    # --- 绘图逻辑 ---
    # 根据像素和 DPI 计算 figsize (默认 DPI=100)
    dpi = 100
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    active_keys = [col for col, durs in durations_by_key.items() if len(durs) > 0]
    x_vals = list(range(161))

    ax = plt.gca()
    # 设置 X 轴主次刻度：主刻度每 10ms，次刻度每 1ms
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    # 设置 Y 轴最小分度值为 1
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    for key in sorted(active_keys):
        valid_durations = [d for d in durations_by_key[key] if 0 <= d <= 160]
        if not valid_durations:
            continue

        duration_counts = Counter(valid_durations)
        y_vals = [duration_counts[x] for x in x_vals]

        # 绘制折线图并隐藏数据点标记
        plt.plot(
            x_vals,
            y_vals,
            label=f'Key {key + 1}',
            alpha=0.85,
            linewidth=(2 if width < 2000 else 3),
            marker=None
        )

    # 设置图表标题和坐标轴标签
    file_basename = os.path.basename(file_path)
    title_fontsize = 14 * (width / 1280)
    label_fontsize = 12 * (width / 1280)

    plt.title(f"{file_basename}\nPlayer: {player_name}", fontsize=title_fontsize, fontweight='bold')
    plt.xlabel("按键持续时间 Duration (ms)", fontsize=label_fontsize)
    plt.ylabel("按键数量 Count", fontsize=label_fontsize)

    plt.xlim(0, 160)
    plt.ylim(0, None)  # 自动适应 Y 轴高度

    # 开启主次刻度的网格线
    plt.grid(which='major', axis='both', linestyle='-', alpha=0.3)
    plt.grid(which='minor', axis='both', linestyle=':', alpha=0.1)

    plt.legend(title="Keys", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=label_fontsize * 0.8)
    plt.tight_layout()

    # --- 导出图片逻辑 ---
    # 获取当前脚本所在目录作为默认输出路径
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    output_filename = f"{os.path.splitext(file_basename)[0]}_analysis.png"
    output_path = os.path.join(script_dir, output_filename)

    plt.savefig(output_path, format='png', dpi=dpi)
    plt.close()  # 关闭图形以释放内存

    print(f"\n解析与渲染完成。图表已保存至：\n{output_path}")


def main():
    print("=== osu!mania Replay 解析工具 ===")
    print("-" * 40)

    # 第一步：获取文件路径
    raw_path = input("[1] 请输入 .osr 文件路径（或将文件拖放至此）：\n> ")
    # 清理路径两端的空白字符及引号
    file_path = raw_path.strip().strip('\"').strip('\'')

    if not os.path.isfile(file_path):
        print("错误：指定的文件不存在，请检查路径后重试。")
        return

    # 第二步：选择分辨率
    print("\n[2] 请选择输出图片的分辨率：")
    print("  1 -> 1280 * 720  (HD)")
    print("  2 -> 1920 * 1080 (FHD)")
    print("  3 -> 2560 * 1440 (2K)")
    print("  4 -> 3840 * 2160 (4K)")

    choice = input("> 请输入选项 (1/2/3/4) [默认: 2]: ").strip()

    resolutions = {
        "1": (1280, 720),
        "2": (1920, 1080),
        "3": (2560, 1440),
        "4": (3840, 2160)
    }

    # 默认使用 1920x1080
    width, height = resolutions.get(choice, (1920, 1080))
    print(f"\n正在以 {width}x{height} 分辨率渲染图表，请稍候...")

    # 第三步：执行解析和渲染
    parse_osr_and_plot_lines(file_path, width, height)

    input("\n处理完毕，请按 Enter 键退出...")


if __name__ == "__main__":
    main()