import os
import numpy as np
import matplotlib.pyplot as plt


def plot_skeleton(txt_file, frame_idx, output_dir):
    """
    绘制骨骼并保存为图片,用于可视化判断脚本是否有问题
    参数:
        skeleton_points: 需要绘制的骨骼点列表（这里是 3D 坐标）
        frame_idx: 该图片的帧数
        output_dir: 保存路径
    返回:无
    """
    skeleton_points = np.loadtxt(txt_file)
    file_name = os.path.basename(txt_file)
    file_name.replace("._color.mp4", "")

    if skeleton_points is None:
        return

    # 将 3D 坐标转换为 2D 坐标（只取 x 和 y）
    frame = skeleton_points[frame_idx]
    formatted_points = [(frame[i], frame[i+1])
                        for i in range(0, len(frame), 3)]

    # 手部和肩部关键点分割
    left_hand = formatted_points[:21] if formatted_points[:21] else []
    right_hand = formatted_points[21:42] if formatted_points[21:42] else []
    shoulders = formatted_points[42:46] if formatted_points[42:46] else []

    # 设置图像大小
    plt.figure(figsize=(6, 8))

    if left_hand:
        plt.scatter(*zip(*left_hand), color='blue', label="Left Hand")
    if right_hand:
        plt.scatter(*zip(*right_hand), color='red', label="Right Hand")
    if shoulders:
        plt.scatter(*zip(*shoulders), color='green', label="Shoulders")

    # 固定坐标轴范围，假设坐标的值域是 -1 到 1
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)

    # 设置其他属性
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"2D Skeleton Visualization - Frame {frame_idx}")
    plt.legend()
    plt.gca().invert_yaxis()  # 反转 Y 轴以符合图像坐标

    # 保存图像
    save_path = os.path.join(
        output_dir, f"{file_name}_frame_{frame_idx:04d}.png")
    plt.savefig(save_path, dpi=800)
    plt.close()
