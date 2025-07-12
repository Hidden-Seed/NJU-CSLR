import os
import subprocess

from utils.plot import plot_skeleton
from utils.prediction import get_data_label

if __name__ == "__main__":
    txt_base_path = "/home/zyn/Data/CSLR_dataset/mp_txt/"
    video_base_path = "/home/zyn/Data/CSL2018/gloss/color-gloss/color/"

    txt_data_files = [
        "P10_01_01_0._color_skeleton.txt",
    ]

    txt_data_path = []
    for file in txt_data_files:
        label = f"{get_data_label(file):03d}"
        txt_data_path.append(os.path.join(txt_base_path, label, file))

    print(txt_data_path)

    save_dir = "./dataset/skeleton_viz/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for txt_data in txt_data_path:

        frame_idx = 36
        plot_skeleton(txt_data, 36, save_dir)

        filename = os.path.basename(txt_data)
        video_filename = filename.replace("_color_skeleton.txt", "_color.mp4")
        video_path = os.path.join(video_base_path, label, video_filename)

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]

        try:
            fps_str = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT).decode().strip()
            # 例如 "30/1"
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
        except subprocess.CalledProcessError as e:
            print("Error reading fps:", e.output.decode())
            fps = 30.0    # fallback

        print(f"{fps} fps")

        # 计算秒数
        time_sec = frame_idx / fps
        time_str = f"{time_sec:.3f}"

        output_image = os.path.join(
            save_dir, f"{os.path.splitext(video_filename)[0]}_frame{frame_idx}.jpg")

        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", time_str,
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",   # 对 JPG：1 最好 31 最差
            output_image
        ]

        print("Running ffmpeg:", " ".join(ffmpeg_cmd))
        subprocess.run(ffmpeg_cmd, check=True)

        print(f"Saved video frame {frame_idx} to {output_image}")
