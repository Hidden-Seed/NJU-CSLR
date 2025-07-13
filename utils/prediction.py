import os
import cv2
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_process import *
from utils.skeleton import *
from utils.logger import Logger


def read_dict_table(path):
    data = dict()
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            line_list = line.split()
            key = int(line_list[0])
            value = str(line_list[1])
            data[key] = value
    return data


def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint["model"]  # Extract the network architecture
    # Load the network weight
    model.load_state_dict(checkpoint["model_state_dict"])

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    return model


def get_data_label(filename):
    parts = filename.split("_")  # 将文件名按 "_" 分割
    data_label = (int(parts[1]) - 1) * 20 + int(parts[2])  # 计算 label_num
    return data_label


def load_txt_data(txt_path, logger: Logger):
    if os.path.getsize(txt_path) == 0:
        # logger.warning(f"Invalid txt file: {txt_path}!")
        return None

    data = np.loadtxt(txt_path)
    if data.shape[1] != 138:
        raise ValueError(f"Expected 138 values per line, but got {data.shape[1]}")

    return data.astype(np.float32)


def get_model_info(info_file, model_name):
    with open(info_file, "r") as f:
        model_info = json.load(f)

    return model_info[model_name]


def process_txt_data(data, config_demo):
    info_file = config_demo["model_info"]
    model_name = os.path.basename(config_demo["model_path"])
    model_info = get_model_info(info_file, model_name)

    input_size = model_info["input_size"]
    keyframe_num = model_info["time_step"]

    index_range, need_index = 42, []
    if input_size % 46 == 0:  # enable_body
        index_range += 4

    if input_size >= 126:
        for i in range(index_range):
            need_index.extend([i * 3, i * 3 + 1, i * 3 + 2])
    else:
        for i in range(index_range):
            need_index.extend([i * 3, i * 3 + 1])

    key_indexes = extract_keyframes_indexes(data, keyframe_num)

    if len(key_indexes) < keyframe_num:
        # Data imputation
        existing_frames = data[key_indexes]
        # Num of extra frames
        needed_frames = keyframe_num - len(existing_frames)
        # Generate new frames using linear interpolation
        new_frames = []

        for i in range(needed_frames):
            ratio = i / (needed_frames + 1)
            # Select two adjacent keyframes for interpolation, the second-to-last and the last one.
            new_frame = existing_frames[-1] * (1 - ratio) + existing_frames[-2] * ratio
            new_frames.append(new_frame)
        new_frames = np.array(new_frames)
        key_frames = np.vstack((existing_frames, new_frames))

    else:
        key_frames = data[key_indexes]

    data_array = np.array(key_frames, dtype=np.float32).reshape(keyframe_num, 138)
    data_array = data_array[:, need_index]

    # x_indices = np.arange(0, data_array.shape[1], 2)
    # data_array[:, x_indices] = 1.0 - data_array[:, x_indices]

    # crop_size = float(config["data"]["crop_size"])
    # for i in range(len(data_array)):
    #     data_array[i] = abs2rel(data_array[i], enable_3D, crop_size)

    return data_array


def class_index2name(dict_table, index, start_index=0):
    if start_index + index >= 500:
        return None
    return dict_table[start_index + index]


def predict(data_tensor, model, logger: Logger):
    with torch.no_grad():
        prediction = model(data_tensor)
        pre_result = torch.max(F.softmax(prediction[:, -1, :], dim=1), 1)
        pre_class = pre_result[1].cpu().data.numpy().tolist()[0]
        pre_prob = pre_result[0].cpu().data.numpy().tolist()[0]

    # if pre_prob < 0.8:
    #     logger.warning(f"Prediction confidence too low: {pre_prob}")

    return pre_class


def create_dataloader(config_demo):
    data_path = config_demo["data_path"]
    label_path = config_demo["label_path"]

    # Read the data and convert to a Tensor
    np_data_x = np.load(data_path)
    np_data_y = np.load(label_path)
    data_x = torch.from_numpy(np_data_x)
    data_y = torch.from_numpy(np_data_y)

    # Model config
    info_file = config_demo["model_info"]
    model_name = os.path.basename(config_demo["model_path"])
    model_info = get_model_info(info_file, model_name)

    input_size = model_info["input_size"]
    time_step = model_info["time_step"]
    cpu_nums = model_info["cpu_nums"]
    batch_size = model_info["batch_size"]

    # Outermost layer is a list, second layer is a tuple, innermost layers are ndarrays.
    data = list(data_x.numpy().reshape(1, -1, time_step, input_size))
    data.append(list(data_y.numpy().reshape(-1, 1)))
    data = list(zip(*data))

    # Create dataLoader
    dataloader = DataLoader(
        data, batch_size=batch_size, num_workers=cpu_nums, pin_memory=True
    )
    return dataloader


def capture_mp(config_demo, logger: Logger, max_frames=120):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置摄像头分辨率

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    logger.info(f"Size of camera: {width}x{height}")

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # 某些摄像头返回 0，手动设个默认值

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 或 'XVID' 对应 avi
    video_path = config_demo["tmp_video"]
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter!")

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot get frames!")

        out.write(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def generate_txt(config_demo, logger: Logger):
    video_path = config_demo["tmp_video"]
    txt_path = config_demo["tmp_txt"]
    vp = VideoProcessor(None, None)

    process_video(vp, video_path, txt_path, "dataset/tmp_invalid.txt", logger)
