import os
import numpy as np

import torch
import torch.nn.functional as F

from utils.data_process import *
from utils.logger import Logger


def read_dict_table(path):
    data = dict()
    with open(path, encoding='utf-8') as f:
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
    parts = filename.split('_')                            # 将文件名按 "_" 分割
    data_label = (int(parts[1]) - 1) * 20 + int(parts[2])  # 计算 label_num
    return data_label


def load_txt_data(txt_path, logger: Logger):
    if os.path.getsize(txt_path) == 0:
        logger.warning(f"Invalid txt file: {txt_path}!")
        return None

    data = np.loadtxt(txt_path)
    if data.shape[1] != 138:
        raise ValueError(
            f"Expected 138 values per line, but got {data.shape[1]}")

    return data.astype(np.float32)


def process_txt_data(data, config):
    enable_3D = config["data"].getboolean("3D_enable")
    enable_body = config["data"].getboolean("pose_enable")
    index_range, need_index = 42, []

    if enable_body:
        index_range += 4
    if enable_3D:
        for i in range(index_range):
            need_index.extend([i * 3, i * 3 + 1, i * 3 + 2])
    else:
        for i in range(index_range):
            need_index.extend([i * 3, i * 3 + 1])

    keyframe_num = int(config["data"]["keyframe_num"])
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
            new_frame = existing_frames[-1] * \
                (1 - ratio) + existing_frames[-2] * ratio
            new_frames.append(new_frame)
        new_frames = np.array(new_frames)
        key_frames = np.vstack((existing_frames, new_frames))

    else:
        key_frames = data[key_indexes]

    data_array = np.array(
        key_frames, dtype=np.float32).reshape(keyframe_num, 138)
    data_array = data_array[:, need_index]

    # crop_size = float(config["data"]["crop_size"])
    # for i in range(len(data_array)):
    #     data_array[i] = abs2rel(data_array[i], enable_3D, crop_size)

    return data_array


def class_index2name(dict_table, index, start_index=0):
    if start_index + index >= 500:
        return None
    return dict_table[start_index + index]


def predict(data_tensor, model, dict_table, logger: Logger):
    with torch.no_grad():
        prediction = model(data_tensor)
        pre_result = torch.max(
            F.softmax(prediction[:, -1, :], dim=1), 1)
        pre_class = pre_result[1].cpu().data.numpy().tolist()[0]
        pre_prob = pre_result[0].cpu().data.numpy().tolist()[0]

    word = class_index2name(dict_table, pre_class)

    if pre_prob < 0.8:
        logger.warning(f"Prediction confidence too low: {pre_prob}")

    return word
