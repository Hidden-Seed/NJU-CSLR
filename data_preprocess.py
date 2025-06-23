import os
from tqdm import tqdm
import numpy as np

from utils.logger import *
from utils.data_process import *
from utils.config.data_config import *


def create_log(config):
    logger = Logger(config["data"]["log_dir"], config["data"]["log_name"])
    logger.info(config)

    return logger


if __name__ == "__main__":
    options = create_parser()
    config = read_config(options)
    logger = create_log(config)

    # Path config
    mat_data_dir = config["data"]["dataset_dir"]
    save_dir = config["data"]["dataset_processed_dir"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Processing config
    crop_size = float(config["data"]["crop_size"])
    keyframe_num = int(config["data"]["keyframe_num"])
    enable_3D = bool(config["data"]["3D_enable"])
    enable_body = bool(config["data"]["pose_enable"])
    index_range, need_index = 42, []

    if enable_body:
        index_range += 4
    if enable_3D:
        for i in range(index_range):
            need_index.extend([i * 3, i * 3 + 1, i * 3 + 2])
    else:
        for i in range(index_range):
            need_index.extend([i * 3, i * 3 + 1])

    data, label = [], []

    for index in tqdm(range(0, 500), desc="Processing labels", colour="green"):
        label_idx = "%03d" % index
        label_dir = os.path.join(mat_data_dir, label_idx)
        # Keyframe data of all mats under a specific label
        label_data = []

        # Traverse all mat data files under the current label directory
        files = [f for f in os.listdir(label_dir) if f.endswith('.mat')]
        for mat_file in tqdm(files, desc=f"Processing files in {label_idx}",
                             colour="blue", leave=False):
            mat_file_path = os.path.join(label_dir, mat_file)
            ok, key_frames = read_mat_file(
                mat_file_path, keyframe_num, label_idx, mat_file)
            if not ok:
                continue

            # Flat append
            label_data.append(key_frames)
            label.append(index)

        # label_data_array = np.array(label_data, dtype=np.float32)
        # data.append(label_data_array)
        # cur_label = int(index)
        # label.append(np.ones(len(label_data)) * cur_label)

    # Extract the required keypoint data of need_index
    print("Cutting data...")
    data_array = np.array(
        data, dtype=np.float32).reshape(-1, keyframe_num, 138)
    data_array = data_array[:, :, need_index]
    label_array = np.array(label, dtype=np.int16).reshape(-1, 1)

    for i in tqdm(range(len(data_array)), desc="Converting to relative coordinates", colour="green"):
        for j in range(len(data_array[i])):
            data_array[i][j] = abs2rel(data_array[i][j], enable_3D, crop_size)

    # Save the processed dataset
    data_npy_name = config["data"]["data_file_name"]
    label_npy_name = config["data"]["label_file_name"]

    np.save(os.path.join(save_dir, data_npy_name),
            data_array, allow_pickle=True)
    np.save(os.path.join(save_dir, label_npy_name),
            label_array, allow_pickle=True)
