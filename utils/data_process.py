import os
import operator
from tqdm import tqdm
import numpy as np
import scipy.io as scio

from utils.logger import Logger


def txt2mat(txt_file_path, mat_path, logger: Logger):
    """
    Convert a .txt file to a .mat file.

    Args:
        txt_path (str): Path to the input .txt file.
        mat_path (str): Path to the output .mat file.
    """

    # Read txt as flat array
    data = np.loadtxt(txt_file_path)
    if data.size == 0:
        data = np.empty((0, 138))
    else:
        data = data.reshape(-1, 138)

    # Save to .mat
    scio.savemat(mat_path, {'data': data})
    logger.info(f"Saved to: {mat_path}")


def generate_mat(txt_data_dir, mat_data_dir, logger):
    for index in tqdm(range(500), desc="Labels", ncols=80):
        label = f"{index:03d}"
        txt_dir = os.path.join(txt_data_dir, label)
        mat_dir = os.path.join(mat_data_dir, label)
        os.makedirs(mat_dir, exist_ok=True)

        txt_files = [f for f in sorted(
            os.listdir(txt_dir)) if f.endswith('.txt')]

        logger.info(f"Processing label {label} with {len(txt_files)} files...")

        for fname in tqdm(txt_files, desc=f"Label {label}", ncols=80):
            txt_path = os.path.join(txt_dir, fname)
            mat_path = os.path.join(mat_dir, fname.replace('.txt', '.mat'))
            txt2mat(txt_path, mat_path, logger)


class Frame:
    """
    class to hold information about each frame
    """

    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def extract_keyframes_indexes(frames, keyframe_num):
    if len(frames) <= keyframe_num:
        return range(len(frames))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    new_frames = []
    for i in range(len(frames)):
        curr_frame = frames[i]
        if curr_frame is not None and \
                prev_frame is not None:
            diff = np.asarray(abs(curr_frame - prev_frame))
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / len(diff)
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            new_frames.append(frame)
        prev_frame = curr_frame

    # 计算关键帧
    keyframe_id_set = set()
    # 排序取前N帧
    new_frames.sort(key=operator.attrgetter("diff"), reverse=True)
    for keyframe in new_frames[:keyframe_num]:
        keyframe_id_set.add(keyframe.id)

    key_indexes = list(keyframe_id_set)
    key_indexes.sort()
    return key_indexes


def read_mat_file(mat_path, keyframe_num, label_idx, mat_file):
    """
    Read MAT file and extract keyframe indices.  
    When the number of keyframes is insufficient, use linear interpolation to fill in.。
    """
    mat_data = scio.loadmat(mat_path)["data"]
    if mat_data.size == 0 or mat_data.shape[0] == 0:
        return False, None

    mat_data = mat_data.astype(np.float32)
    key_indexes = extract_keyframes_indexes(mat_data, keyframe_num)

    if len(key_indexes) < keyframe_num:
        tqdm.write(
            f"{label_idx} {mat_file} length is too short, data supplementation is required.")

        # Data imputation
        existing_frames = mat_data[key_indexes]
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
        return True, key_frames

    else:
        key_frames = mat_data[key_indexes]
        return True, key_frames


# 绝对坐标转化为相对坐标
def abs2rel(data, enable_3D, crop_size):
    if enable_3D:
        data_x = data[0::3]
        data_y = data[1::3]
    else:
        data_x = data[0::2]
        data_y = data[1::2]

    x_min = np.min(data_x)
    x_max = np.max(data_x)
    y_min = np.min(data_y)
    y_max = np.max(data_y)

    data[0::2] = (data_x - x_min) / (x_max - x_min) * crop_size
    data[1::2] = (data_y - y_min) / (y_max - y_min) * crop_size
    return data
