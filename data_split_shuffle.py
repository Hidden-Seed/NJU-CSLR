import os
import sys
import numpy as np

from utils.config.data_config import *


def shuffle_data(data_set, label_set):
    if len(data_set) != len(label_set):
        print(
            f"Error: The size of data ({len(data_set)}) and label ({len(label_set)}) are not equal!")
        return None

    shuffle_index = np.random.permutation(len(data_set))
    data_set_shuffled = data_set[shuffle_index]
    label_set_shuffled = label_set[shuffle_index]

    return data_set_shuffled, label_set_shuffled


if __name__ == "__main__":
    options = create_parser()
    config = read_config(options)

    # Processing config
    keyframe_num = int(config["data"]["keyframe_num"])
    train_size, valid_size, test_size = float(config["data"]["train_data_size"]), float(
        config["data"]["valid_data_size"]), float(config["data"]["test_data_size"])

    # Path config
    save_dir = config["data"]["dataset_processed_dir"]
    data_npy_name = config["data"]["data_file_name"]
    label_npy_name = config["data"]["label_file_name"]
    data_npy_path = os.path.join(save_dir, data_npy_name)
    label_npy_path = os.path.join(save_dir, label_npy_name)

    if os.path.exists(data_npy_path) and os.path.exists(label_npy_path):
        data_array, label_array = np.load(
            data_npy_path), np.load(label_npy_path)
    else:
        print("Data not been preprocessed!")
        sys.exit(1)

    unique_labels = np.unique(label_array)
    print(unique_labels)
    print(data_array.shape[0])
    exit()
    train_data, valid_data, test_data = [], [], []

    for label in unique_labels:
        indices = np.where(label_array == label)[0]
        data_for_label = data_array[indices]

        # Split the data according to the ratio
        num_samples = len(data_for_label)
        train_end = int(train_size * num_samples)
        valid_end = train_end + int(valid_size * num_samples)

        train_data.append(data_for_label[:train_end])
        valid_data.append(data_for_label[train_end:valid_end])
        test_data.append(data_for_label[valid_end:])

    x_train = np.concatenate(train_data, axis=0)
    x_valid = np.concatenate(valid_data, axis=0)
    x_test = np.concatenate(test_data, axis=0)

    # 合并标签
    y_train = np.concatenate([np.full(len(data), label, dtype=np.int16)
                              for label, data in zip(unique_labels, train_data)], axis=0)
    y_valid = np.concatenate([np.full(len(data), label, dtype=np.int16)
                              for label, data in zip(unique_labels, valid_data)], axis=0)
    y_test = np.concatenate([np.full(len(data), label, dtype=np.int16)
                            for label, data in zip(unique_labels, test_data)], axis=0)

    x_train, y_train = shuffle_data(x_train, y_train)
    x_valid, y_valid = shuffle_data(x_valid, y_valid)
    x_test, y_test = shuffle_data(x_test, y_test)

    np.save(os.path.join(
        save_dir, f"train_{data_npy_name}"), x_train, allow_pickle=True)
    np.save(os.path.join(
        save_dir, f"valid_{data_npy_name}"), x_valid, allow_pickle=True)
    np.save(os.path.join(
        save_dir, f"test_{data_npy_name}"), x_test, allow_pickle=True)

    np.save(os.path.join(
        save_dir, f"train_{label_npy_name}"), y_train, allow_pickle=True)
    np.save(os.path.join(
        save_dir, f"valid_{label_npy_name}"), y_valid, allow_pickle=True)
    np.save(os.path.join(
        save_dir, f"test_{label_npy_name}"), y_test, allow_pickle=True)
