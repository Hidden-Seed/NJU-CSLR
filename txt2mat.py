import os
from tqdm import tqdm

from utils.logger import *
from utils.data_process import *
from utils.config.data_config import *


def create_log(config):
    logger = Logger(config["data"]["log_dir"], config["data"]["log_name"])
    logger.info(config)

    return logger


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


if __name__ == "__main__":
    options = create_parser()
    config = read_config(options)
    logger = create_log(config)

    txt_data_dir = config["mp"]["save_path"]
    mat_data_dir = config["data"]["dataset_dir"]

    # Convert txt to mat
    generate_mat(txt_data_dir, mat_data_dir, logger)
