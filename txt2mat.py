from utils.logger import *
from utils.data_process import generate_mat
from utils.config.data_config import *


def create_log(config):
    logger = Logger(config["data"]["log_dir"], config["data"]["log_name"])
    logger.info(config)

    return logger


if __name__ == "__main__":
    options = create_parser()
    config = read_config(options)
    logger = create_log(config)

    txt_data_dir = config["mp"]["save_path"]
    mat_data_dir = config["data"]["dataset_dir"]

    # Convert txt to mat
    generate_mat(txt_data_dir, mat_data_dir, logger)
