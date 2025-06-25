import os

import torch

from utils.data_process import *
from utils.prediction import *
from utils.logger import Logger
from utils.config.pre_config import *


def create_log(config):
    logger = logger = Logger(
        config["demo"]["log_dir"], config["demo"]["log_name"])
    logger.info(config)

    return logger


if __name__ == "__main__":
    options = create_parser()
    config = read_config(options)
    logger = create_log(config)

    # Read dict
    dict_path = config["demo"]["dictionary_path"]
    if not os.path.exists(dict_path):
        logger.error("Class dict does not exist!")
        raise FileNotFoundError("Can not find class dictionary.")
    class_dict = read_dict_table(dict_path)

    # Load txt data
    txt_data_file = "P12_01_01_0._color_skeleton.txt"
    data_label = get_data_label(txt_data_file)

    txt_data_path = os.path.join(
        config["mp"]["save_path"], f"{data_label:03d}", txt_data_file)
    data = load_txt_data(txt_data_path, logger)

    # Set GPU
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    logger.info(str(device))
    logger.info(torch.cuda.get_device_name(0))

    # Set random seed
    torch.manual_seed(int(config["model"]["SEED"]))

    # Model config
    batch_size = int(config["model"]["BATCH_SIZE"])
    cpu_nums = int(config["model"]["CPU_NUMS"])
    time_step = int(config["model"]["TIME_STEP"])
    input_size = int(config["model"]["INPUT_SIZE"])
    output_size = int(config["model"]["OUTPUT_SIZE"])

    # Load model
    model_save_path = config["demo"]["model_path"]
    model = load_checkpoint(model_save_path, device)
    model.to(device)

    # Process data
    data_array = process_txt_data(data, config)
    # Convert to a Tensor.
    data_tensor = (torch.from_numpy(
        data_array).to(device).unsqueeze(0))

    # Predict the result
    pre_word = predict(data_tensor, model, class_dict, logger)
    real_word = class_index2name(class_dict, data_label)
    logger.info(f"Prediction result: {pre_word}")
    logger.info(f"Correct word     : {real_word}")

    if pre_word == real_word:
        logger.info("Predict successfully!")
    else:
        logger.warning("Prediction failed!")