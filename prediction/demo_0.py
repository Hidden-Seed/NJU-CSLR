import os

import torch

from utils.data_process import *
from utils.prediction import *


def main_demo_0(config, logger):
    # Read dict
    dict_path = config["demo"]["dictionary_path"]
    if not os.path.exists(dict_path):
        logger.error("Class dict does not exist!")
        raise FileNotFoundError("Can not find class dictionary.")
    class_dict = read_dict_table(dict_path)

    # Set GPU
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    logger.info(str(device))
    logger.info(torch.cuda.get_device_name(0))

    # Set random seed
    torch.manual_seed(0)

    # Load model
    model_save_path = config["demo"]["model_path"]
    model = load_checkpoint(model_save_path, device)
    model.to(device)

    txt_data_dir = config["mp"]["save_path"]
    total_num, success_num = (0, 0)
    word_indexes = read_word_list(config["data"]["word_list"])

    for label in word_indexes:
        txt_data_path = os.path.join(txt_data_dir, f"{label:03d}")
        for txt_data_file in os.listdir(txt_data_path):

            # Load txt data
            # txt_data_file = "P12_01_01_0._color_skeleton.txt"
            # data_label = get_data_label(txt_data_file)

            txt_data_file = os.path.join(txt_data_path, txt_data_file)
            data = load_txt_data(txt_data_file, logger)
            if data is None:
                continue
            else:
                total_num += 1

            # Process data
            data_array = process_txt_data(data, config["demo"])
            # Convert to a Tensor.
            data_tensor = (torch.from_numpy(
                data_array).to(device).unsqueeze(0))

            # Predict the result
            pre_class_tmp = predict(data_tensor, model, logger)
            pre_class = word_indexes[pre_class_tmp]

            pre_word = class_index2name(class_dict, pre_class)
            real_word = class_index2name(class_dict, label)
            logger.info(f"Prediction result: {pre_word}")
            logger.info(f"Correct word     : {real_word}")

            if pre_class == label:
                logger.info("Predict successfully!")
                success_num += 1
            else:
                logger.warning("Prediction failed!")

    success_rate = success_num / total_num
    logger.info(f"Rate of successful prediction: {success_rate:.2%}")
