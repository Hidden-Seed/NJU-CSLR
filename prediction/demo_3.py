import os
import numpy as np

import torch
import torch.nn.functional as F

from utils.data_process import *
from utils.prediction import *


def main_demo_3(config, logger):
    capture_mp(config["demo"], logger)
    generate_txt(config["demo"], logger)

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

    txt_data_file = config["demo"]["tmp_txt"]
    data = load_txt_data(txt_data_file, logger)

    # Process data
    data_array = process_txt_data(data, config["demo"])
    # Convert to a Tensor.
    data_tensor = torch.from_numpy(data_array).to(device).unsqueeze(0)

    # Predict the result
    with torch.no_grad():
        prediction = model(data_tensor)
        prob_vector = F.softmax(prediction[:, -1, :], dim=1)
        prob_vector_np = prob_vector.cpu().numpy()[0]

    # 原始方差
    var_full = np.var(prob_vector_np)

    # 去掉最大值后
    max_idx = np.argmax(prob_vector_np)
    prob_vector_removed = np.delete(prob_vector_np, max_idx)
    var_removed = np.var(prob_vector_removed)

    diff = float(var_full - var_removed)
    diff_threshold = float(config["demo"]["diff_threshold"])
    logger.info(f"Get diff: {diff}")

    pred_class = int(max_idx)
    word = class_index2name(class_dict, pred_class)
    logger.info(f"Prediction result: {word}")

    if diff <= diff_threshold:
        logger.info("Invalid words!")
