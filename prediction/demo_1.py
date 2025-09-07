import os
import json

import torch

from utils.data_process import *
from utils.prediction import *


def main_demo_1(config, logger):
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

    # Dataloader
    dataloader = create_dataloader(config["demo"])

    final_predict = []
    ground_truth = []

    word_indexes = read_word_list(config["data"]["word_list"])

    for step, (b_x, b_y) in enumerate(dataloader):
        b_x = b_x.type(torch.FloatTensor).to(device)
        b_y = b_y.type(torch.long).to(device)
        with torch.no_grad():
            prediction = model(b_x)  # rnn output
        # h_s = h_s.data        # repack the hidden state, break the connection from last iteration
        # h_c = h_c.data        # repack the hidden state, break the connection from last iteration

        ground_truth.extend(b_y.view(b_y.size()[0]).cpu().numpy().tolist())

        pre_result = torch.max(F.softmax(prediction[:, -1, :], dim=1), 1)
        pre_class = pre_result[1].cpu().data.numpy().tolist()
        final_predict.extend(pre_class)

        # pre_prob = pre_result[0].cpu().data.numpy().tolist()
        # print(pre_class, pre_prob)

    ground_truth = np.asarray(ground_truth)
    final_predict = np.asarray(final_predict)

    accuracy_list = []
    labels = np.unique(ground_truth)

    for label in labels:
        indices = (ground_truth == label)
        total = indices.sum()
        correct = (final_predict[indices] == label).sum()
        accuracy = correct / total if total > 0 else 0.0

        real_label = word_indexes[label]
        cur_word = class_dict.get(int(real_label), "unknown")

        accuracy_list.append({
            "label": int(real_label),
            "word": cur_word,
            "accuracy": accuracy
        })

    # 按 accuracy 从大到小排序
    accuracy_list_sorted = sorted(
        accuracy_list,
        key=lambda x: x["accuracy"],
        reverse=True
    )

    # 转成 dict 并格式化百分比
    accuracy_dict_sorted = {}
    for item in accuracy_list_sorted:
        label_str = f"{item['label']:03d}"
        accuracy_dict_sorted[label_str] = {
            "word": item["word"],
            "accuracy": f"{item['accuracy']:.2%}"
        }

    # 保存 JSON 文件
    accuracy_file = config["demo"]["result_file"]
    if os.path.exists(accuracy_file):
        with open(accuracy_file, "r", encoding="utf-8") as f:
            model_accuracy = json.load(f)
    else:
        model_accuracy = {}

    model_name = os.path.basename(model_save_path)
    model_accuracy[model_name] = accuracy_dict_sorted

    with open(accuracy_file, "w", encoding="utf-8") as f:
        f.truncate(0)
        json.dump(model_accuracy, f, indent=4, ensure_ascii=False)
