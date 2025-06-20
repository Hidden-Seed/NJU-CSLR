import os

from utils.skeleton import *
from utils.logger import *
from utils.config.mp_config import *


# 示例文件名：P01_01_00_0._color.mp4
def get_file_index(index, filename):
    parts = filename.split('_')                              # 将文件名按 "_" 分割
    first_number = int(parts[0][1:])
    valid_number = (int(parts[1]) - 1) * 20 + int(parts[2])  # 计算 label_num
    last_number = int(parts[3][:-1])
    if (valid_number == index):
        file_index = (first_number - 1) * 5 + last_number
        return f"{index}_{file_index}"
    else:
        return None


def create_log(config):
    data_range = options.data_range
    log_name = f"{data_range[0]}_{data_range[1]}_{config['log_name']}"
    logger = Logger(config["log_dir"], log_name)
    logger.info(config)

    return logger


def read_record_file(record_file_path):
    """
    读取已处理记录文件，返回一个已处理的文件索引集合
    """
    if not os.path.exists(record_file_path):
        return set()
    with open(record_file_path, 'r') as f:
        processed_files = {line.strip() for line in f.readlines()}
    return processed_files


def update_record_file(record_file_path, file_index):
    """
    更新记录文件，将处理过的视频索引写入文件
    """
    with open(record_file_path, 'a') as f:
        f.write(f"{file_index}\n")


def check_txt_file_format(txt_path, invalid_list, expected_count=138):
    """
    Check if each line in the txt file contains the expected number of values.
    """
    if os.path.getsize(txt_path) == 0:
        if txt_path in invalid_list:
            return True
        else:
            return False

    with open(txt_path, 'r') as f:
        for line in f:
            items = line.strip().split()
            if len(items) != expected_count:
                return False
    return True


if __name__ == "__main__":
    options = create_parser()
    config = read_config(options)
    logger = create_log(config)

    root_video_dir, output_base_dir = config['video_path'], config['save_path']
    invalid_file = os.path.join(output_base_dir, config['invalid_file'])
    record_file_path = os.path.join(output_base_dir, config['record_file'])
    processed_files = read_record_file(record_file_path)

    vp = VideoProcessor(root_video_dir, output_base_dir)

    data_range = options.data_range
    start_index, end_index = data_range[0], (data_range[1] + 1)
    logger.info(
        f"Processing index from {start_index:03d} to {(end_index - 1):03d}")

    for index in range(start_index, end_index):
        logger.info(f"Processing index {index:03d}...")
        for file in os.listdir(os.path.join(root_video_dir, f"{index:03d}")):
            file_index = get_file_index(index, file)
            if file_index and file_index not in processed_files:
                process_one(vp, f"{index:03d}", file, invalid_file, logger)
                update_record_file(record_file_path, file_index)
            else:
                logger.info(f"Skipping {file} (already processed or invalid file).")

    logger.info("All videos have been processed!")

    valid_flag = True
    with open(invalid_file, 'r') as f:
        invalid_list = set(line.strip() for line in f if line.strip())

    for index in range(start_index, end_index):
        label_dir = os.path.join(output_base_dir, f"{index:03d}")
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            logger.info(f"Check file {file}")
            ok = check_txt_file_format(file_path, invalid_list)
            if not ok:
                logger.warning(f"Invalid format in file {file}")
                valid_flag = False
                continue

    if valid_flag:
        logger.info("All txt files checking passed!")
    else:
        logger.warning("Txt files checking failed!")

    total_num = (end_index - start_index) * 250
    valid_rate = 1 - len(invalid_list) / total_num
    logger.info(f"Valid rate: {valid_rate:.2%}")
