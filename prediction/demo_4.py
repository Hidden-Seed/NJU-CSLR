from utils.data_process import read_word_list
from utils.prediction import read_dict_table


def main_demo_4(config, logger):
    word_list_path = config["data"]["word_list"]
    word_indexes = read_word_list(word_list_path)

    word_dict_path = config["demo"]["dictionary_path"]
    word_dict = read_dict_table(word_dict_path)

    for idx in word_indexes:
        print(f"{idx:03d}: {word_dict[idx]}")
