import importlib

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

    demo_index = options.prediction_type[-1]

    module_name = f"prediction.demo_{demo_index}"
    module = importlib.import_module(module_name)
    main_demo = getattr(module, f"main_demo_{demo_index}")

    main_demo(config, logger)
