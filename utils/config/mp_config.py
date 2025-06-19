import argparse
import configparser


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp_config", type=str, default="config/mp.cfg",
                        help="path to mp config file")
    parser.add_argument("--data_range", type=int, nargs=2, default=[0, 50],
                        help="range of data index")

    options = parser.parse_args()
    print(options)

    return options


def read_config(options):
    """Read the data config files"""
    mp_config_type = "holistic"
    mp_config = configparser.ConfigParser()
    mp_config.read(options.mp_config)

    config = mp_config[mp_config_type]
    return config
