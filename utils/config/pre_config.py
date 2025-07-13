import argparse
import configparser


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_type", type=str,
                        default="local", help="type of prediction (local, server etc.)")
    parser.add_argument("--demo_config", type=str,
                        default="config/demo.cfg", help="path to demo config file")
    parser.add_argument("--mp_config", type=str, default="config/mp.cfg",
                        help="path to mp config file")

    options = parser.parse_args()
    print(options)
    return options


def read_config(options):
    """Read the model and demo config files"""

    demo_config_type = options.prediction_type
    demo_config = configparser.ConfigParser()
    demo_config.read(options.demo_config)

    mp_config_type = "holistic"
    mp_config = configparser.ConfigParser()
    mp_config.read(options.mp_config)

    config = {
        "mp": mp_config[mp_config_type],
        "demo": demo_config[demo_config_type],
    }
    return config
