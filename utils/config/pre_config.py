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
    parser.add_argument("--data_config", type=str,
                        default="config/dataset.cfg", help="path to data config file")

    options = parser.parse_args()
    print(options)
    return options


def read_config(options):
    """Read the model and demo config files"""
    data_config_type = "DEFAULT"
    data_config = configparser.ConfigParser()
    data_config.read(options.data_config)

    # Standardize file name
    keyframe_num = data_config.getint(data_config_type, 'keyframe_num')
    enable_3D = data_config.getboolean(data_config_type, '3D_enable')
    enable_body = data_config.getboolean(data_config_type, 'pose_enable')
    node_num, dimension_num = 42, 2
    if enable_body:
        node_num = 46
    if enable_3D:
        dimension_num = 3
    data_file_name = data_config.get(data_config_type, 'data_file_name').format(
        keyframe_num=keyframe_num, node_num=node_num, dimension_num=dimension_num)
    label_file_name = data_config.get(data_config_type, 'label_file_name').format(
        keyframe_num=keyframe_num, node_num=node_num, dimension_num=dimension_num)

    data_config[data_config_type]['data_file_name'] = data_file_name
    data_config[data_config_type]['label_file_name'] = label_file_name

    demo_config_type = options.prediction_type
    demo_config = configparser.ConfigParser()
    demo_config.read(options.demo_config)

    mp_config_type = "holistic"
    mp_config = configparser.ConfigParser()
    mp_config.read(options.mp_config)

    config = {
        "data": data_config[data_config_type],
        "mp": mp_config[mp_config_type],
        "demo": demo_config[demo_config_type],
    }
    return config
