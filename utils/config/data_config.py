import argparse
import configparser


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str,
                        default="config/dataset.cfg", help="path to data config file")
    parser.add_argument("--mp_config", type=str, default="config/mp.cfg",
                        help="path to mp config file")
    options = parser.parse_args()

    print(options)
    return options


def read_config(options):
    data_config_type = "DEFAULT"
    data_config = configparser.ConfigParser()
    data_config.read(options.data_config)

    # Standardize file name
    keyframe_num = data_config.getint(data_config_type, 'keyframe_num')
    enable_3D = data_config.getboolean(data_config_type, '3D_enable')
    enable_body = data_config.getboolean(data_config_type, 'body_enable')
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

    mp_config_type = "holistic"
    mp_config = configparser.ConfigParser()
    mp_config.read(options.mp_config)

    config = {
        "data": data_config[data_config_type],
        "mp": mp_config[mp_config_type],
    }

    return config
