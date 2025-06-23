import argparse
import configparser
import torch.distributed as dist


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="config/model.cfg",
                        help="path to model definition file")
    parser.add_argument("--model_type", type=str,
                        default="blstm", help="used model type (lstm, blstm)")
    parser.add_argument("--data_config", type=str, default="config/dataset.cfg",
                        help="path to data config file")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="interval between saving model weights",)
    parser.add_argument("--evaluation_interval", type=int, default=1,
                        help="interval evaluations on validation set",)
    parser.add_argument("--train_iteration", type=int, default=0,
                        help="interval evaluations on validation set",)
    # parser.add_argument("--local_rank", default=-1, type=int)
    options = parser.parse_args()

    if dist.get_rank() == 0:
        print(options)
    return options


def read_config(options):
    """Read the data and model config files"""
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

    model_config_type = options.model_type
    model_config = configparser.ConfigParser()
    model_config.read(options.model_config)

    config = {
        "data": data_config[data_config_type],
        "model": model_config[model_config_type],
    }
    return config
