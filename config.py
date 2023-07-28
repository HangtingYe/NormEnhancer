import os
import json
import argparse


class Config(object):
    def __init__(self):
        # ------- Basic Arguments -------
        self.seed = 0  # random seed
        self.device = 'cuda:0'  # please set 'CUDA_VISIBLE_DEVICES' when calling python

        # ------- Optimization Arguments -------
        self.max_epochs = 100
        self.patience = 5  # the number of evaluations waited before early stopping
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.weight_decay = 2e-4
        self.boost_rounds = 5

    def update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    def to_dict(self):
        return dict(self.__dict__)


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


def add_config_to_argparse(config, arg_parser):
    """The helper for adding configuration attributes to the argument parser"""
    for key, val in config.to_dict().items():
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        elif isinstance(val, (int, float, str)):
            arg_parser.add_argument('--' + key, type=type(val), default=val)
        else:
            raise Exception('Do not support value ({}) type ({})'.format(val, type(val)))


def get_config_from_command():
    # add arguments to parser
    config = Config()
    parser = argparse.ArgumentParser(description='Wind Power Forecasting')
    add_config_to_argparse(config, parser)

    # parse arguments from command line
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)

    return config


def get_config_from_file():
    config = Config()
    parser = argparse.ArgumentParser(description='Wind Power Forecasting')
    # parser.add_argument('--config_file', type=str, required=True)
    # parser.add_argument('--log_file', type=str)
    ################################### Tabular #####################
    
    # adult binary
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/binary/adult/tabnet.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/binary/adult/node.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/binary/adult/deepfm.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/binary/adult/fttransformer.json")
    
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/default/autoint.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/default/dcnv2.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/default/mlp.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/default/resnet.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/default/node.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/default/snn.json")

    # epsilon binary
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/binary/epsilon/deepfm.json")


    # higgs binary
    # parser.add_argument('--config_file', type=str, default="./test_config/config.seed.0.dataset.higgs.model.autoint.json")
    
    # year regression
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/regression/year/deepfm.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/regression/year/fttransformer.json")

    # aloi multi-class classification
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/multi-class/aloi/deepfm.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/multi-class/aloi/fttransformer.json")


    # helena multi-class classification
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/multi-class/helena/deepfm.json")
    # parser.add_argument('--config_file', type=str, default="./configs/tabular/classification/multi-class/helena/fttransformer.json")

    ################################### Time Series #####################

    # ETTh1 autoformer
    # parser.add_argument('--config_file', type=str, default="./configs/time/ETTh1/autoformer.json")
    # parser.add_argument('--config_file', type=str, default="./configs/time/ETTh1/gru.json")
    # parser.add_argument('--config_file', type=str, default="./configs/time/ETTh1/tcn.json")
    # parser.add_argument('--config_file', type=str, default="./configs/time//ETTh1lightTS.json")
    # parser.add_argument('--config_file', type=str, default="./configs/time/ETTh1/scinet.json")

    # WTH autoformer
    parser.add_argument('--config_file', type=str, default="./configs/time/WTH/autoformer.json")
    # parser.add_argument('--config_file', type=str, default="./configs/time/WTH/gru.json")
    # parser.add_argument('--config_file', type=str, default="./configs/time/WTH/tcn.json")
    # parser.add_argument('--config_file', type=str, default="./configs/time/WTH/lightTS.json")
    # parser.add_argument('--config_file', type=str, default="./configs/time/WTH/scinet.json")

    args = parser.parse_args()

    config.update_by_dict(
        json.load(open(args.config_file))
    )

    return config
