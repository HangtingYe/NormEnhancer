import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import datetime

from losses import train_loss, boost_loss, eval_loss

from copy import deepcopy
from heapq import heappush, heappop
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, FunctionTransformer, RobustScaler, MinMaxScaler


# Helper functions
def dump_json_to(obj, fpath, indent=2, ensure_ascii=False, **kwargs):
    """The helper for dumping json into the given file path"""
    with open(fpath, 'w') as fout:
        json.dump(obj, fout, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def load_json_from(fpath, **kwargs):
    """The helper for loading json from the given file path"""
    with open(fpath, 'r') as fin:
        obj = json.load(fin, **kwargs)

    return obj



def normalizer_factory(norm_cls):
    if norm_cls == 'z-score':
        return StandardScaler()
    elif norm_cls == 'quant':
        return QuantileTransformer(output_distribution='normal')
    elif norm_cls == 'power':
        return make_pipeline(StandardScaler(), PowerTransformer())
    elif norm_cls == 'none':
        return FunctionTransformer()
    elif norm_cls == 'robust':
        return RobustScaler()
    elif norm_cls == 'minmax':
        return MinMaxScaler()

def get_eval_loss(loss_cls):
    return getattr(eval_loss, loss_cls)


def get_boost_loss(loss_cls):
    return getattr(boost_loss, loss_cls)


def get_train_loss(loss_cls):
    return getattr(train_loss, loss_cls)


def put_batch_to_device(self, batch, device=None):
    if device is None:
        device = self.device

    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
        return batch
    elif isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict) or isinstance(value, list):
                batch[key] = self.put_batch_to_device(value, device=device)
            # retain other value types in the batch dict
        return batch
    elif isinstance(batch, list):
        new_batch = []
        for value in batch:
            if isinstance(value, torch.Tensor):
                new_batch.append(value.to(device))
            elif isinstance(value, dict) or isinstance(value, list):
                new_batch.append(
                    self.put_batch_to_device(value, device=device))
            else:
                # retain other value types in the batch list
                new_batch.append(value)
        return new_batch
    else:
        raise Exception('Unsupported batch type {}'.format(type(batch)))


def prepare_batch(self, batch):
    return self.put_batch_to_device(batch, self.device)


class EarlyStopping(object):
    """Early stopping + checkpoint ensemble"""

    def __init__(self, patience, queue_size=1) -> None:
        self.patience = patience
        assert queue_size >= 1
        self.queue_size = queue_size
        self.latest_iter = -1
        self._no_update_count = 0

        # min heap
        # item: (score, iter, model)
        # score: the larger, the better
        self.model_heaps = []

    def update_and_decide(self, score, iter, model) -> bool:
        if len(self.model_heaps) >= self.queue_size and score <= self.model_heaps[0][0]:
            self._no_update_count += 1
            return self._no_update_count > self.patience

        # reset count due to better checkpoints discovered
        self._no_update_count = 0
        if len(self.model_heaps) >= self.queue_size:
            heappop(self.model_heaps)

        heappush(self.model_heaps, (score, iter, deepcopy(model.state_dict())))
        self.latest_iter = iter

        return False

    def get_best_model_state(self):
        state_dict = {}
        for key, param in self.model_heaps[0][2].items():
            for i in range(1, len(self.model_heaps)):
                param += self.model_heaps[i][2][key]
            state_dict[key] = param / len(self.model_heaps)

        return state_dict

    def get_best_score(self):
        if len(self.model_heaps) > 0:
            return max([x[0] for x in self.model_heaps])
        else:
            return None



output_dir = "output/"

def save_loss_to_file(args, arr, name, extension=""):
    filename = get_output_path(args, directory="logging", filename=name, extension=extension, file_type="txt")
    np.savetxt(filename, arr)


def save_predictions_to_file(arr, args, extension=""):
    filename = get_output_path(args, directory="predictions", filename="p", extension=extension, file_type="npy")
    np.save(filename, arr)


def save_model_to_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    pickle.dump(model, open(filename, 'wb'))


def load_model_from_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    return pickle.load(open(filename, 'rb'))


def save_results_to_json_file(args, jsondict, resultsname, append=True):
    """ Write the results to a json file. 
        jsondict: A dictionary with results that will be serialized.
        If append=True, the results will be appended to the original file.
        If not, they will be overwritten if the file already exists. 
    """
    filename = get_output_path(args, filename=resultsname, file_type="json")
    if append:
        if os.path.exists(filename):
            old_res = json.load(open(filename))
            for k, v in jsondict.items():
                old_res[k].append(v)
        else:
            old_res = {}
            for k, v in jsondict.items():
                old_res[k] = [v]
        jsondict = old_res
    json.dump(jsondict, open(filename, "w"))


def save_results_to_file(args, results, train_time=None, test_time=None, best_params=None):
    filename = get_output_path(args, filename="results", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write(args.model_name + " - " + args.dataset + "\n\n")

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if train_time:
            text_file.write("\nTrain time: %f\n" % train_time)

        if test_time:
            text_file.write("Test time: %f\n" % test_time)

        if best_params:
            text_file.write("\nBest Parameters: %s\n\n\n" % best_params)


def save_hyperparameters_to_file(args, params, results, time=None):
    filename = get_output_path(args, filename="hp_log", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write("Parameters: %s\n\n" % params)

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if time:
            text_file.write("\nTrain time: %f\n" % time[0])
            text_file.write("Test time: %f\n" % time[1])

        text_file.write("\n---------------------------------------\n")


def get_output_path(args, filename, file_type, directory=None, extension=None):
    # For example: output/LinearModel/Covertype
    dir_path = output_dir + args.model_name + "/" + args.dataset

    if directory:
        # For example: .../models
        dir_path = dir_path + "/" + directory

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + "/" + filename

    if extension is not None:
        file_path += "_" + str(extension)

    file_path += "." + file_type

    # For example: .../m_3.pkl

    return file_path


def get_predictions_from_file(args):
    dir_path = output_dir + args.model_name + "/" + args.dataset + "/predictions"

    files = os.listdir(dir_path)
    content = []

    for file in files:
        content.append(np.load(dir_path + "/" + file))

    return content
