# from importlib.metadata import requires
import numpy as np

import torch
import torch.nn.functional as F
from losses import eval_loss
import copy
from utils import normalizer_factory, get_boost_loss, get_eval_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# boost_lrs = [0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

class DataHandler:
    def __init__(self, config, norm_list, logger):
        self.config = config
        self.norm_list = norm_list
        self.logger = logger

        self.prepare_raw()
        self.prepare_additive()

        self.init_norm()

    def prepare_additive(self):
        self.additive_data = {}
        for key in ['train', 'val', 'test']:
            if self.config.task_type == "time":
                self.additive_data[key] = np.zeros_like(self.raw_data[f'{key}_output'])
            elif self.config.task_type == "tabular":
                if self.config.objective == "binary" or self.config.objective == "regression":
                    self.additive_data[key] = np.zeros_like(self.raw_data[f'{key}_output'],dtype=np.float32)
                else:  # multi-class classification
                    self.additive_data[key] = np.zeros((self.raw_data[f'{key}_output'].shape[0], self.config.class_num))

    def prepare_raw(self):
        if self.config.task_type == "tabular":
            self.prepare_tabular_data(self.config.all_data_path)
        self.raw_data = {}
        for key in ['train', 'val', 'test']:
            data_path = getattr(self.config, f'{key}_data_path')
            inputs, outputs,data = self.format_data(data_path)
            self.raw_data[f'{key}_data'] = data
            self.raw_data[f'{key}_input'] = inputs
            self.raw_data[f'{key}_output'] = outputs

    def prepare_tabular_data(self,all_data_path):
        ### the dataset don`t have category feature
        if not self.config.cat_idx:
            self.config.num_idx = range(self.config.num_features)
            return 
        self.label_encoders = {}
        num_idx = []
        data = np.load(all_data_path, allow_pickle=True)
        for i in range(self.config.num_features):
            if self.config.cat_idx and i in self.config.cat_idx:
                le = LabelEncoder()
                data[:, i] = le.fit_transform(data[:, i])
                self.label_encoders[i] = le
            else:
                num_idx.append(i)
        self.config.num_idx = num_idx



    def negative_grad(self, x, y):
        x = torch.from_numpy(x.astype('float32')).requires_grad_()
        y = torch.from_numpy(y).float()
        # use evaluation metric as the loss function
        loss_func = get_boost_loss(self.config.boost_loss)
        loss = loss_func(x, y)
        grad = -torch.autograd.grad(loss, x)[0]

        return grad.numpy()

    def init_norm(self):
        self.input_normalizer = {}
        self.output_normalizer = {}

        self.input_norm_data = {}
        self.output_norm_data = {}

    def reset_norm(self, rid):
        self.logger.info('reset normalizations')

        self.output_normalizer = {}
        self.output_norm_data = {}

        # calculate the output residuals
        self.output_residual = {}
        for key in ['train', 'val', 'test']:
            # directly forecast the label for the first round
            self.output_residual[key] = self.raw_data[f'{key}_output'] if rid == 0 \
                else self.negative_grad(self.additive_data[key], self.raw_data[f'{key}_output'])

        # build normalizers based on the training data
        for norm_cls in self.norm_list:
            self.logger.info(f'Init normalizer {norm_cls}')

            # only calc input normalization once
            if not norm_cls in self.input_normalizer:
                self.input_normalizer[norm_cls] = self.build_normalizer(self.raw_data['train_data'], norm_cls)
                for key in ['train', 'val', 'test']:
                    self.input_norm_data[f'{key}_{norm_cls}'] = self.normalize_data(
                        self.input_normalizer[norm_cls], self.raw_data[f'{key}_input']
                    )
            if self.config.task_type == "time":
                self.output_normalizer[norm_cls] = self.input_normalizer[norm_cls]
                for key in ['train', 'val', 'test']:
                    self.output_norm_data[f'{key}_{norm_cls}'] = self.normalize_data(
                        self.output_normalizer[norm_cls], self.output_residual[key]
                    )
            elif self.config.task_type == "tabular":
                if self.config.objective == "regression":
                    ### for regression label, build a normalizer
                    normalizer = normalizer_factory(norm_cls)
                    normalizer.fit(self.output_residual['train'])
                    self.output_normalizer[norm_cls] = normalizer
                    for key in ['train', 'val', 'test']:
                        self.output_norm_data[f'{key}_{norm_cls}'] = normalizer.transform(self.output_residual[key])

                else:
                    for key in ['train', 'val', 'test']:
                        self.output_norm_data[f'{key}_{norm_cls}'] = self.output_residual[key]


    def format_data(self, data_path):
        def _build_ts_data(data):
            data = [
                data[i - self.config.lookback: i + self.config.lookahead]
                for i in range(self.config.lookback, data.shape[0] - self.config.lookahead + 1)
            ]
            return np.asarray(data)

        def _build_tabular_data(x, y):
            if self.config.objective == "regression":
                y = y.astype(np.float32).reshape(-1,1)
            elif self.config.objective == "binary":
                le = LabelEncoder()
                y = le.fit_transform(y).astype(np.int16).reshape(-1,1)
            else:
                le = OneHotEncoder()
                y = y.astype(np.int16).reshape(-1,1)
                y = le.fit_transform(y).toarray()
            for i in range(self.config.num_features):
                if self.config.cat_idx and i in self.config.cat_idx:
                    x[:, i] = self.label_encoders[i].transform(x[:, i])
            return x, y

        data = np.load(data_path, allow_pickle=True)
        data = np.nan_to_num(data)  ### 

        if self.config.task_type == 'time':
            data = _build_ts_data(data)
            inputs = data[:, :self.config.lookback]
            outputs = data[:, self.config.lookback:]

        else:
            x, y = data[:, :-1], data[:, -1]
            inputs, outputs = _build_tabular_data(x, y)

        return inputs, outputs,data

    def build_normalizer(self, data, norm_cls):
        dim = data.shape[-1]
        normalizer = normalizer_factory(norm_cls)

        if self.config.task_type == 'time':
            data = data.reshape(-1, dim)
            normalizer.fit(data)
            return normalizer
        else:
            numberic_data = data[:, self.config.num_idx]
            numberic_data = normalizer.fit(numberic_data)
            return normalizer

    def normalize_data(self, normalizer, data):
        shape = data.shape
        data = data.reshape(-1, shape[-1])
        if self.config.task_type == "tabular":
            numberic_data = data[:, self.config.num_idx].astype(np.float32)
            category_data = data[:, self.config.cat_idx].astype(np.int16)
            numberic_data = normalizer.transform(numberic_data)
            # rerange cat_idx and num_idx
            self.config.new_num_idx = list(range(len(self.config.num_idx)))
            self.config.new_cat_idx = list(range(len(self.config.num_idx), self.config.num_features))
            data = np.concatenate([numberic_data, category_data], axis=-1)
        elif self.config.task_type == "time":
            data = normalizer.transform(data)
            data = data.reshape(*shape)
        return data

    def denormalize_data(self, normalizer, data):
        shape = data.shape
        data = data.reshape(-1, shape[-1])
        data = normalizer.inverse_transform(data)
        data = data.reshape(*shape)
        return data

    def normalized_data_dict(self, input_norm_cls, output_norm_cls):
        data_dict = {}
        for key_a in ['train', 'val', 'test']:
            for key_b in ['input', 'output']:
                key = f'{key_a}_{key_b}'
                if key_b == 'input':
                    data_dict[key] = self.input_norm_data[f'{key_a}_{input_norm_cls}']
                else:
                    data_dict[key] = self.output_norm_data[f'{key_a}_{output_norm_cls}']
        return data_dict

    def recoverd_pred_dict(self, output_norm_cls, experiment):
        pred_dict = {}
        for key in ['train', 'val', 'test']:
            if self.config.task_type == "time":
                pred_dict[key] = self.denormalize_data(
                    self.output_normalizer[output_norm_cls],
                    getattr(experiment, f'{key}_predictions')
                )
            elif self.config.task_type == "tabular":
                if self.config.objective == "regression":
                    pred_dict[key] = self.denormalize_data(
                        self.output_normalizer[output_norm_cls],
                        getattr(experiment, f'{key}_predictions')
                    )
                else:
                    pred_dict[key] = getattr(experiment, f'{key}_predictions')

        return pred_dict

    def evaluation(self, prediction_dict,rid):
        ### greedy search a reasonal boost_lr
        if rid == 0:
            boost_lrs = [1.0]
        else:
            boost_lrs = [0.009,0.01,0.04,0.05,0.08,0.09,0.1,0.2,0.3,0.5,0.8]
        best_boost_lr = None
        best_metric = None
        for  lr in  boost_lrs:
            metric = {}
            for key_a in ['train', 'val', 'test']:
                metric[key_a] = self.get_eval_metric(
                    prediction_dict[key_a] * lr + self.additive_data[f'{key_a}'],
                    self.raw_data[f'{key_a}_output']
                )
            if  best_metric is None:
                best_boost_lr = lr
                best_metric = copy.deepcopy(metric)
            #### The greater the accuracy/auc, the better
            elif (self.config.objective == "binary" or self.config.objective == "multi-class")  and metric['val'][self.config.eval_loss[0]] > best_metric['val'][self.config.eval_loss[0]]: 
                best_boost_lr = lr
                best_metric = copy.deepcopy(metric)
            ### The smaller the mse/mae ,the better
            elif  (self.config.objective  == "forecasting" or self.config.objective == "regression") and metric['val'][self.config.eval_loss[0]] < best_metric['val'][self.config.eval_loss[0]] :
                best_boost_lr = lr
                best_metric = copy.deepcopy(metric)

        return best_metric,best_boost_lr

    def update_additive(self, prediction_dict, lr):
        for key in ['train', 'val', 'test']:
            self.additive_data[key] += prediction_dict[key] * lr

    def get_eval_metric(self, predictions, outputs):
        metric = {}

        for f_name in self.config.eval_loss:
            func = get_eval_loss(f_name)
            metric[f_name] = func(predictions, outputs)

        return metric
