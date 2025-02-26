from ctypes import util
import torch
import numpy as np
import importlib
import copy
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import lightgbm as lgb

from utils import get_train_loss

from itertools import cycle


class BaseExp:
    """Define the basic APIs"""

    def __init__(self, data_dict, config, logger,rid, device='cuda'):
        super().__init__()
        self.config = config
        self.logger = logger
        self.device = device
        self.cur_rid = rid

        self.prepare_data(data_dict)
        self.prepare_loss()
        self.prepare_metric()

    def prepare_loss(self):
        raise NotImplementedError

    def prepare_data(self, data_dict):
        raise NotImplementedError

    def prepare_metric(self):
        self.best_val_ = np.inf

    def run():
        raise NotImplementedError

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

    def get_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )


class NetExp(BaseExp):
    def __init__(self, data_dict, config, logger,rid):
        super().__init__(data_dict, config, logger,rid)

        self.model = self.build_model()
        self.model.to(self.device)
        self.get_optimizer()

    def prepare_data(self, data_dict):
        def _build_dataloader(inputs, outputs, batch_size, shuffle=False):
            index = torch.arange(inputs.shape[0])
            inputs = torch.from_numpy(inputs).float()
            outputs = torch.from_numpy(outputs).float()
            dataset = TensorDataset(index, inputs, outputs)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            return dataloader

        # for entire run
        self.train_loader = _build_dataloader(
            data_dict['train_input'], data_dict['train_output'], batch_size=self.config.batch_size, shuffle=True
        )
        self.val_loader = _build_dataloader(
            data_dict['val_input'], data_dict['val_output'], batch_size=self.config.batch_size*10, shuffle=False
        )
        self.test_loader = _build_dataloader(
            data_dict['test_input'], data_dict['test_output'], batch_size=self.config.batch_size*10, shuffle=False
        )

        self.input_dim = data_dict['train_input'].shape[-1]

    def prepare_loss(self):
        self.loss_func = get_train_loss(self.config.train_loss)

    def build_model(self):
        model_class = importlib.import_module(
            f'models.time.{self.config.model}'
        )
        model = model_class.Model(self.config).to(self.device)
        return model

    def standard_run(self):
        patience = self.config.patience

        for eid in range(self.config.max_epochs):
            self.model.train()
            train_error = 0.0

            for iter, batch in enumerate(self.train_loader):
                _, inputs, outputs = self.prepare_batch(batch)
                predictions = self.model(inputs)

                loss = self.loss_func(predictions, outputs)

                train_error += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 50)
                self.optimizer.step()

            val_metric, _, _ = self.eval_on(self.val_loader)
            test_metric, _, _ = self.eval_on(self.test_loader)

            if val_metric < self.best_val_:
                patience = self.config.patience
                self.best_val_ = val_metric
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                patience -= 1

            if patience == 0:
                break

            self.logger.info(
                f'Epoch {eid}, Train Loss {train_error / (iter + 1)}, Eval Metric {val_metric}, Test Metric {test_metric}'
            )

    def quick_run(self, max_batches):
        self.model.train()

        for iter, batch in enumerate(cycle(self.train_loader)):
            if iter + 1 == max_batches:
                break
            _, inputs, outputs = self.prepare_batch(batch)
            predictions = self.model(inputs)

            loss = self.loss_func(predictions, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 50)
            self.optimizer.step()

        val_metric, _, val_prediction = self.eval_on(self.val_loader)

        if val_metric < self.best_val_:
            self.best_val_ = val_metric
            self.best_prediction = val_prediction

    def run(self, max_batches=None):
        self.logger.info('Training starts')
        self.logger.info('Train Loader Length: {}'.format(len(self.train_loader)))

        if max_batches is None:
            self.standard_run()
        else:
            self.quick_run(max_batches)

    def get_final_predictions(self):
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()

        _, _, train_predictions = self.eval_on(self.train_loader)
        _, _, val_predictions = self.eval_on(self.val_loader)
        _, _, test_predictions = self.eval_on(self.test_loader)

        self.train_predictions = train_predictions
        self.val_predictions = val_predictions
        self.test_predictions = test_predictions

    def eval_on(self, dataloader):
        self.model.eval()

        index_arr = []
        outputs_arr = []
        predictions_arr = []

        for index, inputs, outputs in dataloader:
            inputs = self.prepare_batch(inputs)
            with torch.no_grad():
                predictions = self.model(inputs)

            index_arr.append(index)
            outputs_arr.append(self.put_batch_to_device(outputs, 'cpu'))
            predictions_arr.append(self.put_batch_to_device(predictions, 'cpu'))

        index_arr = torch.cat(index_arr, dim=0)
        outputs_arr = torch.cat(outputs_arr, dim=0)
        predictions_arr = torch.cat(predictions_arr, dim=0)

        metric = self.loss_func(predictions_arr, outputs_arr)

        index_arr = index_arr.data.numpy()
        outputs_arr = outputs_arr.data.numpy()
        predictions_arr = predictions_arr.data.numpy()

        # re-order outputs and predictions
        outputs_arr = outputs_arr[np.argsort(index_arr)]
        predictions_arr = predictions_arr[np.argsort(index_arr)]

        return metric, outputs_arr, predictions_arr


class TimeLGBExp(BaseExp):
    def __init__(self, data_dict, config, logger):
        super().__init__(data_dict, config, logger)

        self.model = self.build_model()

    def prepare_data(self, data_dict):
        self.train_X, self.train_y = data_dict['train_input'], data_dict['train_output']
        self.val_X, self.val_y = data_dict['val_input'], data_dict['val_output']
        self.test_X, self.test_y = data_dict['test_input'], data_dict['test_output']

        self.train_X = self.train_X.reshape(self.train_X.shape[0], -1)
        self.val_X = self.val_X.reshape(self.val_X.shape[0], -1)
        self.test_X = self.test_X.reshape(self.test_X.shape[0], -1)

        self.train_y = self.train_y.reshape(-1, )
        self.val_y = self.val_y.reshape(-1, )
        self.test_y = self.test_y.reshape(-1, )

    def prepare_loss(self):
        pass

    def build_model(self):
        model = lgb.LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=31,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.seed
        )
        return model

    def standard_run(self):
        self.model.fit(
            self.train_X, self.train_y, eval_set=[(self.val_X, self.val_y), (self.test_X, self.test_y)], early_stopping_rounds=self.config.patience
        )
        self.best_val_ = self.model.best_score_

    def quick_run(self, max_batches):
        pass

    def run(self, max_batches=None):
        self.logger.info('Training starts')

        if max_batches is None:
            self.standard_run()

    def get_final_predictions(self):
        # self.model.load_state_dict(self.best_state_dict)
        # self.model.eval()

        # _, _, train_predictions = self.eval_on(self.train_loader)
        # _, _, val_predictions = self.eval_on(self.val_loader)
        # _, _, test_predictions = self.eval_on(self.test_loader)

        train_predictions = self.model.predict(self.train_X)
        val_predictions = self.model.predict(self.val_X)
        test_predictions = self.model.predict(self.test_X)

        self.train_predictions = train_predictions.reshape(train_predictions.shape[0], 1)
        self.val_predictions = val_predictions.reshape(val_predictions.shape[0], 1)
        self.test_predictions = test_predictions.reshape(test_predictions.shape[0], 1)

class TabularNetExp(NetExp):
    def __init__(self, data_dict, config, logger,rid):
        super().__init__(data_dict, config, logger,rid)

        self.model = self.build_model()
        self.get_optimizer()

    def build_model(self):
        model_class = importlib.import_module(
            f'models.tabular.{self.config.model}'
        )
        model = model_class.Model(self.config).to(self.device)
        return model
    def prepare_loss(self):
        if self.cur_rid == 0:
            self.loss_func = get_train_loss(self.config.train_loss)
        else:
            self.loss_func = get_train_loss("mse")


class TimeNetExp(NetExp):
    def __init__(self, data_dict, config, logger,rid):
        super().__init__(data_dict, config, logger,rid)

        self.model = self.build_model()
        self.get_optimizer()

    def build_model(self):
        model_class = importlib.import_module(
            f'models.time.{self.config.model}'
        )
        model = model_class.Model(self.config).to(self.device)
        return model
