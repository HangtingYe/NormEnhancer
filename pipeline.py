import os
import copy
import random
import itertools
import importlib

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import logging

import experiments

from data_handler import DataHandler
from config import Config

from utils import dump_json_to, load_json_from, EarlyStopping




norm_list = ['none', 'power', 'quant', 'z-score', 'minmax', 'robust']

# norm_list = ['quant', 'z-score',]
class Pipeline(object):
    def __init__(self, config: Config) -> None:
        logger.info('Pipeline Initialization')
        self.set_config(config)
        self.set_random_seed()
        self.prepare_env()

    def set_config(self, config: Config):
        self.config = config
        self.norm_list = norm_list
        self.device = torch.device(self.config.device)

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = self.config.seed
        else:
            self.config.seed = seed
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info(f'Set random seed {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_env(self):
        self.data_handler = DataHandler(self.config, self.norm_list, logger)
        self.exp_cls = getattr(experiments, self.config.exp)

    def _get_boost_lr(self, rid):
        if rid == 0:
            return 1.0
        else:
            return self.config.boosting_lr

    def bandit_search(self, rid):
        if self.config.use_norm_combine:
            norm_options = list(itertools.product(self.norm_list, self.norm_list))
        else:
            norm_options = [(x, x) for x in self.norm_list]

        experiments = {}

        for (input_norm_cls, output_norm_cls) in norm_options:
            norm_data_dict = self.data_handler.normalized_data_dict(input_norm_cls, output_norm_cls)
            experiments[(input_norm_cls, output_norm_cls)] = self.exp_cls(
                data_dict=norm_data_dict,
                config=self.config,
                logger=logger,
                rid = rid
            )

        for t in range(10):
            metrics = []

            max_batches = (1 << t) * self.config.unit_batch
            max_batches = min(max_batches, (1 << t) * self.config.unit_batch)

            for (input_norm_cls, output_norm_cls) in norm_options:
                experiment = experiments[(input_norm_cls, output_norm_cls)]

                experiment.quick_run(max_batches=max_batches)

                val_pred = self.data_handler.denormalize_data(
                    self.data_handler.output_normalizer[output_norm_cls],
                    experiment.best_prediction
                )
                val_additive = self.data_handler.additive_data['val']
                val_raw = self.data_handler.raw_data['val_output']

                _metric = self.data_handler.get_eval_metric(
                    val_pred * self._get_boost_lr(rid) + val_additive, val_raw
                )

                # NOTE: use evaluation metric to select the option
                metrics.append(
                    (_metric[self.config.boost_loss], (input_norm_cls, output_norm_cls))
                )

            metrics = sorted(metrics)[:int(len(metrics) * 0.75)]
            norm_options = [x[1] for x in metrics]
            print(f'On bandit selection round {t}, selected candidate {norm_options}')

            if len(norm_options) <= 3:
                break

        return norm_options[:1]

    def brute_force_search(self, rid, norm_options=None):
        best_metric = None
        best_pred = None 
        best_boost_lr = None

        if norm_options is None:
            if self.config.use_norm_combine:
                norm_options = list(itertools.product(self.norm_list, self.norm_list))
            else:
                norm_options = [(x, x) for x in self.norm_list]

        for (input_norm_cls, output_norm_cls) in norm_options:
            logger.info(
                f'On boosting round {rid}, Input Normalization {input_norm_cls}, Output Normalization {output_norm_cls}'
            )

            norm_data_dict = self.data_handler.normalized_data_dict(input_norm_cls, output_norm_cls)

            # Experiment class takes the normalized data as input
            experiment = self.exp_cls(
                data_dict=norm_data_dict,
                config=self.config,
                logger=logger,
                rid = rid
            )
            try:
                experiment.run()
                experiment.get_final_predictions()

                prediction_dict = self.data_handler.recoverd_pred_dict(output_norm_cls, experiment)

                metric,boost_lr = self.data_handler.evaluation(prediction_dict,rid)

                logger.info(f'selecte best boost lr {boost_lr}')
                logger.info(f'Train Metric: {metric["train"]}')
                logger.info(f'Val Metric: {metric["val"]}')
                logger.info(f'Test Metric: {metric["test"]}')
                logger.info(f'-' * 12)

                # NOTE: use eval_loss for model selection
                if  best_metric is None:
                    best_pred = copy.deepcopy(prediction_dict)
                    best_metric = copy.deepcopy(metric)
                    best_option = (input_norm_cls, output_norm_cls)
                    best_boost_lr = boost_lr
                #### The greater the accuracy/auc, the better
                elif (self.config.objective == "binary" or self.config.objective == "multi-class")  and metric['val'][self.config.eval_loss[0]] > best_metric['val'][self.config.eval_loss[0]]: 
                    best_pred = copy.deepcopy(prediction_dict)
                    best_metric = copy.deepcopy(metric)
                    best_option = (input_norm_cls, output_norm_cls)
                    best_boost_lr = boost_lr
                ### The smaller the mse/mae ,the better
                elif  (self.config.objective  == "forecasting" or self.config.objective == "regression") and metric['val'][self.config.eval_loss[0]] < best_metric['val'][self.config.eval_loss[0]] :
                    best_pred = copy.deepcopy(prediction_dict)
                    best_metric = copy.deepcopy(metric)
                    best_option = (input_norm_cls, output_norm_cls)
                    best_boost_lr = boost_lr
            except Exception as e:
                logger.error(
                    f'On boosting round {rid}, Input Normalization {input_norm_cls}, Output Normalization {output_norm_cls} , An error occurred '
                )
                logger.error(e)

        return best_pred, best_metric, best_option,best_boost_lr

    def boost(self):
        for rid in range(self.config.boost_rounds):
            # init normalizer
            self.data_handler.reset_norm(rid)

            if self.config.use_bandit_search:
                best_option = self.bandit_search(rid)
            else:
                best_option = None

            best_pred, best_metric, best_option,best_boost_lr = self.brute_force_search(rid, best_option)

            # logger.info(f'On boosting round {rid}, Best Option {best_option},
            logger.info('=' * 80)
            logger.info(f'Boosting round {rid} summary:')
            logger.info(f'Best option {best_option}')
            logger.info(f'Best boost learning rate {best_boost_lr}')
            logger.info(f'Validation metric {best_metric["val"]}')
            logger.info(f'Test metric {best_metric["test"]}')
            logger.info('=' * 80)

            self.data_handler.update_additive(best_pred, lr = best_boost_lr)
