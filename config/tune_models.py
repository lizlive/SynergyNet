import os
import os.path as osp
import numpy as np
import ray
from Neural_kits.data_preprocessing import get_dataLoader, visulize_dataloader
from ray import tune
from yacs.config import CfgNode as CN
import torch
import uuid
from runner import Runner
from config.ray_search import unflatten
from config.models import update_dataRealted
from config.train import get_default
from omegaconf import OmegaConf
class tuneModel(tune.Trainable):
    def setup(self, config):
        yacs_cfg = self.convert_tune_cfg(config)
        self.original_config = yacs_cfg.copy()
        self.epochs_per_generation = yacs_cfg.optim.total_epoch
        self.runner = Runner(config=yacs_cfg)
        self.runner.initial_fold()
        self.runner.load_device()
        self.runner.logger.info(f"Starting job: {self.runner.config.name}")
        data_loaders, Data_recover = get_dataLoader(self.runner.config.data)
        data_loader = data_loaders[yacs_cfg.data_index]
        self.runner.data_loader = data_loader
        # model related
        self.runner.config = update_dataRealted(data_loader["trainLoader"], self.runner.config)
        torch.manual_seed(self.runner.config.seed)
        self.runner.load_model()
        # optimization related
        self.runner.load_optimizer()

    def step(self):
        for i in range(self.runner.config.optim.total_epoch):
            metrics = self.runner.train_epoch()
            if metrics['done']:
                return metrics
        return metrics

    def save_checkpoint(self, tmp_ckpt_dir):
        path = osp.join(tmp_ckpt_dir, f"{self.runner.config.VARIANT}.{self.runner.count_checkpoints}.pth")
        self.runner.save_checkpoint(path)
        return path

    def load_checkpoint(self, path):
        self.runner.load_checkpoint(path)

    def reset_config(self, new_config):
        ori_cfg_node = self.original_config
        new_cfg_node = self.convert_tune_cfg(new_config)
        new_cfg_node = new_cfg_node.merge_from_other_cfg(ori_cfg_node)
        new_cfg_node.config_id = uuid.uuid4()
        self.runner.update_config(new_cfg_node)
        return True

    def convert_tune_cfg(self, flat_cfg_dict):
        flat_cfg_dict['TUNE_MODE'] = True
        cfg_update = CN(unflatten(flat_cfg_dict))

        return cfg_update


class tuneModelFolds(tune.Trainable):
    def setup(self, config):
        yacs_cfg = self.convert_tune_cfg(config)
        self.original_config = yacs_cfg.copy()
        self.epochs_per_generation = yacs_cfg.optim.total_epoch
        self.runner = Runner(config=yacs_cfg)

    def step(self):
        metrics_ave = self.runner.train()
        if metrics_ave['done']:
            return metrics_ave

    def save_checkpoint(self, tmp_ckpt_dir):
        self.runner.save_checkpoint(tmp_ckpt_dir)
        return tmp_ckpt_dir

    def load_checkpoint(self, path):
        self.runner.load_checkpoint(path)

    def reset_config(self, new_config):
        ori_cfg_node = self.original_config
        new_cfg_node = self.convert_tune_cfg(new_config)
        new_cfg_node = new_cfg_node.merge_from_other_cfg(ori_cfg_node)
        new_cfg_node.config_id = uuid.uuid4()
        self.runner.update_config(new_cfg_node)
        return True

    def convert_tune_cfg(self, flat_cfg_dict):
        flat_cfg_dict['TUNE_MODE'] = True
        cfg_update = CN(unflatten(flat_cfg_dict))
        return cfg_update