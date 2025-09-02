from omegaconf import OmegaConf
from config.data import DataArgs
from config.models import ModelArgs
from dataclasses import dataclass, field
import logging
from config.checkpoint import CheckpointArgs
from config.metrics import (
    LoggingArgs,
)
from os import path
from config.profiling import ProfilerArgs
from config.optim import OptimArgs
from typing import List
logger = logging.getLogger()
import os

@dataclass
class TrainArgs:
    name: str = "train"
    root_dir: str = "results"
    dump_dir: str = ""
    dataset_folder:str = ""
    seed: int = 666
    eval: bool = True
    num_gpus: int = 1
    torch_gpu_id: int = 1
    gpu_auto_assign: bool = True
    TUNE_MODE: bool = False
    config_id: int = 1 # give ray tuned with a unified id
    data_index: int = -1 # data fold of the dataset
    VARIANT: str= "Default" # for ray tuned
    dump_dirs: List[str] = field(default_factory=lambda: ["Default", "Default", "Default", "Default"])

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: ModelArgs = field(default_factory=ModelArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)



    def update(self):
        self.data.update()
        dataset_folder = f"{self.data.dataset}_{self.data.date}_{self.data.session}_{self.data.des}"
        self.dataset_folder = dataset_folder
        self.dump_dirs = []
        for i_fold in range(self.data.K_fold):
            self.dump_dirs.append(os.path.join(self.root_dir, self.dataset_folder+f"_D{i_fold}", self.model.name))
        self.VARIANT = self.model.name
def prepare_config_from_files(model_yaml:str):
    dataset_path = f"/home/shq/project/SynergyNet/config/{model_yaml}.yaml"
    file_cfg = OmegaConf.load(dataset_path)

    # update config from yaml
    data_cfg = OmegaConf.load(file_cfg.data_cfg_yaml)
    # remove 'data_cfg_yaml' attribute from config as the underlying DataClass does not have it
    del file_cfg.data_cfg_yaml

    # update config from yaml
    optim_cfg = OmegaConf.load(file_cfg.optim_cfg_yaml)
    # remove 'optim_cfg_yaml' attribute from config as the underlying DataClass does not have it
    del file_cfg.optim_cfg_yaml

    # update config from yaml
    model_cfg = OmegaConf.load(file_cfg.model_cfg_yaml)
    # remove 'data_cfg_yaml' attribute from config as the underlying DataClass does not have it
    del file_cfg.model_cfg_yaml


    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, data_cfg, optim_cfg, model_cfg)
    cfg = OmegaConf.to_object(cfg)
    cfg.update()
    return cfg

def prepare_config(model_yaml:str):
    if len(path.split(model_yaml)[0]) > 0:
        dataset_path = model_yaml
    else:
        dataset_path = f"/home/shq/project/SynergyNet/config/train/{model_yaml}.yaml"
    file_cfg = OmegaConf.load(dataset_path)

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg)
    cfg = OmegaConf.to_object(cfg)
    cfg.update()
    return cfg

def get_default():
    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.to_object(default_cfg)
    cfg.update()
    return cfg

