from dataclasses import dataclass, field
from typing import List, Any
from omegaconf import OmegaConf

@dataclass
class raySearchArgs:
    name: str = "ray_search"
    root_dir: str = f"ray_results"

    pbt_metric: str = "best_loss"
    best_model_metric: str = "best_R2"
    logged_columns: List[str] = field(default_factory=lambda: ['best_R2', 'test_R2', 'test_CC', 'test_RMSE'])
    model_yaml: str = "SynergyNet_oneFile"
    exp_config: str =  f"/home/shq/project/SynergyNet/config/train/{model_yaml}.yaml"
    tune_hp_json: str = f"/home/shq/project/SynergyNet/config/train/{model_yaml}.json"
    default_config_dir: str = f"/home/shq/project/SynergyNet/config/train"
    eval_only: bool = False
    overwrite: bool = False
    single_machine: bool = True # whether to use single machine or cluster

    gpus_per_worker: float = 1
    cpus_per_worker: float = 4.0
    workers: int = -1 # -1 indicates -- use max possible workers on machine (assuming 0.5 GPUs per trial)
    samples: int = 120 # samples for random search

    seed: int = 666 # seed for config




    def update(self):
        pass


def prepare_config(model_yaml:str , ymal_path=None):
    if ymal_path is None:
        ymal_path = "config/ray"
    dataset_path = f"{ymal_path}/{model_yaml}.yaml"
    file_cfg = OmegaConf.load(dataset_path)

    default_cfg = OmegaConf.structured(raySearchArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg)
    cfg = OmegaConf.to_object(cfg)
    cfg.update()
    return cfg


def flatten(dictionary, level=[]):
    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict['.'.join(level + [key])] = val
    return tmp_dict


def unflatten(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict