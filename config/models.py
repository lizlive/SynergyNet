from dataclasses import dataclass, field
from typing import List
import torch
@dataclass
class ModelArgs:
    # SynergyNet related
    name: str = "SynergyNet"
    input_dim: int = 196
    intermediate_dim: int = 96
    timeConvLen: int = 51
    embedding_dim: int = 64
    output_dim: int = 15
    n_step: int = 51
    step_len: int = 20
    kin_intermediate_dim: int = 25
    kin_dictionary_num: int = 15
    kin_dictionary_timeNum: int = 20
    alpha: float = 0.5
    beta_smooth: float = 10.0
    beta_diversity: float = 0.01


def update_dataRealted(data, cfg):
    if cfg.model.name == "SynergyNet":
        sample = data[0]
        X_data_sample = sample["X_data"]
        data_shape = X_data_sample.shape
        sample = data[list(range(len(data)))]
        Y_data_sample = sample["Y_data"]
        if cfg.data.task == "regression":
            num_classes = Y_data_sample.shape[1]
        elif cfg.data.task == "classification":
            num_classes = torch.max(Y_data_sample).item() + 1
        cfg.model.input_dim = data_shape[0]
        cfg.model.output_dim = num_classes
        cfg.model.n_step = data_shape[1]

    return cfg




