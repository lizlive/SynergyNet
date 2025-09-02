from dataclasses import dataclass, field
import yaml
import numpy as np
from typing import List
import os
import scipy.io as sio
from omegaconf import OmegaConf
@dataclass
class SessinInfo:
    def __init__(self, data_type):
        self.data_type = data_type
        self.root_dir = "Dataset/Info"
        self.info = self.read_yaml()

    def read_yaml(self):
        file_path = f"{self.root_dir}/{self.data_type}.yaml"
        data = OmegaConf.load(file_path)
        return data


@dataclass
class SpikeArgs:
    selectchannel: bool = False
    downsample: bool = True
    win_len: int = 100
    step_len: int = 20
    smooth: bool = True
    clip_abnormal: bool = True
    Standardization: bool = True
    mean_sub: bool = False
    smooth_after: bool = True
    delay: int = 0
    PCA: bool = False


@dataclass
class ClassificationArgs:
    aug: bool = False
    step_len: int = 5

@dataclass
class KinArgs:
    root_dir: str = "Dataset/KinData"
    # kin_path = root_dir # fix some stupid bug

    kin_space: str = "rotation_angle"  # euler: 48; original_quaternion: 64; quaternion: 64; rotation_angle: 16; rotation_axis: 48
    kin_type: str = "position"  # 'velocity' ; 'position';  'velocity_position'
    dim_0: bool = False
    if kin_space == "rotation_angle":
        if kin_type == "velocity":
            if dim_0:
                dim_name: List[str] = field(default_factory=lambda:["v-0", "1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1",
                                                                    "3-2", "3-3", "4-1", "4-2", "4-3", "5-1", "5-2", "5-3"])
            else:
                dim_name: List[str] = field(default_factory=lambda:["1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1",
                                                                "3-2", "3-3", "4-1", "4-2", "4-3", "5-1", "5-2", "5-3"])
        elif kin_type == "position":
            if dim_0:
                dim_name: List[str] = field(default_factory=lambda:["p-0", "1-1", "1-2", "1-3", "2-1", "2-2", "2-3",
                                                                    "3-1","3-2", "3-3", "4-1", "4-2", "4-3", "5-1", "5-2", "5-3"])
            else:
                dim_name : List[str] = field(default_factory=lambda: ["1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1",
                                                                "3-2", "3-3", "4-1", "4-2", "4-3", "5-1", "5-2", "5-3"])
        elif kin_type == "velocity_position":
            if dim_0:
                dim_name : List[str] = field(default_factory=lambda:["v-0", "1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1",
                                                                     "3-2", "3-3", "4-1", "4-2", "4-3","5-1", "5-2", "5-3",
                                                                     "p-0", "1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1",
                                                                     "3-2", "3-3", "4-1", "4-2", "4-3", "5-1", "5-2", "5-3"])
            else:
                dim_name : List[str] = field(default_factory=lambda:["1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1",
                                                                     "3-2", "3-3", "4-1", "4-2", "4-3", "5-1", "5-2", "5-3",
                                                                     "1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1",
                                                                     "3-2", "3-3", "4-1", "4-2", "4-3","5-1", "5-2", "5-3"])

    Standardization: bool = True


@dataclass
class RegressionArgs:
    flatten_matrix: bool = False
    kin_space: str = "rotation_angle"
    step_len: int = 5
    padding: bool = True
    RNN: bool = True
    des: str = "SL_{}_PD_{}".format(step_len, int(padding))


