from config.data_utils import *
from typing import List, Any

@dataclass
class DataArgs:
    root_dir: str = "/home/shq/project/SynergyNet/Dataset/NeuralData/processed_format"
    raw_root_dir: str ="/home/shq/project/SynergyNet/Dataset/NeuralData"
    # data_path: str = ""
    data_pathes: str = "" # if multi-session dataset
    date: str = "2022-06-27"
    session: str = "Session_1"
    date_type: str = f"{date}_{session}"


    shift_time: int = 0
    SR: int = 1000 # sample rate
    categories: List[str] = field(default_factory=lambda: ["ThumbAdd", "Index", "Middle", "Ring", "Pinky", "PalmarGrasp-4",
                                                           "PalmarGrasp-3", "Power", "Tripod", "Pinch", "ThumbAbd"])
    categories_order: List[str] = field(default_factory=lambda:["ThumbAbd","ThumbAdd","Index","Middle","Ring","Pinky",
                                                                "PalmarGrasp-3","PalmarGrasp-4","Pinch","Tripod","Power"])
    # color_list: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    colors = sio.loadmat(current_file_path + "/move_dim_signal_colors.mat")
    color_list = np.array(colors["colors"])

    delete_1st_trial: bool = True
    dataset: str = "SBP"
    dataset_type: str = "data_session"
    Kfold_random: bool = False
    D1TF: bool = False
    valid: bool = True
    split_name: str = "K_fold_repeat"

    S2S: bool = False # sequence to sequence decoding

    select_feature: bool = False
    select_feature_method: str = "MI" # mutual information

    # if use one Utah array for analysis
    one_array: bool = False
    array_idx: int = 0

    # model-related
    channelAttention: bool= False

    task:str = "regression" # regression classification
    regression: RegressionArgs = field(default_factory=RegressionArgs)
    KIN: KinArgs = field(default_factory=KinArgs)
    classification: ClassificationArgs = field(default_factory=ClassificationArgs)

    spike_flag: bool = True
    LFP_flag: bool = False
    SPIKE: SpikeArgs = field(default_factory=SpikeArgs)

    # need update
    des: str = "default"
    data_path: str = "default"
    cross_vel: bool = False
    data_split: str = "default"
    delete_1st_trial: bool = True
    K_fold: int = 4
    start_bin: int = 0
    end_bin: int =2400
    kin_data: str = "2400"
    num_repeat: int = 5
    num_block: int = 4
    num_movement: int = 11

    def update(self):
        self.paramRegist_config_split_data_session()
        self.get_sessionInfo()
        self.get_des()
        self.get_data_path()

    def get_data_path(self):
        self.data_path = f"{self.raw_root_dir}/{self.dataset}_align_delays/{self.date}/{self.session}/Cerebus_data_delay_{self.SPIKE.delay}.mat"
    def get_sessionInfo(self):
        # self.cross_vel = False
        config_session = SessinInfo(self.date_type)
        for key, value in config_session.info.items():
            setattr(self, key, value)
    def paramRegist_config_split_data_session(self):
        if self.D1TF:
            data_split, delete_1st_trial, K_fold = self.config_split_data_session_D1TF()
        else:
            data_split, delete_1st_trial, K_fold = self.config_split_data_session()
        self.data_split = data_split
        self.delete_1st_trial = delete_1st_trial
        self.K_fold = K_fold


    def config_split_data_session_D1TF(self):
        config_session = SessinInfo(self.date_type)

        num_repeat = config_session.info.num_repeat
        num_block = config_session.info.num_block
        num_movement = config_session.info.num_movement

        if self.split_name == "K_fold":
            data_split = "K_fold"
            delete_1st_trial = False
            K_folds = num_block * num_repeat

        if self.split_name == "K_fold_repeat":
            data_split = "K_fold"
            delete_1st_trial = False
            K_folds = num_block

        if self.split_name == "LeaveOneOut":
            data_split = "LeaveOneOut"
            delete_1st_trial = False
            K_folds = num_movement

        if self.split_name == "LeaveOneBlockOut":
            data_split = "LeaveOneBlockOut"
            delete_1st_trial = False
            K_folds = num_block

        return data_split, delete_1st_trial, K_folds

    def config_split_data_session(self):
        config_session = SessinInfo(self.date_type)
        num_repeat = config_session.info.num_repeat
        num_block = config_session.info.num_block
        num_movement = config_session.info.num_movement

        if self.split_name == "K_fold":
            data_split = "K_fold"
            delete_1st_trial = True
            K_folds = num_block * (num_repeat - 1)

        if self.split_name == "K_fold_repeat":
            data_split = "K_fold"
            delete_1st_trial = True
            K_folds = num_block

        if self.split_name == "LeaveOneOut":
            data_split = "LeaveOneOut"
            delete_1st_trial = True
            K_folds = num_movement

        if self.split_name == "LeaveOneBlockOut":
            data_split = "LeaveOneBlockOut"
            delete_1st_trial = True
            K_folds = num_block

        return data_split, delete_1st_trial, K_folds

    def get_des(self):
        if self.split_name == "K_fold_9fold":
            des_ori = "KF9_KR_{}_KF_{}_D1st_{}".format(int(self.Kfold_random), self.K_fold, int(self.delete_1st_trial))
        if self.split_name == "K_fold_10fold":
            des_ori = "KF10_KR_{}_KF_{}_D1st_{}".format(int(self.Kfold_random), self.K_fold, int(self.delete_1st_trial))
        if self.split_name == "K_fold_repeat":
            des_ori = "KB_KR_{}_KF_{}_D1st_{}".format(int(self.Kfold_random), self.K_fold, int(self.delete_1st_trial))
        if self.split_name == "LeaveOneOut":
            des_ori = "LOU_D1st_{}".format(int(self.delete_1st_trial))
        if self.split_name == "LeaveOneBlockOut":
            des_ori = "LOBO_D1st_{}".format(int(self.delete_1st_trial))

        if self.select_feature:
            des_ori = "{}_{}_{}".format(des_ori, int(self.select_feature), self.select_feature_method)

        des_spike: str = "SL_{}_DS_{}_WL_{}_SL_{}_SM_{}_CA_{}_STD_{}_SMA_{}_DL_{}_MS_{}_PCA_{}".format(
            int(self.SPIKE.selectchannel),
            int(self.SPIKE.downsample),
            int(self.SPIKE.win_len),
            int(self.SPIKE.step_len),
            int(self.SPIKE.smooth),
            int(self.SPIKE.clip_abnormal),
            int(self.SPIKE.Standardization),
            int(self.SPIKE.smooth_after),
            int(self.SPIKE.mean_sub),
            int(self.SPIKE.delay),
            int(self.SPIKE.PCA)
        )

        if self.LFP_flag:
            des_LFP = "SL_{}".format(self.LFP.step_len)
        else:
            des_LFP = ""

        des_kin = "KS_{}_KT_{}_STD_{}_dim_0_{}".format(self.KIN.kin_space, self.KIN.kin_type,
                                                       int(self.KIN.Standardization), int(self.KIN.dim_0))
        des_regression = "SL_{}_PD_{}".format(self.regression.step_len, int(self.regression.padding))

        if self.classification.aug:
            des_classifiction = "SL_{}".format(self.classification.step_len)
        else:
            des_classifiction = ""
        des: str = ""
        if self.LFP_flag and self.spike_flag:
            des = f"LFP_{des_LFP}_SPIKE_{des_spike}"
        elif self.spike_flag:
            des = f"SPIKE_{des_spike}"
        elif self.LFP_flag:
            des = f"LFP_{des_LFP}"
        if self.task == "regression":
            des = "{}_Kin_{}_{}".format(des, des_kin, des_regression)
        if self.task == "classification":
            des = "{}_{}".format(des, des_classifiction)
        des = "{}_{}".format(des_ori, des)
        self.des = des
        return des


@dataclass
class DataArgs:
    root_dir: str = "/home/shq/project/SynergyNet/Dataset/NeuralData/processed"
    raw_root_dir: str ="/home/shq/project/SynergyNet/Dataset/NeuralData"
    # data_path: str = ""
    data_pathes: str = "" # if multi-session dataset
    date: str = "2022-06-27"
    session: str = "Session_1"
    date_type: str = f"{date}_{session}"


    shift_time: int = 0
    SR: int = 1000 # sample rate
    categories: List[str] = field(default_factory=lambda: ["ThumbAdd", "Index", "Middle", "Ring", "Pinky", "PalmarGrasp-4",
                                                           "PalmarGrasp-3", "Power", "Tripod", "Pinch", "ThumbAbd"])
    categories_order: List[str] = field(default_factory=lambda:["ThumbAbd","ThumbAdd","Index","Middle","Ring","Pinky",
                                                                "PalmarGrasp-3","PalmarGrasp-4","Pinch","Tripod","Power"])
    # color_list: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    colors = sio.loadmat(current_file_path + "/move_dim_signal_colors.mat")
    color_list = np.array(colors["colors"])

    delete_1st_trial: bool = True
    dataset: str = "SBP" # SBP SUA　LFP
    dataset_type: str = "data_session"
    Kfold_random: bool = False
    D1TF: bool = False
    valid: bool = True
    split_name: str = "K_fold_repeat"


    S2S: bool = False # sequence to sequence decoding

    select_feature: bool = False
    select_feature_method: str = "MI" # mutual information


    # if use one Utah array for analysis
    one_array: bool = False
    array_idx: int = 0




    # model-related
    channelAttention: bool= False

    task:str = "regression" # regression classification
    regression: RegressionArgs = field(default_factory=RegressionArgs)
    KIN: KinArgs = field(default_factory=KinArgs)
    classification: ClassificationArgs = field(default_factory=ClassificationArgs)

    spike_flag: bool = True
    LFP_flag: bool = False
    SPIKE: SpikeArgs = field(default_factory=SpikeArgs)

    # need update
    des: str = "default"
    data_path: str = "default"
    cross_vel: bool = False
    data_split: str = "default"
    delete_1st_trial: bool = True
    K_fold: int = 4
    start_bin: int = 0
    end_bin: int =2400
    kin_data: str = "2400"
    num_repeat: int = 5
    num_block: int = 4
    num_movement: int = 11


    def update(self):
        self.paramRegist_config_split_data_session()
        self.get_sessionInfo()
        self.get_des()
        self.get_data_path()

    def get_data_path(self):
        self.data_path = f"{self.raw_root_dir}/{self.dataset}_align_delays/{self.date}_{self.session}.mat"
    def get_sessionInfo(self):
        # self.cross_vel = False
        config_session = SessinInfo(self.date_type)
        for key, value in config_session.info.items():
            setattr(self, key, value)
    def paramRegist_config_split_data_session(self):
        if self.D1TF:
            data_split, delete_1st_trial, K_fold = self.config_split_data_session_D1TF()
        else:
            data_split, delete_1st_trial, K_fold = self.config_split_data_session()
        self.data_split = data_split
        self.delete_1st_trial = delete_1st_trial
        self.K_fold = K_fold


    def config_split_data_session_D1TF(self):
        config_session = SessinInfo(self.date_type)

        num_repeat = config_session.info.num_repeat
        num_block = config_session.info.num_block
        num_movement = config_session.info.num_movement

        if self.split_name == "K_fold":
            data_split = "K_fold"
            delete_1st_trial = False
            K_folds = num_block * num_repeat

        if self.split_name == "K_fold_repeat":
            data_split = "K_fold"
            delete_1st_trial = False
            K_folds = num_block

        if self.split_name == "LeaveOneOut":
            data_split = "LeaveOneOut"
            delete_1st_trial = False
            K_folds = num_movement

        if self.split_name == "LeaveOneBlockOut":
            data_split = "LeaveOneBlockOut"
            delete_1st_trial = False
            K_folds = num_block

        return data_split, delete_1st_trial, K_folds

    def config_split_data_session(self):
        config_session = SessinInfo(self.date_type)
        num_repeat = config_session.info.num_repeat
        num_block = config_session.info.num_block
        num_movement = config_session.info.num_movement

        if self.split_name == "K_fold":
            data_split = "K_fold"
            delete_1st_trial = True
            K_folds = num_block * (num_repeat - 1)

        if self.split_name == "K_fold_repeat":
            data_split = "K_fold"
            delete_1st_trial = True
            K_folds = num_block

        if self.split_name == "LeaveOneOut":
            data_split = "LeaveOneOut"
            delete_1st_trial = True
            K_folds = num_movement

        if self.split_name == "LeaveOneBlockOut":
            data_split = "LeaveOneBlockOut"
            delete_1st_trial = True
            K_folds = num_block

        return data_split, delete_1st_trial, K_folds

    def get_des(self):
        if self.split_name == "K_fold_9fold":
            des_ori = "KF9_KR_{}_KF_{}_D1st_{}".format(int(self.Kfold_random), self.K_fold, int(self.delete_1st_trial))
        if self.split_name == "K_fold_10fold":
            des_ori = "KF10_KR_{}_KF_{}_D1st_{}".format(int(self.Kfold_random), self.K_fold, int(self.delete_1st_trial))
        if self.split_name == "K_fold_repeat":
            des_ori = "KB_KR_{}_KF_{}_D1st_{}".format(int(self.Kfold_random), self.K_fold, int(self.delete_1st_trial))
        if self.split_name == "LeaveOneOut":
            des_ori = "LOU_D1st_{}".format(int(self.delete_1st_trial))
        if self.split_name == "LeaveOneBlockOut":
            des_ori = "LOBO_D1st_{}".format(int(self.delete_1st_trial))

        if self.select_feature:
            des_ori = "{}_{}_{}".format(des_ori, int(self.select_feature), self.select_feature_method)

        des_spike: str = "SL_{}_DS_{}_WL_{}_SL_{}_SM_{}_CA_{}_STD_{}_SMA_{}_DL_{}_MS_{}_PCA_{}".format(
            int(self.SPIKE.selectchannel),
            int(self.SPIKE.downsample),
            int(self.SPIKE.win_len),
            int(self.SPIKE.step_len),
            int(self.SPIKE.smooth),
            int(self.SPIKE.clip_abnormal),
            int(self.SPIKE.Standardization),
            int(self.SPIKE.smooth_after),
            int(self.SPIKE.mean_sub),
            int(self.SPIKE.delay),
            int(self.SPIKE.PCA)
        )

        if self.LFP_flag:
            des_LFP = "SL_{}".format(self.LFP.step_len)
        else:
            des_LFP = ""

        des_kin = "KS_{}_KT_{}_STD_{}_dim_0_{}".format(self.KIN.kin_space, self.KIN.kin_type,
                                                       int(self.KIN.Standardization), int(self.KIN.dim_0))
        des_regression = "SL_{}_PD_{}".format(self.regression.step_len, int(self.regression.padding))

        if self.classification.aug:
            des_classifiction = "SL_{}".format(self.classification.step_len)
        else:
            des_classifiction = ""
        des: str = ""
        if self.LFP_flag and self.spike_flag:
            des = f"LFP_{des_LFP}_SPIKE_{des_spike}"
        elif self.spike_flag:
            des = f"SPIKE_{des_spike}"
        elif self.LFP_flag:
            des = f"LFP_{des_LFP}"
        if self.task == "regression":
            des = "{}_Kin_{}_{}".format(des, des_kin, des_regression)
        if self.task == "classification":
            des = "{}_{}".format(des, des_classifiction)
        des = "{}_{}".format(des_ori, des)
        self.des = des
        return des



if __name__ == "__main__":
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.structured(DataArgs())
    file_path = "spike_20220627S1.yaml"
    file_cfg = OmegaConf.load(file_path)
    default_cfg = OmegaConf.merge(default_cfg, file_cfg)
    cfg = OmegaConf.to_object(default_cfg)
    print(cfg.get_des())
