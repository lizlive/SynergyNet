from config.train import TrainArgs
import numpy as np
import os
import time
import os.path as osp
from Neural_kits.data_preprocessing import get_dataLoader, visulize_dataloader
from config.logger_wrapper import create_logger
import torch
from typing import Any, Dict, List, Optional
from config.optim import OptimArgs, build_optimizer
from Neural_model_kits.utils import get_model
from Model_kits.utils import count_parameters
from sklearn.metrics import r2_score,mean_squared_error
from scipy.stats import pearsonr
from config.models import update_dataRealted
from config.ray_search import flatten
import pandas as pd
def get_lightest_gpus(num_gpus):
    # TODO update with better CUDA_VISIBLE_DEVICES support (or just use ray)
    if torch.cuda.device_count() == 1:
        return [0]
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argsort(memory_available)[-num_gpus:].tolist()

def criterion_function_class(y_true, y_pre):
    _, preds = y_pre.max(1)
    num_correct = (preds == y_true.view(-1)).sum().item()

    acc = num_correct / (len(y_true))
    criterion = {"acc": acc, "y_pre": preds.cpu().view(-1).numpy(), "y_true": y_true.cpu().view(-1).numpy()}
    return criterion

def regist_cir_prepare(criterion):
    # Create a new dictionary that filters only int or float values
    filtered_data = {key: value for key, value in criterion.items() if isinstance(value, (int, float))}

    return filtered_data


def criterion_function_regression(y_true, y_pre):
    if not type(y_true) is np.ndarray:
        if torch.is_tensor(y_true):
            if y_true.is_cuda:
                y_true = y_true.cpu()
            y_true = y_true.numpy()
    if not type(y_pre) is np.ndarray:
        if torch.is_tensor(y_pre):
            if y_pre.is_cuda:
                y_pre = y_pre.cpu()
            y_pre = y_pre.numpy()
    y_pre[np.isnan(y_pre)] = 0
    r2_dims = r2_score(y_true, y_pre, multioutput='raw_values')
    r2_dims[r2_dims < 0] = 0
    r2 = np.mean(r2_dims)
    cc_dims = []
    rmse_dims = []
    for i_dim in range(y_true.shape[1]):
        if np.all(y_true[:, i_dim] == y_true[0, i_dim]):
            y_true[:, i_dim] = y_true[:, i_dim] + np.random.uniform(0, 1, np.shape(
                y_true[:, i_dim])) * 10e-7
        if np.all(y_pre[:, i_dim] == y_pre[0, i_dim]):
            y_pre[:, i_dim] = y_pre[:, i_dim] + np.random.uniform(0, 1, np.shape(
                y_pre[:, i_dim])) * 10e-7
        cc_tmp = pearsonr(y_true[:, i_dim], y_pre[:, i_dim])
        cc_dims.append(cc_tmp[0])
        rmse_dims.append(np.sqrt(mean_squared_error(y_true[:, i_dim], y_pre[:, i_dim])).astype(np.float64))

    rmse = np.mean(rmse_dims)
    cc = np.mean(np.array(cc_dims))
    criterion = {"r2": r2, "rmse": rmse, "cc": cc, "r2_dims": r2_dims, "y_pre": y_pre, "y_true": y_true}
    return criterion
class Runner:
    def __init__(self, config: TrainArgs):
        self.config = config



    def load_device(self):
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.num_gpus = min(self.config.num_gpus, torch.cuda.device_count())
            self.logger.info(f"Using {self.num_gpus} GPUs")
            gpu_id = self.config.torch_gpu_id
            if self.config.gpu_auto_assign:
                gpu_id = get_lightest_gpus(1)[0]
            self.device = (
                torch.device("cuda", gpu_id)
            )
            self.device_gpu = gpu_id

        self.logger.info(f"Using {self.device}")
    def initial_fold(self):
        self.dump_dir = self.config.dump_dir
        if not osp.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        logfile_path = osp.join(self.dump_dir, f"{self.config.VARIANT}.log")
        self.logger = create_logger(self.config)
        self.logger.clear_filehandlers()
        self.logger.add_filehandler(logfile_path)
        if hasattr(self.config, "TUNE_MODE") and self.config.TUNE_MODE:
            self.logger.clear_streamhandlers()
        self.config.checkpoint.path = os.path.join(self.dump_dir, "checkpoints")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.pth_time = 0
        self.count_updates = 0
        self.count_checkpoints = 0
        self.num_gpus = 0
        self.tuning_metric = {} # metric for selecting best model in tuning algorithm
        self.log_interval = self.config.optim.log_interval
        self.val_interval = self.config.optim.val_interval
        self.patience = self.config.optim.patience
        self.best_val = {
            "value": 1e9,
            "update": -1,
        }
        self.best_R2 = {
            "value": -100,
            "update": -1,
        }
        self.best_R2 = {
            "value": -100,
            "update": -1,
        }
        self.best_CC = {
            "value": -100,
            "update": -1,
        }
        self.best_RMSE = {
            "value": 1e9,
            "update": -1,
        }
        self.test_R2 = 0
        self.test_CC = 0
        self.test_RMSE = 0
        self.count_updates = 0
        self.criterion_epoch = []

    def load_model(self):
        self.logger.info(f"Building model")
        model = get_model(self.config)
        model.to(self.device)
        self.logger.info(f"Model is built !")
        count_parameters(model)
        self.model = model



    # unfihsed: test if work
    def update_config(self, config):
        r""" Update config node and propagate through model. Used for pbt.
        """
        self.config = config
        self.config = update_dataRealted(self.data_loader["trainLoader"], self.config)

        torch.manual_seed(self.config.seed)
        self.load_model()
        self.load_optimizer()
        # self.model.update_config(config.MODEL)
    def load_optimizer(self):
        self.logger.info("Starting build of optimizer...")
        self.optimizer, self.scheduler = build_optimizer(self.model, self.config.optim, self.config.optim.total_epoch)
        self.logger.info(f"Optimizer is built !")

    def setup_model(self):
        model = get_model(self.config)
        model.to(self.device)
        count_parameters(model)
        self.model = model


    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.
        Will fully load model if not already configured. Expects runner devices to be set.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        ckpt_dict = torch.load(checkpoint_path, *args, **kwargs)
        if self.model is None:
            self.setup_model()
        self.model.load_state_dict(ckpt_dict["state_dict"])
        if "optim_state" in ckpt_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt_dict["optim_state"])
        if "scheduler" in ckpt_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt_dict["scheduler"])
        if "best_R2" in ckpt_dict:
            self.best_R2 = ckpt_dict["best_R2"]
        if "best_CC" in ckpt_dict:
            self.best_CC = ckpt_dict["best_CC"]
        if "best_RMSE" in ckpt_dict:
            self.best_RMSE = ckpt_dict["best_RMSE"]
        if "test_R2" in ckpt_dict:
            self.test_R2 = ckpt_dict["test_R2"]
        if "test_CC" in ckpt_dict:
            self.test_CC = ckpt_dict["test_CC"]
        if "test_RMSE" in ckpt_dict:
            self.test_RMSE = ckpt_dict["test_RMSE"]
        if "extra_state" in ckpt_dict:
            self.count_updates = ckpt_dict["extra_state"]["update"]
            self.logger.info("Update loaded -- {}".format(self.count_updates))
            self.count_checkpoints = ckpt_dict["extra_state"]["checkpoint"]
            self.pth_time = ckpt_dict["extra_state"]["pth_time"]
        return ckpt_dict
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optim_state": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "config": self.config,
            "best_val": self.best_val,
            "best_R2": self.best_R2,
            "best_CC": self.best_CC,
            "best_RMSE": self.best_RMSE,
            "test_R2": self.test_R2,
            "test_CC": self.test_CC,
            "test_RMSE": self.test_RMSE,
        }
        checkpoint["extra_state"] = dict( # metadata
            update=self.count_updates,
            checkpoint=self.count_checkpoints,
            pth_time=self.pth_time,
        )

        if extra_state is not None:
            checkpoint["extra_state"].update(extra_state)

        if len(osp.split(file_name)[0]) > 0:
            full_path = file_name
        else:
            os.makedirs(self.config.checkpoint.path, exist_ok=True)
            full_path = osp.join(self.config.checkpoint.path, file_name)
        torch.save(
            checkpoint, full_path
        )

    def regist_training(self):
        # Convert list of dictionaries to pandas DataFrame
        df = pd.DataFrame(self.criterion_epoch)
        os.makedirs(self.dump_dir, exist_ok=True)
        data_path = osp.join(self.dump_dir, f"{self.config.VARIANT}_criterion_epoch.csv")
        # Save the DataFrame to a CSV file
        df.to_csv(data_path, index=True)

    def average_dict_list(self, dict_list):
        # Initialize a dictionary to store averages for float values and same values for others
        average_dict = {}

        # Initialize a dictionary to store the types of each key
        types_dict = {}

        # Iterate through each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                # Track the data type for each key
                if key not in types_dict:
                    types_dict[key] = set()  # Use a set to track unique types for each key
                types_dict[key].add(type(value))

                # Initialize the key in average_dict if not already present
                if key not in average_dict:
                    if isinstance(value, (int, float)):  # If it's numeric, start with 0
                        average_dict[key] = {'sum': 0, 'count': 0}
                    else:
                        average_dict[key] = {'value': value}  # If non-numeric, keep the first value

                # For numeric types (int or float), accumulate the sum and count for averaging
                if isinstance(value, (int, float)):
                    average_dict[key]['sum'] += value
                    average_dict[key]['count'] += 1
                else:
                    # For non-numeric types, we just keep the first encountered value (or custom logic)
                    if 'value' not in average_dict[key]:
                        average_dict[key]['value'] = value

        # Now compute the averages for numeric values
        for key, value in average_dict.items():
            if 'sum' in value and 'count' in value:
                # Compute the average for float or int values
                if value['count'] > 0:
                    average_dict[key] = value['sum'] / value['count']
            else:
                # Keep the non-numeric value as is
                average_dict[key] = value['value']

        # Now check the data types of each key across all dictionaries
        return average_dict

    def train(self, checkpoint_pathes=None) -> None:
        data_loaders, Data_recover = get_dataLoader(self.config.data)
        metrics_list = []
        for i in range(len(data_loaders)):
            cfg = self.config

            self.data_index = i
            self.config.dump_dir = self.config.dump_dirs[self.data_index]
            self.initial_fold()
            self.load_device()
            self.logger.info(f"Starting job: {cfg.name}")

            if checkpoint_pathes is not None:
                self.load_checkpoint(checkpoint_pathes[i], map_location=self.device)

            # data related
            data_loader = data_loaders[self.data_index]
            self.data_loader = data_loader

            # model related
            cfg = update_dataRealted(data_loader["trainLoader"], cfg)
            torch.manual_seed(cfg.seed)
            self.load_model()

            # optimization related
            self.load_optimizer()
            self.model.device = self.device

            start_updates = self.count_updates

            for update in range(start_updates, cfg.optim.total_epoch):
                metrics = self.train_epoch()
                if metrics["done"]:
                    break
                torch.cuda.empty_cache()
            if not metrics["done"]:
                self.logger.info("Reached max updates without early stopping. Consider training some more.")
                self.logger.info(
                    "Best val: {:.4f} at {} updates.".format(self.best_val['value'], self.best_val['update']))
                self.logger.info("Best R2: {:.4f} at {} updates.".format(self.best_R2['value'], self.best_R2['update']))
                metrics["done"] = True
            torch.cuda.empty_cache()
            metrics_list.append(metrics)
            self.regist_training()
        return self.average_dict_list(metrics_list)

    def eval_model(self, data_loader):
        args = self.config
        self.model.eval()
        data = data_loader[list(range(len(data_loader)))]
        x = data["X_data"].to(self.device)
        y_true = data["Y_data"].to(self.device)
        y_dis = data["Label_data"].to(self.device)
        y_dis_process = data["TrialTimeIdx"].to(self.device)
        with torch.no_grad():
            y_pre, _ = self.model(x)  # model返回的是（bs,num_classes）和weight
            eval_loss = self.model(x, y_true, y_dis, y_dis_process)
            # split position for evaluation`
            if self.config.data.KIN.kin_type == "velocity_position":
                n_dim = y_true.shape[-1]//2
                y_true = y_true[:,n_dim:]
                y_pre =y_pre[:,n_dim:]
            if args.data.task == "regression":
                criterion = criterion_function_regression(y_true, y_pre)
            elif args.data.task == "classification":
                criterion = criterion_function_class(y_true, y_pre)
        return eval_loss, criterion
    def train_epoch(self):
        data_loader = self.data_loader
        cfg = self.config
        train_loader, test_loader = data_loader["trainLoader"], data_loader["testLoader"]
        if cfg.data.valid:
            valid_loader = data_loader["validLoader"]
        visulize_dataloader(train_loader)

        self.model.train()

        t_start = time.time()
        permutation = torch.randperm(len(train_loader))
        for i_batch in range(0, len(train_loader), cfg.optim.batch_size):
            indices = permutation[i_batch:i_batch + cfg.optim.batch_size]
            batch = train_loader[indices]
            x = batch["X_data"].to(self.device)
            y = batch["Y_data"].to(self.device)
            y_dis = batch["Label_data"].to(self.device)
            y_dis_process = batch["TrialTimeIdx"].to(self.device)
            loss = self.model(x, y, y_dis, y_dis_process)
            loss.backward(retain_graph=True)

            # For logging we undo that scaling
            loss = loss.detach()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=cfg.optim.clip, foreach=True
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.pth_time += time.time() - t_start
        self.count_updates += 1
        update = self.count_updates

        self.scheduler.step()
        self.logger.queue_stat("LR", self.scheduler.get_last_lr()[0])
        self.logger.queue_stat("loss", loss.item())

        metrics_dict = dict(
            done = False,
            epoch = self.count_updates,
            best_loss=self.best_val["value"],  # Tune will reference this value to select best model.
            best_R2 = self.best_R2["value"], # Tune will reference this value to select best model.
            best_CC = self.best_CC["value"],
            best_RMSE = self.best_RMSE["value"],
            test_CC=self.test_CC,
            test_RMSE=self.test_RMSE,
            test_R2=self.test_R2,
        )

        torch.cuda.empty_cache()
        train_loss, criterion_train = self.eval_model(train_loader)
        self.logger.log_criterion(update, criterion_train, prefix = "train")
        eval_loss, criterion = self.eval_model(valid_loader)
        self.logger.log_criterion(update, criterion, prefix = "valid")
        test_loss, criterion_test = self.eval_model(test_loader)
        self.logger.log_criterion(update, criterion_test, prefix = "test")
        # regist all criterion during training in a csv
        cri_regist = {}
        cri_regist["train"] = regist_cir_prepare(criterion_train)
        cri_regist["test"] = regist_cir_prepare(criterion_test)
        cri_regist["valid"] = regist_cir_prepare(criterion)
        self.criterion_epoch.append(flatten(cri_regist))

        if eval_loss.item() < self.best_val["value"]:
            self.logger.info(
                "Overwriting best loss {:.4f} from {} with {:.4f} at {}.".format(
                    self.best_val['value'], self.best_val['update'], eval_loss.item(), update))
            self.best_val["value"] = eval_loss.item()
            self.best_val["update"] = update
            self.save_checkpoint(f'{self.config.VARIANT}.lve.pth')


        r2 = criterion["r2"]
        if r2 > self.best_R2["value"]:
            self.logger.info(
                "Overwriting best r2 {:.4f} from {} with {:.4f} at {}.".format(
                    self.best_R2['value'], self.best_R2['update'], r2.item(), update))
            self.best_R2["value"] = r2
            self.best_R2["update"] = update
            self.test_R2 = criterion_test["r2"]
            self.test_CC = criterion_test["cc"]
            self.test_RMSE = criterion_test["rmse"]
            self.save_checkpoint(f'{self.config.VARIANT}.r2.pth')

        cc = criterion["cc"]
        if cc > self.best_CC["value"]:
            self.logger.info(
                "Overwriting best cc {:.4f} from {} with {:.4f} at {}.".format(
                    self.best_CC['value'], self.best_CC['update'], cc.item(), update))
            self.best_CC["value"] = cc
            self.best_CC["update"] = update
            # self.save_checkpoint(f'{self.config.VARIANT}.cc.pth')

        rmse = criterion["rmse"]
        if rmse < self.best_RMSE["value"]:
            self.logger.info(
                "Overwriting best rmse {:.4f} from {} with {:.4f} at {}.".format(
                    self.best_RMSE['value'], self.best_RMSE['update'], rmse.item(), update))
            self.best_RMSE["value"] = rmse
            self.best_RMSE["update"] = update
            # self.save_checkpoint(f'{self.config.VARIANT}.rmse.pth')

        if update - self.best_val["update"] > cfg.optim.patience and update - self.best_R2["update"] > cfg.optim.patience:
            self.logger.info(
                f"Val loss or R2 has not improved for {cfg.optim.patience} updates. Stopping...")
            self.logger.info("Best val: {:.4f} at {} updates.".format(self.best_val['value'], self.best_val['update']))
            self.logger.info("Best R2: {:.4f} at {} updates.".format(self.best_R2['value'], self.best_R2['update']))
            metrics_dict["done"] = True

        metrics_dict["best_loss"] = self.best_val["value"]
        metrics_dict["best_R2"] = self.best_R2["value"]
        metrics_dict["best_CC"] = self.best_CC["value"]
        metrics_dict["best_RMSE"] = self.best_RMSE["value"]
        metrics_dict["test_R2"] = self.test_R2
        metrics_dict["test_CC"] = self.test_CC
        metrics_dict["test_RMSE"] = self.test_RMSE

        self.logger.log_update(update)
        self.logger.info(
            "update: {}\tpth-time: {:.3f}s\t".format(
                update, self.pth_time))


        if not cfg.TUNE_MODE: # Don't save extra checkpoints when sweeping
            self.save_checkpoint(f'{self.config.VARIANT}.{self.count_checkpoints}.pth')
            self.count_checkpoints += 1
        torch.cuda.empty_cache()
        return metrics_dict
