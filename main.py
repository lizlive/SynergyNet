import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from config.ray_search import prepare_config as prepare_config_ray
from config.train import prepare_config as prepare_config_train
from config.ray_search import raySearchArgs
from os import path
import torch
import ray, yaml, shutil
from ray import tune
from ray.tune.suggest import skopt
import json
from omegaconf import OmegaConf
import os
import pandas as pd
import uuid
from config.tune_models import tuneModelFolds
from config.ray_search import flatten
from Result_analysis_kits.parameter_vis import parallel_coordinates
def build_hp_dict(raw_json: dict):
    hp_dict = {}
    for key in raw_json:
        info: dict = raw_json[key]
        sample_fn = info.get("sample_fn", "uniform")

        if sample_fn == "odd_int":
            assert "low" in info, "high" in info
            hp_dict[key] = tune.choice([i for i in range(info['low'], info['high']+1) if i % 2 == 1])
        else:
            assert hasattr(tune, sample_fn)
            if sample_fn == "choice":
                hp_dict[key] = tune.choice(info['opts'])
            else:
                assert "low" in info, "high" in info
                sample_fn = getattr(tune, sample_fn)
                if "q" in info:
                    hp_dict[key] = sample_fn(info['low'], info['high'], info['q'])
                else:
                    hp_dict[key] = sample_fn(info['low'], info['high'])
    return hp_dict

def check_standardColumn(pbt_dir_in):
    default_standard_columns = [
        "done", "epoch", "best_loss", "best_R2", "best_CC", "best_RMSE",
        "test_CC", "test_RMSE", "test_R2", "timesteps_total", "episodes_total",
        "training_iteration", "trial_id", "experiment_id", "date", "timestamp",
        "time_this_iter_s", "time_total_s", "pid", "hostname", "node_ip",
        "time_since_restore", "timesteps_since_restore", "iterations_since_restore"
    ]
    for subdir in os.listdir(pbt_dir_in):
        subdir_path = os.path.join(pbt_dir_in, subdir)
        progress_file = os.path.join(subdir_path, 'progress.csv')

        if os.path.exists(progress_file):
            try:
                df_temp = pd.read_csv(progress_file)  
                if not df_temp.empty:  
                    standard_columns = df_temp.columns.tolist()
                    print("找到标准列名")
                    return standard_columns
            except Exception as e:
                print(f"无法读取 {progress_file}: {e}")
    print("使用预定义的标准列名")
    return default_standard_columns


def launch_search(cfg:raySearchArgs):
    train_cfg = prepare_config_train(cfg.exp_config)
    cfg.root_dir = path.join(cfg.root_dir, train_cfg.dataset_folder)
    variant_name = train_cfg.model.name + "_oneFile"
    variant_name = variant_name + "_lite"
    name = variant_name
    pbt_dir = path.join(cfg.root_dir, name)
    pbt_dataset_dirs = []
    for i_fold in range(train_cfg.data.K_fold):
        pbt_dataset_dirs.append(os.path.join(pbt_dir, f"D{i_fold}"))
    train_cfg.dump_dirs = pbt_dataset_dirs

    # ---------- PBT RUN CONFIGURATION ----------
    num_workers = cfg.workers if  cfg.workers > 0 else int(torch.cuda.device_count() //  cfg.gpus_per_worker)
    # the resources to allocate per model
    resource_per_trial = {"cpu": cfg.cpus_per_worker, "gpu": cfg.gpus_per_worker}
    def train(train_cfg):
        if path.exists(pbt_dir):
            if cfg.overwrite:
                print("Run exists!!! Overwriting.")
                if path.exists(pbt_dir):
                    shutil.rmtree(pbt_dir)

        try:
            # the hyperparameter space to search
            with open(cfg.tune_hp_json) as f:
                raw_hp_json = json.load(f)
            cfg_samples = build_hp_dict(raw_hp_json)

            flat_cfg_dict = flatten(OmegaConf.to_container(OmegaConf.structured(train_cfg), resolve=True))

            flat_cfg_dict.update(cfg_samples)

            reporter = tune.CLIReporter(metric_columns=cfg.logged_columns)
            search_alg = skopt.SkOptSearch(metric='best_R2', mode='max')

            # connect to Ray cluster or start on single machine
            address = None if cfg.single_machine else 'localhost:6379'
            ray.init(address=address)

            tune.run(
                tuneModelFolds,
                name=name,
                local_dir=pbt_dir,
                stop={'done': True},
                # stop={"training_iteration": 2},
                config=flat_cfg_dict,
                resources_per_trial=resource_per_trial,
                num_samples=cfg.samples ,
                search_alg=search_alg,
                verbose=1,
                progress_reporter=reporter,
                resume="AUTO",
                max_failures=3
            )

        finally:
            torch.cuda.empty_cache()  
            ray.shutdown()

    if not cfg.eval_only:
        train(train_cfg)

    # load the results dataframe for this run
    analysis = tune.ExperimentAnalysis(pbt_dir)
    df = analysis.dataframe()  
    # exclude the best_model result
    df = df[df.logdir.apply(lambda path: not 'best_model' in path)]
    # check the data type, if it's a string(0), convert to float
    if df[cfg.best_model_metric].dtype == 'O':
        df = df.assign(best_R2=lambda df: df[cfg.best_model_metric].str[7:13].astype(float))

    best_model_logdir = df.loc[df[cfg.best_model_metric].idxmax()].logdir

    # Set destination for the best model
    best_model_dest = path.join(pbt_dir, 'best_model')
    if path.exists(best_model_dest):
        shutil.rmtree(best_model_dest)

    # Copy the best model to the destination
    shutil.copytree(best_model_logdir, best_model_dest)

    df.to_csv(os.path.join(best_model_dest, "all_results.csv"), index=True, header=True, encoding='utf-8')

    # save the best model's results in a csv file
    best_model_results = df.loc[df[cfg.best_model_metric].idxmax()]
    best_model_results = pd.DataFrame(best_model_results).T
    best_model_results.to_csv(os.path.join(best_model_dest, "best_model_results.csv"), index=True, header=True,
                                encoding='utf-8')


    # show parallel results
    metric_show = "test_R2"
    parallel_coordinates(metric_show, cfg, df, best_model_dest)

def main(dataset: str):
    cfg = prepare_config_ray(dataset)
    launch_search(cfg)

if __name__ == "__main__":
    
    main("SynergyNet_ray_SBP220627S1")



