Source code for paper: *Decoding multi-joint hand movements from brain signals by learning a synergy-based neural manifold*
## How to Run: Instructions for Users

### 1. **Download the Dataset**  
Download the dataset and place it in the *Dataset* directory. The dataset is organized into the following three folders:
- **NeuralData**: Contains invasive brain signal recordings from each session.
- **KinData**: Contains joint angle trajectories for 11 hand movements.
- **Info**: Contains session metadata.

### 2. **Modify Configuration Files**  
In the *config* directory, you need to update the following configuration files:

- **Data-related Config**:  
  In the *datasets* section, update the `root_dir` and `raw_root_dir` to the paths where your dataset is stored.

- **Training-related Config**:  
  In the *train* section, update the `root_dir` and `raw_root_dir` to match the location of your dataset. You can also modify other training parameters in this file.

- **Bayesian Hyperparameter Optimization Config**:  
  In the *ray* section, update the following parameters:
  - `default_config_dir`: Specifies the path to the configuration file.
  - `exp_config`: The filename (.yaml) that specifies the default configuration for training and data processing.
  - `tune_hp_json`: The filename (.json) that defines the hyperparameter search space.

### 3. **Run the Code**  
To execute the code, use the `main.py` script. Before running it, ensure you update the Ray configuration name based on the session you are working with.

For example:
```python
main("SynergyNet_ray_SBP220627S1")

```
This runs the SynergyNet model with neural signals (SBP) collected on 2022-06-27 from Session 1. The search space is defined in the file located at *config/ray/SynergyNet_ray_SBP220627S1.yaml*







