# imports as in "Peds Adative.ipynb"
import re
import os
import glob
import json
import pickle
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
import trajectron.visualization as visualization
import trajdata.visualization.vis as trajdata_vis

from torch import optim, nn
from torch.utils import data
from tqdm.notebook import tqdm
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.model_utils import UpdateMode
from trajectron.model.trajectron import Trajectron
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Final, List, Optional, Union
from trajdata import UnifiedDataset, AgentType, AgentBatch

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


TRAJDATA_CACHE_DIR: Final[str] = "/home/Projects/RobotSocialNavigation/.unified_data_cache"
ETH_UCY_RAW_DATA_DIR: Final[str] = "/home/Projects/RobotSocialNavigation/datasets/eth_ucy_peds"

# load Trajectron model
def load_model(model_dir: str, device: str, epoch: int = 10, custom_hyperparams: Optional[Dict] = None):
    while epoch > 0:
        save_path = Path(model_dir) / f'model_registrar-{epoch}.pt'
        if save_path.is_file():
            break
        epoch -= 1

    model_registrar = ModelRegistrar(model_dir, device)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
        
    if custom_hyperparams is not None:
        hyperparams.update(custom_hyperparams)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment()
    trajectron.set_annealing_params()

    checkpoint = torch.load(save_path, map_location=device)
    trajectron.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return trajectron, hyperparams

if __name__=="__main__":

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(0)
    else:
        device = 'cpu'

    history_sec = 2.8
    prediction_sec = 4.8

    base_checkpoint = 100

    base_model = glob.glob("../experiments/pedestrians/kf_models/eth_1mode_base_tpp-10_Nov_2024_21_41_22")[0]

    base_trajectron, _ = load_model(base_model, device, epoch=base_checkpoint,
                            custom_hyperparams={"trajdata_cache_dir": TRAJDATA_CACHE_DIR,
                            "single_mode_multi_sample": False})
    
    print(base_trajectron)
