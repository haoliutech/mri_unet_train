import os
from pathlib import Path

from utils.config_utils import *
from model.train import train

import pprint

def run_manual_configs(config):
    pprint.pprint(config)
    train(config)


if __name__ == "__main__":

    project_name = 'project_name'

    cwd_path = Path('/Users/hao/MRI_Research_Project/mproj7205')
    data_path = Path( '/Users/hao/MRI_Research_Project/data_organized_1year')
    config_path = Path(f'/Users/hao/MRI_Research_Project/mproj7205/configs/{project_name}/')    
    models_path = Path( '/Users/hao/MRI_Research_Project/mproj7205/data/saved_models')
    
    json_files = os.listdir(config_path)

    for json_file in json_files:

        config_name = config_path / json_file
        if config_name.is_file():
            sweep_config = read_json(config_name)
            sweep_config['project_name'] = project_name
            sweep_config['model_dir'] = str(models_path)
            sweep_config['dataset_dir'] = str(data_path)
            sweep_config['verbose'] = True  

            run_manual_configs(sweep_config)

