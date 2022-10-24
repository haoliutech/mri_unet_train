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

    cwd_path = directory = Path(os.getcwd())
    
    parent_path = cwd_path.parent.absolute()
    data_path = Path(os.path.join(parent_path,"data/data_organized_1year"))
    config_path = Path(os.path.join(cwd_path,'configs/project_name/'))
    models_path =  os.path.join(cwd_path,'data/saved_models')
    
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

