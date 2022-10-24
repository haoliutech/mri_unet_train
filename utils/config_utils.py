import os
import json
from pathlib import Path
import re

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        json_dict = json.load(handle)

    json_dict = _convert_known_tuples(json_dict)
    return json_dict

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def _convert_known_tuples(json_dict):

        # **************************
    try:        
        s = json_dict['parameters']['RandomAffine']['value']['translation']
        regex = re.search(r'\((.*?)\)',s)
        if regex:
            regex.group(1)
            tuple_ = tuple(map(int, regex.group(1).replace(' ', '').split(',')))
            json_dict['parameters']['RandomAffine']['value']['translation'] = tuple_
    except:
        pass

    # **************************
    try:    
        s = json_dict['parameters']['RandomElasticDeformation']['value']['max_displacement']
        regex = re.search(r'\((.*?)\)',s)
        if regex:
            regex.group(1)
            tuple_ = tuple(map(int, regex.group(1).replace(' ', '').split(',')))

            json_dict['parameters']['RandomElasticDeformation']['value']['max_displacement'] = tuple_
    except:
        pass

    # **************************
    try:
        s = json_dict['parameters']['RandomBlur']['value']['std']
        regex = re.search(r'\((.*?)\)',s)
        if regex:
            regex.group(1)
            tuple_ = tuple(map(int, regex.group(1).replace(' ', '').split(',')))

            json_dict['parameters']['RandomBlur']['value']['std'] = tuple_
    except:
        pass

    # **************************
    try:
        s = json_dict['parameters']['RandomBiasField']['value']['coefficients']
        regex = re.search(r'\((.*?)\)',s)
        if regex:
            regex.group(1)
            tuple_ = tuple(map(float, regex.group(1).replace(' ', '').split(',')))

            json_dict['parameters']['RandomBiasField']['value']['coefficients'] = tuple_
    except:
        pass


    return json_dict



# def read_config_and_run(run_count=1):

#     project_name='delete_me'
#     wandb_entity='apollo'
#     sweep_config = read_json('test.xml')
#     sweep_id = wandb.sweep(sweep_config, project=project_name, entity=wandb_entity)
#     wandb.agent(sweep_id, train, count=1)
