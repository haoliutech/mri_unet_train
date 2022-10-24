# from utils.wandb_analysis_2C_2D import *
from utils.config_utils import read_json, write_json
from utils.results_manager import ResultsManager
# from utils.wandb_utils_v2 import get_wandb_config_from_json, update_config_from_df, compute_metrics_by_run
# from utils.wandb_utils_v2 import get_compare_labellers_df_v3

import os
import shutil

from pathlib import Path

import re
import numpy as np
import pandas as pd

import seaborn as sns
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

import torch
import torch.nn as nn
import nibabel as nib
import torchio as tio


tex_base_path='./FinalReport'
results_path = Path('/Users/hao/MRI_Research_Project/mproj7205/data/report_results/')
multi_SN_RN_path = Path('/Users/hao/MRI_Research_Project/mproj7205/data/multi_SN_RN/npy_2d/SN/')
cwd_path = Path('/Users/hao/MRI_Research_Project/mproj7205')


def compute_metrics_by_df(df, load_from_file=False, save_to_file=True):
    
    def ppv(tp, fp):
        return (tp)/(tp+fp)

    def sensitivity(tp, fn):
        return (tp)/(tp+fn)

    def dscoeff(tp, fn, fp):
        return (2*tp)/(2*tp+fn+fp)

    df_box_fname = results_path / 'df_box.csv'

    if load_from_file and df_box_fname.is_file():
        return pd.read_csv(df_box_fname, index_col=0)

    #df = map_columns(df)

    df_all = None
    
    for idx in df.index:
        df1 = compute_metrics_by_run(df, idx=idx, sweep_name=None)

        df1['ppv'] = df1.apply(lambda x: ppv(x.tp, x.fp), axis=1)
        df1['sensitivity'] = df1.apply(lambda x: sensitivity(x.tp, x.fn), axis=1)
        df1['dscoeff'] = df1.apply(lambda x: dscoeff(x.tp, x.fn, x.fp), axis=1)

        #df1['include_T1_'] = df['include_T1'].replace({False: inlcude_T1_labels[0], True : inlcude_T1_labels[1]})
        df1['include_T1_'] = df.loc[idx]['include_T1_']
        df1['include_T1'] = df.loc[idx]['include_T1']
        df1['loss_function'] = df.loc[idx]['loss_function']
        df1['loss_function_'] = df.loc[idx]['loss_function_']
        df1['network_name'] = df.loc[idx]['network_name']
        df1['network_name_'] = df.loc[idx]['network_name_']
        df1['name'] = df.loc[idx]['name']
        df1['table_name'] = df.loc[idx]['name']
        if 'pre_trained_' in df.keys():
            df1['pre_trained_'] = df.loc[idx]['pre_trained_']
        if 'augmentation' in df.keys():
            df1['augmentation'] = df.loc[idx]['augmentation']
            df1['augmentation_'] = df.loc[idx]['augmentation_']

        if 'beta' in df.keys():
            df1['beta'] = df.loc[idx]['beta']

        if 'loss_function_trim' in df.keys():
            df1['loss_function_trim'] = df.loc[idx]['loss_function_trim']

        if isinstance(df_all, pd.DataFrame):
            df_all = pd.concat([df_all, df1], axis=0, ignore_index=True)
        else:
            df_all = df1
    #df_all =  pd.concat([df_all, df], axis=1, ignore_index=True)

    if save_to_file:         
        df_all.to_csv(df_box_fname, index=True)

    return df_all



def compute_metrics_by_run(df, idx=None, sweep_name=None, phase='validation', cwd_path=None):

    if sweep_name:
        if not sweep_name.endswith('.pt'):
            model_name = f'{sweep_name}.pt'
        else:
            model_name = sweep_name

        idx = df[df['model_name'] == model_name].index.values.astype(int)[0] 
        # print(f'found idx {idx}')
        
    if idx is None:
        print('Must provide valid idx or sweep_name')
        return
    
    project_name = df.loc[idx]['project_name']
    config_fname = df.loc[idx]['config_fname']
    # config = get_wandb_config_from_json(config_fname)
    # config = update_config_from_df(config, df_series = df.loc[idx])

    # wa = WandbAnalysis(project_name=project_name)
    # config['name'] = df.loc[idx]['name']
    # config['model_name'] = df.loc[idx]['model_name']
    # _ = wa.compute_metrics_from_saved_model(config, phase=phase, save_by_patient_id=True, clear_results_df=True)

    # rm = ResultsManager(results_df_path=f'/home/apollo/data/report_results/{phase}_metrics.csv')

    # df_by_run = rm.get_results_df()
    # df_by_run = df_by_run.sort_values(['dsc'])

    # return df_by_run



def compute_metrics_from_saved_model(config, phase='validation', save_by_patient_id=False, **kwargs):

    config['batch_size']=1
    config['valid_batch_size']=1

    plot_title_prefix = kwargs.pop('plot_title_prefix', None)
    filter_by_patients = kwargs.pop('filter_by_patients', [])
    
    network = load_model(config)

    device = config.['device']

    if 'include_T1' in config.keys():
        include_T1 = config.['include_T1']
    else:
        include_T1 = False        
    
    if phase == 'validation':
        _, loader = build_datasets(config, verbose=False)
    elif phase == 'train':
        loader, _ = build_datasets(config, verbose=False)
    else:
        print('Error, phase must be validation (default) or train')
        return

    network.float()

    log_metrics = LogMetrics()
    if save_by_patient_id:
        rm = ResultsManager(results_df_path=f'/Users/hao/MRI_Research_Project/mproj7205/data/report_results/{phase}_metrics.csv')

        clear_results_df = kwargs.pop('clear_results_df', False)
        if clear_results_df:
            rm.clear_results_df()

    is_train=False
    #use_sigmoid = False
    use_sigmoid = get_use_sigmoid(config)
    tp_sum=0
    fp_sum=0
    tn_sum=0
    fn_sum=0
    for _, data in enumerate(loader):

        inputs = data['image'][tio.DATA]
        inputs = np.squeeze(inputs, axis=4)

        if include_T1:
            inputs_T1 = data['image_T1'][tio.DATA]
            inputs_T1 = np.squeeze(inputs_T1, axis=4)
            inputs = torch.cat((inputs, inputs_T1), 1)

        inputs = inputs.to(device)

        labels = data['label'][tio.DATA]
        labels = np.squeeze(labels, axis=4)
        labels = labels.to(device)

        # Forward pass
        with torch.set_grad_enabled(is_train):
            if use_sigmoid:
                predictions = torch.sigmoid(network(inputs))
            else:
                predictions = network(inputs)

            patient_id = data['patient_id'][0]
            if int(patient_id) == 3607:
                continue                

        dsc, (tp, fp, tn, fn) = log_metrics(torch.round(predictions), torch.round(labels))
        dsc, tp, fp, tn, fn = dsc.item(), tp.item(), fp.item(), tn.item(), fn.item()

        tp_sum+=tp
        fp_sum+=fp
        tn_sum+=tn
        fn_sum+=fn

        if save_by_patient_id:        
            rm.add_to_results_df(model_name = f'{config.model_name}', 
                                patient_id=patient_id,
                                phase=phase,
                                dsc=dsc,
                                tp=tp,
                                fp=fp,
                                tn=tn,
                                fn=fn)
        
    return dsc, tp_sum, fp_sum, tn_sum, fn_sum  

