import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

from model.train import *

#from utils.analysis_2C_2D import *
from utils.config_utils import read_json, write_json
from utils.results_manager import ResultsManager
from model.train import *


tex_base_path='./FinalReport'
results_path = Path('/home/apollo/data/report_results/')
multi_SN_RN_path = Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/')
cwd_path = Path('/home/apollo/code/mproj7205')
project_configs_path = Path('/home/apollo/data/saved_configs/')

def create_project_df(project_name='project_name', df=None):

    configs_path = project_configs_path /  f'{project_name}'

    json_files = os.listdir(configs_path)
    for json_file in json_files:

        config_name = configs_path / json_file
        updates_={}
        if config_name.is_file():
            sweep_config = read_json(config_name)   
            for k, v in sweep_config.items():
                if k == 'loss_criterion':
                    if 'beta' in v.keys():
                        updates_['beta'] = v['beta']
                    updates_['loss_function'] = v['loss_function']

                    
                if isinstance(v, dict) or isinstance(v, list):
                    sweep_config[k] = str(v)
        sweep_config.update(updates_)

        df_run = pd.DataFrame.from_dict(sweep_config, orient='index').T

        # Add any standard columns needed here
        model_name = df_run['model_name']
        df_run['name'] = model_name.replace('.pt','').replace('-', '_')
        df_run['config_fname'] = str(config_name)
        df_run['project_name'] = project_name

        if isinstance(df, pd.DataFrame):
            df = pd.concat([df, df_run], ignore_index=True)
        else:
            # df was None, so create the df (first row=df_run)
            df = df_run

        if 'augmentation' not in df.keys():
            df['augmentation'] = False
        else:
            df['augmentation'] = df['augmentation'].fillna(value=False)
            
        replace_dict = {False: 'No augmentation', True : 'Augmentation'}
        df['augmentation_'] = df['augmentation'].replace(replace_dict)

    return df

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

    df = map_columns(df)

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

    config = read_json(config_fname) 
    _ = compute_metrics_from_saved_model(config, phase=phase, save_by_patient_id=True, clear_results_df=True)

    rm = ResultsManager(results_df_path=f'/home/apollo/data/report_results/{phase}_metrics.csv')

    df_by_run = rm.get_results_df()
    df_by_run = df_by_run.sort_values(['dsc'])

    return df_by_run



def compute_metrics_from_saved_model(config, phase='validation', save_by_patient_id=False, **kwargs):

    config['batch_size']=1
    config['valid_batch_size']=1

    plot_title_prefix = kwargs.pop('plot_title_prefix', None)
    filter_by_patients = kwargs.pop('filter_by_patients', [])
    
    network = load_model(config)

    device = config['device']

    if 'include_T1' in config.keys():
        include_T1 = config['include_T1']
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
        rm = ResultsManager(results_df_path=f'/home/apollo/data/report_results/{phase}_metrics.csv')

        clear_results_df = kwargs.pop('clear_results_df', False)
        if clear_results_df:
            rm.clear_results_df()

    is_train=False
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
            rm.add_to_results_df(model_name = f'{config["model_name"]}', 
                                patient_id=patient_id,
                                phase=phase,
                                dsc=dsc,
                                tp=tp,
                                fp=fp,
                                tn=tn,
                                fn=fn)
        
    return dsc, tp_sum, fp_sum, tn_sum, fn_sum  


def map_columns(df):

    #df.reset_index(inplace=True, drop=True)
    df_keys = df.keys()

    # if 'RandomAffine' in df_keys:
    #     df['augmentation'] = df['RandomAffine'].isna()
    #     df['augmentation_network_'] = ( df['network_name'] + df['RandomAffine'].isna().astype('str')).replace(replace_dict)

    if 'augmentation' in df_keys:
        replace_dict = {False: 'No augmentation', True : 'Augmentation'}
        df['augmentation_'] = df['augmentation'].replace(replace_dict)

    replace_dict = {'UNetFalse': 'UNet (no augmentation)', 'UNetTrue' : 'UNet (augmentation)',
                    'UNet_2PlusFalse': 'UNet ++ (no augmentation)', 'UNet_2PlusTrue': 'UNet ++ (augmentation)', 
                    'UNet_3PlusFalse': 'UNet 3+ (no augmentation)', 'UNet_3PlusTrue': 'UNet 3+ (augmentation)', 
                    'smp.UnetFalse' : 'UNet (smp) (no augmentation)', 'smp.UnetTrue' : 'UNet (smp) (augmentation)'
                    }

    inlcude_T1_labels=['T2 only', 'T1 and T2']  
    df['include_T1_'] = df['include_T1'].replace({False: inlcude_T1_labels[0], True : inlcude_T1_labels[1]})
    
    replace_dict = {'TverskyLoss': 'Tversky Loss', 
                    'TverskyFocalLoss': 'Tversky Focal Loss', 
                    'BCELoss': 'BCE Loss', 
                    'FocalLoss': 'Focal Loss', 
                    'WeightedDiceLoss': 'Weighted Dice Loss', 
                    'FBetaLoss' : 'F-Beta Loss', 
                    'BCEDiceLoss' : 'BCE Dice Loss',
                    'DiceLoss' : 'Dice Loss'}
    df['loss_function_'] = df['loss_function'].replace(replace_dict)

    replace_dict = {'TverskyLoss': 'Tversky', 
                    'TverskyFocalLoss': 'Tversky Focal', 
                    'BCELoss': 'BCE', 
                    'FocalLoss': 'Focal', 
                    'WeightedDiceLoss': 'Wgt. Dice', 
                    'FBetaLoss' : 'F-Beta', 
                    'BCEDiceLoss' : 'BCE Dice',
                    'DiceLoss' : 'Dice'}
    df['loss_function_trim'] = df['loss_function'].replace(replace_dict)


    replace_dict = {'UNet_2Plus': 'UNet ++', 'UNet_3Plus': 'UNet 3+', 'smp.Unet' : 'UNet (smp)'}
    df['network_name_'] = df['network_name'].replace(replace_dict)


    return df


def append_mean_median(df, df_box):
    df['mean dsc'] = 0
    df['median dsc'] = 0

    for idx in df.index:
        model_name = df.loc[idx]['model_name']
        model_name = df.loc[idx]['model_name']
        dsc_vals = df_box[df_box['model_name']==model_name]['dsc']
        df.loc[idx, 'mean dsc'] = dsc_vals.mean()
        df.loc[idx, 'median dsc'] = dsc_vals.median()

        df.loc[idx, 'mean tp'] = df_box[df_box['model_name']==model_name]['tp'].mean()
        df.loc[idx, 'mean fp'] = df_box[df_box['model_name']==model_name]['fp'].mean()
        df.loc[idx, 'mean fn'] = df_box[df_box['model_name']==model_name]['fn'].mean()

        df.loc[idx, 'mean ppv'] = df_box[df_box['model_name']==model_name]['ppv'].mean()
        df.loc[idx, 'mean sensitivity'] = df_box[df_box['model_name']==model_name]['sensitivity'].mean()
        df.loc[idx, 'mean dscoeff'] = df_box[df_box['model_name']==model_name]['dscoeff'].mean()

    return df



def plot_results_from_prediction(original_, 
                                prediction_, 
                                label_,
                                ax=None,
                                title_prefix_= '',
                                show_title=True,
                                show_both_masks=False, 
                                transparency=0.5, 
                                zoom_inset=None, 
                                patient_id_=None,
                                model_name_=None,
                                save_fname=None, 
                                tp_label='true positives',
                                fn_label='false negatives',
                                fp_label='false positives',    
                                return_metrics=False,  
                                crop_shape=None,    
                                overwrite_window=None,    
                                hide_legend=False,                       
                                legend_loc=4):

    if ax is None:
        f, ax = plt.subplots(1,1, figsize=(16,10))

    original_ = np.squeeze(original_.cpu().numpy())
    prediction_ = np.squeeze(prediction_.int().cpu().numpy())
    label_ = np.squeeze(label_.int().cpu().numpy())

    if crop_shape is not None:
        original_ = crop_npy_to_shape(original_)
        prediction_ = crop_npy_to_shape(prediction_)
        label_ = crop_npy_to_shape(label_)

        # max_x, max_y = original_.shape
        # crop_x, crop_y = int((max_x - crop_shape[0])/2), int((max_y- crop_shape[1])/2)
        # crop_y_30 = int(crop_y*.4)
        # extra_x = crop_shape[0] - (max_x - crop_x-crop_x)
        # extra_y = crop_shape[1] - (max_y - crop_y+crop_y_30-crop_y-crop_y_30)

        # original_ = original_[crop_x:-crop_x+extra_x, crop_y-crop_y_30:-crop_y-crop_y_30+extra_y]
        # prediction_ = prediction_[crop_x:-crop_x+extra_x, crop_y-crop_y_30:-crop_y-crop_y_30+extra_y]
        # label_ = label_[crop_x:-crop_x+extra_x, crop_y-crop_y_30:-crop_y-crop_y_30+extra_y]

    true_pred, false_pred = label_ == prediction_, label_ != prediction_
    pos_pred, neg_pred = prediction_ == 1, prediction_ == 0

    tp = (true_pred * pos_pred)
    fp = (false_pred * pos_pred)

    tn = (true_pred * neg_pred)
    fn = (false_pred * neg_pred)

    if return_metrics:            

        epsilon=0
        dsc =  2 * (tp.sum() + epsilon) / ( prediction_.sum()+label_.sum()+epsilon )
        
        metrics = f'{title_prefix_}TP:{tp.sum()}, FN:{fn.sum()}, FP:{fp.sum()}, DSC:{dsc:.2f}'
    else:
        metrics = ''

    combined_ = tp * 1 + fn * 2 + fp * 3
    
    colour_list = ['white', 'green', 'orange', 'red']
    cmap = colors.ListedColormap(colour_list)
    bounds=[0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N) 

    alphas = np.ones(np.rot90(combined_).shape)*transparency
    alphas[np.rot90(combined_)==0] = 0

    #bounds=[0., 0.00784314, 0.01176471, 0.01568628]

    if zoom_inset is None:            

        ax.imshow(np.rot90(original_), cmap="gray")
        ax.imshow(np.rot90(combined_), alpha=alphas, cmap=cmap, norm=norm)
        if show_title:
            title = f'{title_prefix_}{metrics}'
            ax.set_title(title, fontsize=30)
        ax.axis('off')

    else:
        ax.imshow(np.rot90(original_), cmap="gray")

        if show_both_masks:
            ax.imshow(np.rot90(combined_), alpha=alphas, cmap=cmap, norm=norm)

        pop_a = mpatches.Patch(color=colour_list[1], label=tp_label)
        pop_b = mpatches.Patch(color=colour_list[2], label=fn_label)
        pop_c = mpatches.Patch(color=colour_list[3], label=fp_label)

        # pop_a = mpatches.Patch(color=colour_list[1], label='true positives')
        # pop_b = mpatches.Patch(color=colour_list[2], label='false negatives')
        # pop_c = mpatches.Patch(color=colour_list[3], label='false positives')

        if legend_loc is None:
            pass
        elif isinstance(legend_loc, int):
            plt.legend(handles=[pop_a,pop_b,pop_c], loc=legend_loc, fontsize=20)
        else:
            plt.legend(handles=[pop_a,pop_b,pop_c], bbox_to_anchor=(1, 1), loc='upper left', fontsize=20)

        if overwrite_window is None:
            x1, x2, y1, y2 = get_cropped_window_dims(np.rot90(combined_))
        else:    
            x1, x2, y1, y2 = overwrite_window

        axins = zoomed_inset_axes(ax, zoom_inset, loc=9) # zoom-factor: 2.5, location: upper-left
        ax.imshow(np.rot90(original_), cmap="gray")

        if show_both_masks:            
            ax.imshow(np.rot90(combined_), alpha=alphas, cmap=cmap, norm=norm)

        axins.imshow(np.rot90(original_), cmap="gray")
        axins.imshow(np.rot90(combined_), alpha=alphas, cmap=cmap, norm=norm)
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y2, y1) # apply the y-limits
        plt.yticks(visible=False)
        plt.xticks(visible=False)

        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="1.0")

        if show_title:
            ax.set_title(title, fontsize=26)
        ax.axis('off')

        if hide_legend:
            ax.get_legend().remove()                

    if save_fname is not None:
        f.savefig(save_fname)

    if return_metrics:
        return metrics
        


def get_cropped_window_dims(mask, pad=4):
    df = pd.DataFrame(mask)
    df = df.loc[(df!=0).any(axis=1)]
    df = df.T
    df = df.loc[(df!=0).any(axis=1)]
    df = df.T
    
    return df.columns[0]-pad, df.columns[-1]+pad, df.index[0]-pad, df.index[-1]+pad
        


def crop_npy_to_shape(input_, crop_shape=(160,160)):
        max_x, max_y = input_.shape
        crop_x, crop_y = int((max_x - crop_shape[0])/2), int((max_y- crop_shape[1])/2)
        crop_y_30 = -int(crop_y*.8)
        extra_x = crop_shape[0] - (max_x - crop_x-crop_x)
        extra_y = crop_shape[1] - (max_y - crop_y+crop_y_30-crop_y-crop_y_30)

        #input_ = input_[crop_x:-crop_x+extra_x, crop_y-crop_y_30:-crop_y-crop_y_30+extra_y]
        input_ = input_[crop_x:crop_x+crop_shape[0], crop_y-crop_y_30:crop_y-crop_y_30+crop_shape[1]]

        return input_

        
# target_shape = {'target_shape': [160, 160, 1]}
# update_target_shape = False

# def _remove_wandb_run_folders(remove_folders):

#     wandb_runs_folder = Path('/home/apollo/code/mproj7205/wandb')

#     for remove in remove_folders:
#         remove_path = wandb_runs_folder / remove
#         if os.path.isdir(remove_path): 
#             #print(remove)
#             shutil.rmtree(remove_path)

# def keep_by_project(project_names):

#     wandb_runs_folder = Path('/home/apollo/code/mproj7205/wandb')
    
#     keep_folders = [] 
#     for project_name in project_names:
#         wa = WandbAnalysis(project_name=f'apollo/{project_name}')
#         df = wa.all_df

#         # Move all files from this run to a temp directory for deletion
#         move_from_dir = os.listdir(wandb_runs_folder)
#         for idx in wa.all_df.index:
#             dirnames=[x for x in move_from_dir if wa.all_df.loc[idx,'wandb_id'] in x]
#             #print(dirnames)
#             if dirnames:
#                 dirname = dirnames[0]
#                 #print(f'Removing run folder {dirname}')  
#                 keep_folders.append(dirname)
                
#     remove_folders = [x for x in move_from_dir if x not in keep_folders]
#     remove_folders = [x for x in remove_folders if x.startswith('run')]
    
#     _remove_wandb_run_folders(remove_folders)

# def keep_by_metric(project_names, min_dice_score=0.67, min_acc=0.6):

#     wandb_runs_folder = Path('/home/apollo/code/mproj7205/wandb')
#     wandb_ids = []
#     for project_name in project_names:
#         wa = WandbAnalysis(project_name=f'apollo/{project_name}')
#         df = wa.all_df
        
#         if 'dice score (val)' in df.columns:
#             df = df[df['dice score (val)'] > min_dice_score]
#             wandb_id = df['wandb_id']
#             wandb_ids += list(wandb_id)
#         elif 'acc (val)' in df.columns:
#             df = df[df['acc (val)'] < min_acc]
#             wandb_id = df['wandb_id']
#             wandb_ids += list(wandb_id)

#     wandb_runs_folder = os.listdir(wandb_runs_folder)

#     remove_folders = []
#     for wandb_id in wandb_ids:
#         remove_folders += [x for x in wandb_runs_folder if x.endswith(wandb_id)]
   
#     _remove_wandb_run_folders(remove_folders)
#     _delete_poor_runs(project_names, min_dice_score=min_dice_score, min_acc=min_acc)

# def _delete_poor_runs(project_names, min_dice_score=0.67, min_acc=0.5):

#     wandb_ids = []
#     for project_name in project_names:
#         wa = WandbAnalysis(project_name=f'apollo/{project_name}')
#         df = wa.all_df
        
#         if 'dice score (val)' in df.columns:
#             df = df[df['dice score (val)'] < min_dice_score]
#             wandb_id = df['wandb_id']
#             wandb_ids += list(wandb_id)
#         elif 'acc (val)' in df.columns:
#             df = df[df['acc (val)'] > min_acc]
#             wandb_id = df['wandb_id']
#             wandb_ids += list(wandb_id)
            
#     # for run_name in df['name']:
#     #     try:
#     #         wa.runs[wa.get_run_idx(run_name)].delete()
#     #     except:
#     #         pass

# def save_models_by_metric(project_names, min_dice_score=0.67, min_acc=0.6, delete_original=False):

#     wandb_runs_folder = Path('/home/apollo/code/mproj7205/wandb')
#     new_models_path = Path('/home/apollo/data/new_models')
    
#     models_to_save = []
#     for project_name in project_names:
#         wa = WandbAnalysis(project_name=f'apollo/{project_name}')
#         df = wa.all_df
        
#         if 'dice score (val)' in df.columns:
#             df = df[df['dice score (val)'] > min_dice_score]
#             model_names = df['model_name']
#             model_dirs = df['model_dir']
#             models_paths = [os.path.join(x,y) for x,y in zip(list(model_dirs), list(model_names))]
#             models_to_save +=models_paths
#         elif 'acc (val)' in df.columns:
#             df = df[df['acc (val)'] > min_acc]
#             model_names = df['model_name']
#             model_dirs = df['model_dir']
#             models_paths = [os.path.join(x,y) for x,y in zip(list(model_dirs), list(model_names))]
#             models_to_save +=models_paths
            
#         for from_image in models_to_save:

#             from_image = Path(from_image)  
#             if from_image.is_file():
#                 shutil.copy(from_image, new_models_path / from_image.name)   
#                 if delete_original:
#                     from_image.unlink()
#                     print(f'saving model from {from_image} to {new_models_path / from_image.name}')
    



# def save_models_to_dir(df, old_models_path, new_models_path):

#     old_models_path = Path(old_models_path)
#     new_models_path = Path(new_models_path)
#     new_models_path.mkdir(parents=True, exist_ok=True)

#     for idx in df.index:
#         model_name = df.loc[idx]['model_name']
#         shutil.copy(old_models_path / model_name, new_models_path / model_name)  


# def keep_state_dict_by_min_score(project_name, models_path, min_dice_score=0.67, min_acc=0.5):

#     _delete_poor_runs(project_names=[project_name], min_dice_score=min_dice_score, min_acc=min_acc)
#     models_path = Path(models_path)

#     wa = WandbAnalysis(project_name=f'apollo/{project_name}')
#     df = wa.all_df

#     all_models_ = os.listdir(models_path)
#     keep_models_ = list(df['model_name'])

#     delete_models_ = [x for x in all_models_ if x not in keep_models_]
#     for model_name in delete_models_:
#         model_path = models_path / model_name
#         if model_path.is_file():
#             model_path.unlink()


# def get_wandb_config_from_json(config_fname):
#     wandb_config = read_json(config_fname)
#     new_config = {}
#     for k1, k2 in wandb_config['parameters'].items():
#         if isinstance(k2,dict):
#             if 'value' in k2.keys():
#                 #print(k1,k2['value'])
#                 new_config.update({k1:k2['value']})
#             if 'values' in k2.keys():
#                 #print(k1,k2['value'])
#                 new_config.update({k1:k2['values'][0]})
#     return new_config


# def update_config_from_df(config, df_series):

#     config['model_name'] = df_series['model_name']
#     config['network_name'] = df_series['network_name']
#     config['include_T1'] = df_series['include_T1']
#     config['model_dir'] = df_series['model_dir']
#     config['encoder_name'] = df_series['encoder_name']
#     config['encoder_weights'] = df_series['encoder_weights']
#     if update_target_shape: config['CropOrPad'] = target_shape

#     return config

# def drop_unnecessary_columns(df):
    
#     dont_display_cols = ['device', 'verbose', 'workers','epochs', 'epoch',\
#                          'batch_size','valid_batch_size','callback_log_model_freq','early_stopping', 'log_dce_loss',\
#                         'callback_log_images_freq','_step','_runtime','_timestamp', 'fn (train)', 'tn (val)', \
#                         'fn (val)', 'fp (val)', 'tn (train)', 'tp (val)', 'tp (train)', 'fp (train)', 'wandb_id']
    
#     return df.drop(dont_display_cols,axis=1)

# def remove_rows_without_saved_model(df, models_path, verbose=False):
        
#     for sweep in df['name']:
#         model_pth = models_path / f'{sweep}.pt'
#         if model_pth.is_file(): 
#             pass
#         else:
#             df.drop(df[df['name'] ==sweep].index, inplace=True)        
#             if verbose: print(f'model {sweep} not found')
            
#     return df

# def get_df_for_analysis(project_name, models_path=None, min_dice_score=0.75, verbose=False):

#     wa = WandbAnalysis(project_name=project_name)
    
#     df = wa.all_df
#     df = drop_unnecessary_columns(df)
#     df = remove_rows_without_saved_model(df, models_path, verbose=verbose)
    
#     df = df[df['dice score (val)'] > min_dice_score]

#     df = df[df['random_split'] == False]

#     df['project_name'] = project_name
#     df['beta'] = df['loss_criterion'].apply(lambda x: x['beta'] if 'beta' in x.keys() else '')
#     df['loss_function'] = df['loss_criterion'].apply(lambda x: x['loss_function'])
#     df.drop('loss_criterion', axis=1, inplace=True)

#     front_try = ['name', 'network_name', 'include_T1', 'loss_function', 'encoder_name', 'encoder_weights'\
#                  'dice score (val)', 'dice score (train)', 'loss (val)', 'loss (train)', 'project_name']

#     old_order = list(df.columns)

#     front = [x for x in front_try if x in old_order]

#     back = [x for x in old_order if x not in front_try]

#     new_order = front + back
#     df = df[new_order]

#     df['model_dir'] = str(models_path)

#     return df

# def get_multiple_projects_df_for_analysis(project_names, models_path=None, min_dice_score=0.6, verbose=False):

#     project_name = project_names[0]
#     df = get_df_for_analysis(project_name, models_path=models_path, min_dice_score=min_dice_score, verbose=verbose)

#     for project_name in project_names[1:]:
#         df1 = get_df_for_analysis(project_name, models_path=models_path, min_dice_score=min_dice_score, verbose=verbose)
#         df = pd.concat([df, df1], axis=0, ignore_index=True)

#     return df
        

# def get_multiple_projects_df_for_analysis_from_multiple_model_paths_v2(project_names, models_paths=None, min_dice_score=0.6, verbose=False):

#     first=True
#     for models_path in models_paths:
#         for project_name in project_names:
#             if first:
#                 df = get_df_for_analysis(project_name, models_path=models_path, min_dice_score=min_dice_score, verbose=verbose)
#                 first = False
#             else:
#                 df1 = get_df_for_analysis(project_name, models_path=models_path, min_dice_score=min_dice_score, verbose=verbose)
#                 df = pd.concat([df, df1], axis=0, ignore_index=True)

#     return df

# def add_final_metrics_to_saved_model_v2(df1, cwd_path=None):

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')

#     df2 = pd.DataFrame(columns=['dsc', 'tp_sum', 'fp_sum', 'tn_sum', 'fn_sum' ])
#     for idx in df1.index:
#         project_name = df1.loc[idx]['project_name']        

#         config_fname = cwd_path / f"configs/{project_name}.json"
#         config = get_wandb_config_from_json(config_fname)
#         config = update_config_from_df(config, df_series = df1.loc[idx])

#         wa = WandbAnalysis(project_name=project_name)
#         try:
#             config['name'] = df1.loc[idx]['name']
#             config['model_name'] = df1.loc[idx]['model_name']
#             df2.loc[idx] = wa.compute_metrics_from_saved_model(config, phase='validation', save_by_patient_id=False)
#             #print(df2.loc[idx])
#             run_name = df1.loc[idx]['name']
#             print(f'Found valid model for {run_name} ')

#         except:
#             #run_name = df1.loc[idx]['name']
#             #print(f'skipping and deleting {run_name} ')
#             pass

#             df2.loc[idx] = -1.0, -1, -1, -1, -1

#     df = pd.concat([df1, df2], axis=1)

#     df = df[df['dsc'] > 0]
#     old_order = list(df1.columns)
#     front = old_order[:5]
#     back =  old_order[5:]
#     middle = list(df2.columns)
    
#     new_order = front + middle + back
#     df = df[new_order]

#     df = df.sort_values(['dsc'], ascending=False)

#     return df


# def add_final_metrics_to_saved_model_wandb_v2(df1, cwd_path=None):

#     df1 = df1.copy()

#     # Onle consider models created using wandb
#     df1 = df1[df1['dice score (val)'] != 0.666]

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')

#     df2 = pd.DataFrame(columns=['dsc', 'tp_sum', 'fp_sum', 'tn_sum', 'fn_sum' ])
#     for idx in df1.index:
#         project_name = df1.loc[idx]['project_name']        

#         config_fname = cwd_path / f"configs/{project_name}.json"
#         config = get_wandb_config_from_json(config_fname)
#         config = update_config_from_df(config, df_series = df1.loc[idx])

#         wa = WandbAnalysis(project_name=project_name)

#         try:
#             config['name'] = df1.loc[idx]['name']
#             config['model_name'] = df1.loc[idx]['model_name']
#             df2.loc[idx] = wa.compute_metrics_from_saved_model(config, phase='validation', save_by_patient_id=False)
#             #print(df2.loc[idx])
#             run_name = df1.loc[idx]['name']
#             print(f'Found valid model for {run_name} ')

#         except:
#             #run_name = df1.loc[idx]['name']
#             #print(f'skipping and deleting {run_name} ')
#             pass

#             df2.loc[idx] = -1.0, -1, -1, -1, -1

#     df = pd.concat([df1, df2], axis=1)

#     df = df[df['dsc'] > 0]
#     old_order = list(df1.columns)
#     front = old_order[:5]
#     back =  old_order[5:]
#     middle = list(df2.columns)
    
#     new_order = front + middle + back
#     df = df[new_order]

#     df = df.sort_values(['dsc'], ascending=False)

#     return df

# def add_final_metrics_to_saved_model_nowandb_v2(df1, cwd_path=None, models_path_=None, configs_path=None):

#     df1 = df1.copy()

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')

#     df2 = pd.DataFrame(columns=['dsc', 'tp_sum', 'fp_sum', 'tn_sum', 'fn_sum', 'model_dir'])
#     for idx in df1.index:
#         if configs_path is None:
#             project_name = df1.loc[idx]['project_name']
#             configs_path_ =  Path(f'/home/apollo/code/mproj7205/configs/{project_name}')
#         else:
#             configs_path_ = configs_path

#         model_name = df1.loc[idx]['model_name']

#         config_path = configs_path_ / f"{model_name.replace('.pt', '.json')}"

#         if models_path_ is None:
#             models_path = Path(df1.loc[idx]['model_dir'])
#         else:
#             # Overwrite with the models_path provided
#             models_path = models_path_
            
#         model_path = models_path / model_name

#         if config_path.is_file() and model_path.is_file():
#             config = read_json(config_path)
#             config['model_dir'] = str(models_path)
#             if update_target_shape: config['CropOrPad'] = target_shape


#         model_path = Path(os.path.join(config['model_dir'], config['model_name']))

#         try:
#             metrics = compute_metrics_from_saved_model(config, phase='validation', save_by_patient_id=False)
#             df2.loc[idx] = list(metrics) + [str(models_path)]
#         except:
#             #print(f'skipping {model_path} not found')
#             df2.loc[idx] = -1.0, -1, -1, -1, -1, [""]


#     df1.drop(['model_dir'], axis=1, inplace=True)
#     df = pd.concat([df1, df2], axis=1)

#     df = df[df['dsc'] > 0]
#     old_order = list(df1.columns)
#     front = old_order[:5]
#     back =  old_order[5:]
#     middle = list(df2.columns)
    
#     new_order = front + middle + back
#     df = df[new_order]

#     df = df.sort_values(['dsc'], ascending=False)

#     return df

# def compute_metrics_by_run(df, idx=None, sweep_name=None, phase='validation', cwd_path=None):

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')

#     if sweep_name:
#         idx = df[df['name'] == sweep_name].index.values.astype(int)[0] 
        
#     if idx is None:
#         print('Must provide valid idx or sweep_name')
#         return
    
#     project_name = df.loc[idx]['project_name']
#     config_fname = cwd_path / f"configs/{project_name}.json"
#     config = get_wandb_config_from_json(config_fname)
#     config = update_config_from_df(config, df_series = df.loc[idx])

#     wa = WandbAnalysis(project_name=project_name)
#     config['name'] = df.loc[idx]['name']
#     config['model_name'] = df.loc[idx]['model_name']
#     _ = wa.compute_metrics_from_saved_model(config, phase=phase, save_by_patient_id=True, clear_results_df=True)

#     rm = ResultsManager(results_df_path=f'/home/apollo/data/report_results/{phase}_metrics.csv')

#     df_by_run = rm.get_results_df()
#     df_by_run = df_by_run.sort_values(['dsc'])

#     return df_by_run


# def run_all_predictions_by_patient_id(df, patient_id, plot_T2_only=True, verbose=False, cwd_path=None):

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')

#     filter_by_patients = [patient_id]
    
#     for idx in df.index:
#         project_name = df.loc[idx]['project_name']
#         config_fname = cwd_path / f"configs/{project_name}.json"
#         config = get_wandb_config_from_json(config_fname)
#         config = update_config_from_df(config, df_series = df.loc[idx])  

#         wa = WandbAnalysis(project_name=project_name)  
#         config['name'] = df.loc[idx]['name']
#         config['model_name'] = df.loc[idx]['model_name']
#         wa.show_networks_predictions(config, plot_title_prefix='', filter_by_patients=filter_by_patients, plot_T2_only=plot_T2_only, verbose=verbose)    


# def run_all_predictions_by_patient_id_by_project_name(project_name, patient_id, models_path, plot_T2_only=True, verbose=False, cwd_path=None):

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')

#     filter_by_patients = [patient_id]
#     df = get_df_for_analysis(project_name, models_path=models_path, min_dice_score=0.67, verbose=verbose)

#     for idx in df.index:
#         config_fname = cwd_path / f"configs/{project_name}.json"
#         config = get_wandb_config_from_json(config_fname)
#         config = update_config_from_df(config, df_series = df.loc[idx])  

#         wa = WandbAnalysis(project_name=project_name)  
#         config['name'] = df.loc[idx]['name']
#         config['model_name'] = df.loc[idx]['model_name']
#         wa.show_networks_predictions(config, plot_title_prefix='', filter_by_patients=filter_by_patients, plot_T2_only=plot_T2_only, verbose=verbose)    



# def plot_history_from_multiple_df(df, dataset='train', save_fname=None, \
#                             annotate_xy=None, xlim=None, fontsize=12, **kwargs):

#     f, ax = plt.subplots(1, 1, **kwargs)

#     y_col = f'dice score ({dataset})'
        
#     labels = []

#     for idx in df.index:
#         project_name = df.loc[idx]['project_name']
#         run_name = df.loc[idx]['name']

#         wa = WandbAnalysis(project_name=project_name) 
        
#         try:
#             run_idx = wa.get_run_idx(run_name)
#             h = wa.get_history(run_idx)
#             wa.plot_history(h, ax, y_col=y_col, fontsize=fontsize, xlim=xlim)
#             labels.append(f'{run_name}')
#         except:
#             pass
    
#     ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize-4)
#     if annotate_xy is not None:
#         ax.annotate('selected baseline experiment', size=15, color='red', xy=(70, 0.22), xytext=(160, 0.6), \
#                     arrowprops=dict(facecolor='red', shrink=0.05),)
#         #ax.annotate('Text 1', xy=annotate_xy, xytext=(500,0.8), arrowprops=dict(arrowstyle='->'))
#     #ax.legend(labels, fontsize=fontsize-2)
#     f.tight_layout()
#     f.show()
    
#     if save_fname is not None:
#         f.savefig(save_fname, bbox_inches='tight')
        

# def compare_labellers(patent_ids=[], multi_SN_RN_path=None, show_patient_id=True, figs_save_path=None, verbose=False):
#     """
#     patient_ids: list of patient ids. If not set, all examples with two labels will be printed.

#     """
#     if figs_save_path is not None:
#         figs_save_path = Path(figs_save_path)
#         figs_save_path.mkdir(parents=True, exist_ok=True)

#     if multi_SN_RN_path is None:
#         multi_SN_RN_path = Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/')

#     npy_path_0 = multi_SN_RN_path / "0/angus/"
#     npy_path_1 = multi_SN_RN_path / "1/ben/"

#     npy_1 = os.listdir(npy_path_0 / 'labels')
#     npy_2 = os.listdir(npy_path_1 / 'labels')

#     both = [x for x in npy_1 if x in npy_2]

#     # Remove 3607 from this list
#     both = [x for x in both if '3828' not in str(x)]
#     both = [x for x in both if '3607' not in str(x)]

#     pid2label_dict = {}
#     for i, k in enumerate(both):
#         regex = re.search(r'label_(.*)_(.*)', k)
#         if regex:
#             patient_id = int(regex.group(1))
#             pid2label_dict[patient_id] = i

#     wa = WandbAnalysis() 

#     if patent_ids:
#         # Only get the patient ids that exist in both
#         patent_ids = [x for x in patent_ids if x in pid2label_dict.keys()]

#         # Get the index from patient ids
#         index_list = [pid2label_dict[i] for i in patent_ids]
#     else:
#         index_list = list(pid2label_dict.values())
#         if verbose: print(f'total number of images with multiple labellers: {len(both)}')


#     for idx in index_list:

#         label_ = npy_path_0 / 'labels' / both[idx]
#         prediction_ = npy_path_1 / 'labels' /  both[idx]
#         original_ = npy_path_0 / 'images' /  both[idx].replace('label', 'image')

#         title_prefix=''
#         model_name=''

#         regex = re.search(r'label_(.*)_(.*)', both[idx])
#         if regex:
#             patient_id = int(regex.group(1))
#         else:
#             patient_id = ''

#         if show_patient_id:
#             title_prefix+=f'{patient_id} '

#         if label_.is_file() and prediction_.is_file() and original_.is_file():
            
#             label_ = torch.from_numpy(np.load(label_))
#             prediction_ = torch.from_numpy(np.load(prediction_))
#             original_ = torch.from_numpy(np.load(original_))
            
#             f, ax = plt.subplots(1,1, figsize=(16,10))
#             wa.plot_results_from_prediction(original_=original_, 
#                                             label_=label_, 
#                                             prediction_=prediction_,
#                                             ax=ax,
#                                             title_prefix_=title_prefix,
#                                             model_name_ = model_name,
#                                             patient_id_ = patient_id,
#                                             zoom_inset=2.5,
#                                             tp_label='agreement',
#                                             fn_label='labeller 1 only',
#                                             fp_label='labeller 2 only',   
#                                             legend_loc=4)

#             if figs_save_path is not None:
#                 save_fname = figs_save_path / f'compare_labellers_{patient_id}.png'
#                 f.savefig(save_fname, bbox_inches='tight')


# def compare_labellers_create_subfig_v3(patent_ids=[], 
#                                       multi_SN_RN_path=None, 
#                                       show_patient_id=False,                                   
#                                       fname_stem='compare_labellers', 
#                                       tex_base_path='./Figures/label_uncertainty', 
#                                       main_caption='', 
#                                       verbose=False, 
#                                       crop_shape=None):
#     """
#     patient_ids: list of patient ids. If not set, all examples with two labels will be printed.

#     """

#     results_path = Path('/home/apollo/data/report_results/')
#     figs_save_path = results_path / tex_base_path /'figs'
#     figs_save_path.mkdir(parents=True, exist_ok=True)

#     tex_save_path = results_path / tex_base_path /'tex'
#     tex_save_path.mkdir(parents=True, exist_ok=True)

#     if multi_SN_RN_path is None:
#         multi_SN_RN_path = Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/')

#     npy_path_0 = multi_SN_RN_path / "0/angus/"
#     npy_path_1 = multi_SN_RN_path / "1/ben/"

#     npy_1 = os.listdir(npy_path_0 / 'labels')
#     npy_2 = os.listdir(npy_path_1 / 'labels')

#     both = [x for x in npy_1 if x in npy_2]

#     # Remove 3607 from this list
#     both = [x for x in both if '3828' not in str(x)]
#     both = [x for x in both if '3607' not in str(x)]

#     pid2label_dict = {}
#     for i, k in enumerate(both):
#         regex = re.search(r'label_(.*)_(.*)', k)
#         if regex:
#             patient_id = int(regex.group(1))
#             pid2label_dict[patient_id] = i

#     wa = WandbAnalysis() 

#     if patent_ids:
#         # Only get the patient ids that exist in both
#         patent_ids = [x for x in patent_ids if x in pid2label_dict.keys()]

#         # Get the index from patient ids
#         index_list = [pid2label_dict[i] for i in patent_ids]
#     else:
#         index_list = list(pid2label_dict.values())
#         if verbose: print(f'total number of images with multiple labellers: {len(both)}')

#     subfig_fnames = []
#     subfig_captions = []
#     subfig_labels = []

#     for idx in index_list:

#         label_ = npy_path_0 / 'labels' / both[idx]
#         prediction_ = npy_path_1 / 'labels' /  both[idx]
#         original_ = npy_path_0 / 'images' /  both[idx].replace('label', 'image')

#         title_prefix=''
#         model_name=''

#         regex = re.search(r'label_(.*)_(.*)', both[idx])
#         if regex:
#             patient_id = int(regex.group(1))
#         else:
#             patient_id = ''

#         if show_patient_id:
#             title_prefix+=f'{patient_id} '

#         if label_.is_file() and prediction_.is_file() and original_.is_file():
                        
#             label_ = torch.from_numpy(np.load(label_))
#             prediction_ = torch.from_numpy(np.load(prediction_))
#             original_ = torch.from_numpy(np.load(original_))

#             f, ax = plt.subplots(1,1, figsize=(16,10))
#             wa.plot_results_from_prediction(original_=original_, 
#                                             label_=label_, 
#                                             prediction_=prediction_,
#                                             ax=ax,
#                                             show_title=False,
#                                             title_prefix_=title_prefix,
#                                             model_name_ = model_name,
#                                             patient_id_ = patient_id,
#                                             zoom_inset=2.5,
#                                             tp_label='agreement',
#                                             fn_label='labeller 1 only',
#                                             fp_label='labeller 2 only',   
#                                             crop_shape=crop_shape,    
#                                             legend_loc=4)

#             # f.set_size_inches(8, 6)
#             # f.savefig(save_fname, bbox_inches='tight', dpi=300)

#             # NOW ALSO CREATE THE TEX FILE THAT GOES WITH THESE FIGURES

#             original_ = np.squeeze(original_.cpu().numpy())
#             prediction_ = np.squeeze(prediction_.int().cpu().numpy())
#             label_ = np.squeeze(label_.int().cpu().numpy())
                
#             true_pred, false_pred = label_ == prediction_, label_ != prediction_
#             pos_pred, neg_pred = prediction_ == 1, prediction_ == 0

#             tp = (true_pred * pos_pred)
#             fp = (false_pred * pos_pred)

#             tn = (true_pred * neg_pred)
#             fn = (false_pred * neg_pred)
            
#             epsilon=0
#             dsc =  2 * (tp.sum() + epsilon) / ( prediction_.sum()+label_.sum()+epsilon )
#             iou =  (tp.sum() + epsilon) / ( tp.sum()+fn.sum()+fn.sum()+epsilon )

#             fname = f'{fname_stem}_{patient_id}.png'
#             subfig_fname = f'{tex_base_path}/figs/{fname}'
#             f.savefig(results_path / subfig_fname, bbox_inches='tight')

#             subfig_fnames.append(subfig_fname)
#             #caption = f'TP:{tp.sum()}, FN:{fn.sum()}, FP:{fp.sum()}, DSC:{dsc:.2f}'
#             caption = f' DSC={dsc:.2f} ({tp.sum()}/{fn.sum()}/{fp.sum()})'
#             subfig_captions.append(caption)
#             subfig_labels.append(f'fig:{fname_stem}_{patient_id}')
                
#     create_latex_subfigure(fname_stem, tex_save_path, subfig_fnames, subfig_captions, subfig_labels, main_caption)




# def save_experiment_figures_v3(df, crop_shape=None,
#                                 figs_save_path=None, 
#                                 results_path=None,
#                                 cwd_path=None, 
#                                 tex_base_path='./Figures/experiments', 
#                                 sort_column = 'loss_function',
#                                 fname_stem='compare_',
#                                 main_caption='', 
#                                 verbose=False):
#     if figs_save_path is not None:
#         figs_save_path = Path(figs_save_path)
#         figs_save_path.mkdir(parents=True, exist_ok=True)

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')

#     if results_path is None:
#         results_path = Path('/home/apollo/data/report_results/')

#     validation_patient_ids = get_validation_patient_ids()

#     captions_dict={}
#     stats_by_patient_id={}

#     for idx in df.index:
#         project_name = df.loc[idx]['project_name']
#         model_name = df.loc[idx]['model_name'].lower()
#         network_name = df.loc[idx]['network_name']

#         experiment_name = model_name.replace('.pt', '').replace('-', '_')
#         captions_dict[experiment_name] = {}

#         dsc_values = []
#         subfig_fnames_inner = []

#         caption_column_value = df.loc[idx][sort_column]

#         config_fname = cwd_path / f"configs/{project_name}.json"
#         config = get_wandb_config_from_json(config_fname)
#         config = update_config_from_df(config, df_series = df.loc[idx])  
#         config['batch_size']=1
#         config['valid_batch_size']=1
#         config = Bunch(config)

#         wa = WandbAnalysis(project_name=project_name) 

#         network = load_model(config)

#         device = config.device

#         if 'include_T1' in config.keys():
#             include_T1 = config.include_T1
#         else:
#             include_T1 = False        
        
#         _, val_loader = build_datasets(config, verbose=verbose)

#         network.float()
#         is_train=False
#         use_sigmoid = get_use_sigmoid(config)
#         use_sigmoid = False

#         for _, data in enumerate(val_loader):

#             patient_id = int(data['patient_id'][0])

#             if patient_id not in validation_patient_ids:
#                 #print(f'skipping {patient_id}')
#                 continue

#             if int(patient_id) == 3607 or int(patient_id) == 3828:
#                 continue

#             inputs = data['image'][tio.DATA]
#             inputs = np.squeeze(inputs, axis=4)

#             if include_T1:
#                 inputs_T1 = data['image_T1'][tio.DATA]
#                 inputs_T1 = np.squeeze(inputs_T1, axis=4)
#                 inputs = torch.cat((inputs, inputs_T1), 1)

#             inputs = inputs.to(device)

#             labels = data['label'][tio.DATA]
#             labels = np.squeeze(labels, axis=4)
#             labels = labels.to(device)

#             # Forward pass
#             with torch.set_grad_enabled(is_train):
#                 if use_sigmoid:
#                     predictions = torch.sigmoid(network(inputs))
#                 else:
#                     predictions = network(inputs)
                    
#                 # model_name_check = model_name
#                 # model_name = config.model_name
#                 # print(model_name_check, model_name)

#                 inputs = inputs[:,0,:,:].unsqueeze(1)

#                 f, ax = plt.subplots(1,1, figsize=(16,10))
#                 metrics = wa.plot_results_from_prediction(original_=inputs, 
#                                                             label_=labels, 
#                                                             prediction_=predictions,
#                                                             ax=ax,
#                                                             show_title=False,
#                                                             title_prefix_='',
#                                                             model_name_ = model_name,
#                                                             patient_id_ = patient_id,
#                                                             return_metrics=True,
#                                                             crop_shape=crop_shape, 
#                                                             zoom_inset=2.5)

#                 caption_ = metrics

#                 if caption_ is None:
#                     plt.close()
#                     continue

#                 split_caption = caption_.replace(' ', '').replace(':',',').split(',')

#                 if float(split_caption[-1]) < 0.4:
#                     plt.close()
#                     continue


#                 fname = f'{fname_stem}_{patient_id}_{experiment_name}_{idx}.png'
#                 base_path = os.path.join(tex_base_path, f'figs/{str(experiment_name)}')
#                 save_path = results_path / base_path
#                 save_path.mkdir(parents=True, exist_ok=True)
#                 f.savefig(save_path / fname, bbox_inches='tight')

#                 result_dict = dict(zip(split_caption[0::2], split_caption[1::2]))
#                 result_dict.update({'df_idx':idx, 'png_path': f'{base_path}/{fname}'})

#                 if patient_id not in captions_dict[experiment_name].keys():
#                     captions_dict[experiment_name][patient_id] = result_dict
#                 else:
#                     print('WARNING multiple entries for this patient and experiment {patient_id} {experiment_name}')

#                 if patient_id in stats_by_patient_id.keys():
#                     stats_by_patient_id[patient_id]['count'] += 1
#                     stats_by_patient_id[patient_id]['dsc'] += float(split_caption[-1])
#                 else:
#                     stats_by_patient_id[patient_id] = {'count' : 1, 'dsc' : float(split_caption[-1]) }


#                 dsc_values.append(split_caption[-1])
#                 subfig_fnames_inner.append(f'{tex_base_path}/{fname}')

#                 plt.close()

#     for k in stats_by_patient_id.keys():
#         stats_by_patient_id[k]['dsc'] = stats_by_patient_id[k]['dsc'] / stats_by_patient_id[k]['count']


#     captions_dict['stats_by_patient_id'] = stats_by_patient_id
#     fname = f'{fname_stem}_captions.json'
#     save_path = results_path / tex_base_path / 'captions'
#     save_path.mkdir(parents=True, exist_ok=True)
#     print(save_path / fname)
#     write_json(captions_dict, save_path / fname)

#             # caption = f'No.: {patient_id}, {caption_column_value}, ({dsc_values[0]}/{dsc_values[1]})'
#             # subfig_captions.append(caption)
#             # subfig_labels.append(f'fig:{fname}')
#             # subfig_fnames1.append(subfig_fnames_inner[0])
#             # subfig_fnames2.append(subfig_fnames_inner[1])
            
#             # if verbose: print(f'{tex_base_path}/{fname}')
#             # if verbose: print(caption)
#             # if verbose: print(f'fig:{fname}')
#             # if verbose: print(f'fig:{save_fname}')

# def create_latex_row_v3(ROWNAME, FIGURE_NAMES, DOUBLEBAR=""):

#     #     ROWNAME="Dice Loss", 
#     #     FIGURE1 = r"""./Figures/experiments/lossv5/compare_networks_by_pid_3603_tverskyfocallosst1andt2_5.png"""
#     #     FIGURE2 = r"""./Figures/experiments/lossv5/compare_networks_by_pid_3603_tverskyfocallosst1andt2_5.png"""
#     #     FIGURE3 = r"""./Figures/experiments/lossv5/compare_networks_by_pid_3603_tverskyfocallosst1andt2_5.png"""

#     FIGURE1, FIGURE2, FIGURE3 = FIGURE_NAMES[0], FIGURE_NAMES[1], FIGURE_NAMES[2]    
    
#     ADJUSTBOX = r"""\adjustbox{valign=m,vspace=1pt}{\includegraphics[width=.29\linewidth]{FIGUREX}}"""
    
    
#     original_row = r"""    \rotatebox[origin=c]{90}{ROWNAME}  & ADJUSTBOX1 & ADJUSTBOX2 & ADJUSTBOX3 DOUBLEBAR"""
    
#     ADJUSTBOX1 = ADJUSTBOX.replace('FIGUREX', FIGURE1)
#     ADJUSTBOX2 = ADJUSTBOX.replace('FIGUREX', FIGURE2)
#     ADJUSTBOX3 = ADJUSTBOX.replace('FIGUREX', FIGURE3)
    
#     new_row = original_row.replace("ROWNAME", ROWNAME) \
#                           .replace("ADJUSTBOX1", ADJUSTBOX1) \
#                           .replace("ADJUSTBOX2", ADJUSTBOX2) \
#                           .replace("ADJUSTBOX3", ADJUSTBOX3) \
#                           .replace("DOUBLEBAR", DOUBLEBAR)
#     return new_row
    

# def create_latex_row_v4(ROWNAME, FIGURE_NAMES, DOUBLEBAR=""):

#     #     ROWNAME="Dice Loss", 
#     #     FIGURE1 = r"""./Figures/experiments/lossv5/compare_networks_by_pid_3603_tverskyfocallosst1andt2_5.png"""
#     #     FIGURE2 = r"""./Figures/experiments/lossv5/compare_networks_by_pid_3603_tverskyfocallosst1andt2_5.png"""
#     #     FIGURE3 = r"""./Figures/experiments/lossv5/compare_networks_by_pid_3603_tverskyfocallosst1andt2_5.png"""

#     FIGURE1, FIGURE2, FIGURE3 = FIGURE_NAMES[0], FIGURE_NAMES[1], FIGURE_NAMES[2]    
    
#     ADJUSTBOX = r"""\adjustbox{valign=m,vspace=1pt}{\includegraphics[width=.29\linewidth]{FIGUREX}}"""
    
    
#     original_row = r"""    \rotatebox[origin=c]{90}{\textbf{ROWNAME}}  & ADJUSTBOX1 & ADJUSTBOX2 & ADJUSTBOX3 \\[-2mm]"""
    
#     if FIGURE1 == '':
#         ADJUSTBOX1 = ''
#     else:
#         ADJUSTBOX1 = ADJUSTBOX.replace('FIGUREX', FIGURE1)

#     ADJUSTBOX2 = ADJUSTBOX.replace('FIGUREX', FIGURE2)
#     ADJUSTBOX3 = ADJUSTBOX.replace('FIGUREX', FIGURE3)
    
#     new_row = original_row.replace("ROWNAME", ROWNAME) \
#                           .replace("ADJUSTBOX1", ADJUSTBOX1) \
#                           .replace("ADJUSTBOX2", ADJUSTBOX2) \
#                           .replace("ADJUSTBOX3", ADJUSTBOX3)

#     return new_row

# def create_latex_3_col_with_baseline_v3(ROWS, CAPTION):
#     latex_str = r"""\begin{figure}[t]
#     \centering
#     \begin{tabular}{cccc}
#         & \textbf{Baseline} & \textbf{T2 Only} & \textbf{T2 and T1} \\[2mm]   
#     ROWS
#     \end{tabular}
#     \caption{CAPTION}
#     \label{LABEL}
# \end{figure}"""
    
#     DOUBLEBAR = r"\\"
#     ROWS = f'\n'.join(ROWS)
    
#     latex_str = latex_str.replace('CAPTION', CAPTION) \
#                          .replace('ROWS', ROWS)
    
#     return latex_str

# # def create_latex_3_col_with_baseline_v3_df_v3(df_ungrouped, \
# #                                               patient_ids, \
# #                                               num_rows = 1, \
# #                                               figure_fname='fig_compare_.tex', \
# #                                               fname_stem='compare_', \
# #                                               results_path=None, \
# #                                               tex_base_path='./Figures/experiments', \
# #                                               sort_column='loss_function_', \
# #                                               groupby_column='include_T1', \
# #                                               group_values=[False, True]):
    
    
# #     """
# #     Function should take a different sort column. But if needed pass patient_ids of the same length in order to 
# #     change patient id with each new row
# #     """

# #     use_blanks=False
# #     if len(patient_ids) == 1:
# #         patient_id_ = patient_ids[0]
# #         patient_ids = [patient_id_ for _ in range(num_rows)]
# #         use_blanks=True
        
# #     assert len(patient_ids) == num_rows
    
# #     if results_path is None:
# #         results_path = Path('/home/apollo/data/report_results/')
    
# #     fname = f'{fname_stem}_captions.json'
# #     save_path = results_path / tex_base_path / 'captions'
    
# #     read_dict = read_json(save_path / fname)
    
# #     ROWS = []
# #     COMMENT=""
# #     df_grouped = df_ungrouped.sort_values(sort_column)
    
    
# #     row_idx = 0   
# #     for group_name, df_group in df_grouped.groupby(sort_column):
# #         pid_count=0

# #         patient_id = patient_ids[row_idx]
# #         if use_blanks and (row_idx > 0):
# #             ref_image = f'./Figures/label_uncertainty/figs/compare_labellers_blank.png'
# #             ref_cap = ''
# #         else:
# #             ref_image = f'./Figures/label_uncertainty/figs/compare_labellers_{patient_id}.png'
# #             ref_cap = get_label_uncertainty_caption(patient_id=patient_id)
        
# #         FIGURE_NAMES = [ref_image]
# #         SUBCAPTIONS = [ref_cap]
# #         for group_column in [False, True]:
# #             df = df_group[df_group[groupby_column]==group_column].copy().reset_index(drop=True)

# #             if df.empty:
# #                 break

# #             idx=0
# #             project_name = df.loc[idx]['project_name']
# #             network_name = df.loc[idx]['network_name']
# #             model_name = df.loc[idx]['model_name'].lower()
# #             experiment_name = model_name.replace('.pt', '').replace('-', '_')
            

# #             if str(patient_id) in read_dict[experiment_name].keys():
# #                 COMMENT += f'{group_name}, {group_column}, {sort_column}, {read_dict[experiment_name][str(patient_id)]}\n'

# #                 #caption = f'DSC={dsc:.2f} ({tp.sum()}/{fn.sum()}/{fp.sum()})'
# #                 SUBCAPTIONS.append('')
# #                 FIGURE_NAMES.append(read_dict[experiment_name][str(patient_id)]['png_path'])
                
        
# #         if len(FIGURE_NAMES) == 3:
# #             ROWS.append(create_latex_row_v3(group_name, FIGURE_NAMES))
# #             row_idx+=1
            
# #         if row_idx == num_rows:
# #             break
            
# #     CAPTION = 'Comparison of segmentation for models on selected subjects for various loss criterions. For each image pair, the image on the left is using only the T2-weighted images, XYZ whereas the image on the right is using both T1 and T2-weighted images. The value in brackets show the dice scores of each (Left/Right).'
# #     figure_str = create_latex_3_col_with_baseline_v3(ROWS=ROWS, CAPTION=CAPTION)
# #     figure_str += add_comments(COMMENT)

# #     save_path = results_path / tex_base_path / 'tex'
# #     save_path.mkdir(parents=True, exist_ok=True)

# #     tex_fname = save_path / figure_fname    

# #     LABEL = f'fig:{tex_fname.stem}'

# #     figure_str = figure_str.replace('LABEL',LABEL)

# #     f = open(tex_fname, "w")
# #     f.write(figure_str)
# #     f.close()   

# def create_caption_row(ROWVALS):

#     ROWVAL1, ROWVAL2, ROWVAL3 = ROWVALS[0], ROWVALS[1], ROWVALS[2]    

#     original_row = r"""    & ROWVAL1 & ROWVAL2 & ROWVAL3 \\[2mm]"""
    
#     new_row = original_row.replace("ROWVAL1", ROWVAL1) \
#                           .replace("ROWVAL2", ROWVAL2) \
#                           .replace("ROWVAL3", ROWVAL3) 

#     return new_row


# def create_latex_3_col_with_baseline_df_v4(df_ungrouped, \
#                                               patient_ids, \
#                                               num_rows = 1, \
#                                               figure_fname='fig_compare_.tex', \
#                                               fname_stem='compare_', \
#                                               results_path=None, \
#                                               tex_base_path='./Figures/experiments', \
#                                               sort_column='loss_function_', \
#                                               groupby_column='include_T1', \
#                                               caption='',\
#                                               group_values=[False, True]):
    
    
#     """
#     Function should take a different sort column. But if needed pass patient_ids of the same length in order to 
#     change patient id with each new row
#     """

#     use_blanks=False
#     if len(patient_ids) == 1:
#         patient_id_ = patient_ids[0]
#         patient_ids = [patient_id_ for _ in range(num_rows)]
#         use_blanks=True
        
#     assert len(patient_ids) == num_rows
    
#     if results_path is None:
#         results_path = Path('/home/apollo/data/report_results/')
    
#     fname = f'{fname_stem}_captions.json'
#     save_path = results_path / tex_base_path / 'captions'
    
#     read_dict = read_json(save_path / fname)
    
#     ROWS = []
#     COMMENT=""
#     df_grouped = df_ungrouped.sort_values(sort_column)
    
    
#     row_idx = 0   
#     for group_name, df_group in df_grouped.groupby(sort_column):
#         pid_count=0

#         patient_id = patient_ids[row_idx]
#         if use_blanks and (row_idx > 0):
#             #ref_image = f'./Figures/label_uncertainty/figs/compare_labellers_blank.png'
#             ref_image = ''
#             ref_cap = ''
#         else:
#             ref_image = f'./Figures/label_uncertainty/figs/compare_labellers_{patient_id}.png'
#             ref_cap = get_label_uncertainty_caption(patient_id)
        
#         FIGURE_NAMES = [ref_image]
#         SUBCAPTIONS = [ref_cap]
#         for group_column in [False, True]:
#             df = df_group[df_group[groupby_column]==group_column].copy().reset_index(drop=True)

#             if df.empty:
#                 break

#             idx=0
#             project_name = df.loc[idx]['project_name']
#             network_name = df.loc[idx]['network_name']
#             model_name = df.loc[idx]['model_name'].lower()
#             experiment_name = model_name.replace('.pt', '').replace('-', '_')
            

#             if str(patient_id) in read_dict[experiment_name].keys():
#                 COMMENT += f'{group_name}, {group_column}, {sort_column}, {read_dict[experiment_name][str(patient_id)]}\n'

#                 metrics = read_dict[experiment_name][str(patient_id)]
                
#                 subcaption = f"DSC={metrics['DSC']} ({metrics['TP']}/{metrics['FN']}/{metrics['FP']})"
#                 SUBCAPTIONS.append(subcaption)
#                 FIGURE_NAMES.append(read_dict[experiment_name][str(patient_id)]['png_path'])

#         print(len(FIGURE_NAMES))        
#         if len(FIGURE_NAMES) == 3:
#             ROWS.append(create_latex_row_v4(group_name, FIGURE_NAMES))
#             ROWS.append(create_caption_row(SUBCAPTIONS))
#             row_idx+=1
            
#         if row_idx == num_rows:
#             break
            
#     if row_idx != num_rows:
#         return

#     CAPTION = caption.replace('PATIENTID', str(patient_id))
#     figure_str = create_latex_3_col_with_baseline_v3(ROWS=ROWS, CAPTION=CAPTION)
#     figure_str += add_comments(COMMENT)

#     save_path = results_path / tex_base_path / 'tex'
#     save_path.mkdir(parents=True, exist_ok=True)

#     tex_fname = save_path / figure_fname    

#     LABEL = f'fig:{tex_fname.stem}'

#     figure_str = figure_str.replace('LABEL',LABEL)

#     f = open(tex_fname, "w")
#     f.write(figure_str)
#     f.close()   

#     write_to_input_commands_tex(results_path, tex_fname)
    
# def add_comments(COMMENT):

#     comment_str = r"""

# \begin{comment}
#     COMMENT
# \end{comment}
# """
#     return comment_str.replace("COMMENT", COMMENT)

# def write_to_input_commands_tex(results_path, tex_fname):

#     FILELOCATION = str(tex_fname).replace(str(results_path), '.')
#     input_str = r"""\input{FILELOCATION}""" +"\n"

#     input_str = input_str.replace('FILELOCATION', FILELOCATION)

#     input_commands = tex_fname.parent / 'input_commands.tex'
#     print(input_str)

#     f = open(input_commands, "a")
#     f.write(input_str)
#     f.close()   

#     return 

# def create_latex_subfigure(fname_stem, tex_save_path, subfig_fnames, subfig_captions, subfig_labels, main_caption=''):

#     if tex_save_path is not None:
#         tex_save_path = Path(tex_save_path)
#         tex_save_path.mkdir(parents=True, exist_ok=True)


#     figure_str = r"""\begin{figure}[htp]
#     \centering
#         content
#     \caption{main_caption}
#     \label{main_label}
# \end{figure}
# """
    
#     subfig_str = r"""\begin{subfigure}[b]{0.32\textwidth}
#     \includegraphics[width=0.99\linewidth]{subfig_fname}
#     \caption{subfig_caption}
#     \label{subfig_label}
# \end{subfigure}"""
        

#     content = ''
#     i=1
#     for subfig_fname, subfig_caption, subfig_label in zip(subfig_fnames, subfig_captions, subfig_labels):        
#         content += subfig_str.replace('subfig_fname',subfig_fname).replace('subfig_caption',subfig_caption).replace('subfig_label',subfig_label)
        
#         if i < len(subfig_fnames):
#             content+="\n" + r"""\hfil"""
#         i+=1

#     main_label = f'fig:{fname_stem}'
#     figure_str = figure_str.replace('content', content).replace('main_caption', main_caption).replace('main_label',main_label)
#     #print(figure_str)


#     tex_fname = Path(tex_save_path) / f'{fname_stem}.tex'    
#     f = open(tex_fname, "w")
#     f.write(figure_str)
#     f.close()   



# def create_latex_subfigure_v2(fname_stem, figs_save_path, subfig_fnames1, subfig_fnames2, subfig_captions, subfig_labels, main_caption=''):

#     if figs_save_path is not None:
#         figs_save_path = Path(figs_save_path)
#         figs_save_path.mkdir(parents=True, exist_ok=True)


#     figure_str = r"""\begin{figure}[htp]
#     \centering
#         content
#     \caption{main_caption}
#     \label{main_label}
# \end{figure}
# """
    
#     subfig_str = r"""\begin{subfigure}[b]{0.49\textwidth}
#     \includegraphics[width=0.49\linewidth]{subfig_fname1}
#     \includegraphics[width=0.49\linewidth]{subfig_fname2}
#     \caption{subfig_caption}
#     \label{subfig_label}
# \end{subfigure}"""
        

#     content = ''
#     i=1
#     for subfig_fname1, subfig_fname2, subfig_caption, subfig_label in zip(subfig_fnames1, subfig_fnames2, subfig_captions, subfig_labels):        
#         content += subfig_str.replace('subfig_fname1',subfig_fname1).replace('subfig_fname2',subfig_fname2).replace('subfig_caption',subfig_caption).replace('subfig_label',subfig_label)
        
#         if i < len(subfig_fnames1):
#             content+="\n" + r"""\hfil"""
#         i+=1

#     main_label = f'fig:{fname_stem}'
#     figure_str = figure_str.replace('content', content).replace('main_caption', main_caption).replace('main_label',main_label)
#     #print(figure_str)


#     tex_fname = Path(figs_save_path) / f'{fname_stem}.tex'    
#     f = open(tex_fname, "w")
#     f.write(figure_str)
#     f.close()   



# def create_latex_figure_v4(fname_stem, tex_base_path, results_path=None, caption=''):

#     if results_path is None:
#         results_path = Path('/home/apollo/data/report_results/')

#     figure_str = r"""\begin{figure}[htp]
#     \centering
#     \includegraphics[width=0.99\linewidth]{fig_fname_str}
#     \caption{caption_str}
#     \label{label_str}
# \end{figure}
# """

#     save_path = results_path / tex_base_path / 'tex'
#     save_path.mkdir(parents=True, exist_ok=True)

#     tex_fname = save_path / f'{fname_stem}.tex'    

#     fname = f'{fname_stem}.png'
#     fig_fname = f'{tex_base_path}/figs/{fname}'
#     label = f'fig:{fname_stem}'

#     figure_str = figure_str.replace('fig_fname_str', fig_fname).replace('caption_str', caption).replace('label_str',label)
 
#     f = open(tex_fname, "w")
#     f.write(figure_str)
#     f.close()   

#     # Also, create the save path for the figure if needed
#     save_path = results_path / tex_base_path / 'figs'
#     save_path.mkdir(parents=True, exist_ok=True)

#     return results_path / fig_fname


# def create_latex_figure_v3(fname_stem, tex_base_path, results_path=None, caption=''):

#     if results_path is None:
#         results_path = Path('/home/apollo/data/report_results/')

#     figure_str = r"""\begin{figure}[htp]
#     \centering
#     \includegraphics[width=0.99\linewidth]{fig_fname_str}
#     \caption{caption_str}
#     \label{label_str}
# \end{figure}
# """

#     save_path = results_path / tex_base_path / 'tex'
#     save_path.mkdir(parents=True, exist_ok=True)

#     tex_fname = save_path / f'{fname_stem}.tex'    

#     fname = f'{fname_stem}.png'
#     fig_fname = f'{tex_base_path}/figs/{fname}'
#     label = f'fig:{fname_stem}'

#     figure_str = figure_str.replace('fig_fname_str', fig_fname).replace('caption_str', caption).replace('label_str',label)
 
#     f = open(tex_fname, "w")
#     f.write(figure_str)
#     f.close()   


# def compare_models_by_df_create_subfig_v2(df_ungrouped, df_column, patent_ids=[], 
#                                           figs_save_path=None, cwd_path=None, tex_base_path='./Figures/experiments',  \
#                                           sort_column = 'loss_function', \
#                                           fname_stem='compare_', \
#                                           main_caption='', verbose=False):

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')
    
#     assert figs_save_path is not None

#     figs_save_path = Path(figs_save_path)
#     figs_save_path.mkdir(parents=True, exist_ok=True)


#     subfig_fnames1 = []
#     subfig_fnames2 = []
#     subfig_captions = []
#     subfig_labels = []

#     df_grouped = df_ungrouped.sort_values(sort_column)
#     for group_name, df_group in df_grouped.groupby(sort_column):
#         pid_count=0
#         for patient_id in patent_ids:

#             if pid_count==2:
#                 break

#             dsc_values = []
#             subfig_fnames_inner = []
#             for include_T1 in [False, True]:
#                 df = df_group[df_group['include_T1']==include_T1].copy().reset_index(drop=True)

#                 if df.empty:
#                     break

#                 idx=0
#                 project_name = df.loc[idx]['project_name']
#                 network_name = df.loc[idx]['network_name']
#                 identifier = df.loc[idx][df_column].lower()
#                 identifier = ''.join(i for i in identifier if i.isalnum())

#                 caption_column_value = df.loc[idx][sort_column]

#                 config_fname = cwd_path / f"configs/{project_name}.json"
#                 config = get_wandb_config_from_json(config_fname)
#                 config = update_config_from_df(config, df_series = df.loc[idx])  

#                 wa = WandbAnalysis(project_name=project_name) 

#                 f, ax = plt.subplots(1,1, figsize=(16,10))

#                 # config['name'] = df.loc[idx]['name']
#                 # config['model_name'] = df.loc[idx]['model_name']
#                 filter_by_patients = [patient_id]
#                 caption_ = wa.show_networks_predictions_v2(config, ax, filter_by_patients, verbose=verbose)

#                 if caption_ is None:
#                     plt.close()
#                     break

#                 split_caption = caption_.split(':')

#                 if float(split_caption[-1]) < 0.67:
#                     plt.close()
#                     break

#                 fname = f'{fname_stem}_{patient_id}_{identifier}_{idx}.png'
#                 save_fname = figs_save_path / fname
#                 f.savefig(save_fname, bbox_inches='tight')

#                 dsc_values.append(split_caption[-1])
#                 subfig_fnames_inner.append(f'{tex_base_path}/{fname}')

#             if len(subfig_fnames_inner) != 2:
#                 continue
#             caption = f'No.: {patient_id}, {caption_column_value}, ({dsc_values[0]}/{dsc_values[1]})'
#             subfig_captions.append(caption)
#             subfig_labels.append(f'fig:{fname}')
#             subfig_fnames1.append(subfig_fnames_inner[0])
#             subfig_fnames2.append(subfig_fnames_inner[1])
            
#             if verbose: print(f'{tex_base_path}/{fname}')
#             if verbose: print(caption)
#             if verbose: print(f'fig:{fname}')
#             if verbose: print(f'fig:{save_fname}')

#             pid_count+=1
        
#     create_latex_subfigure_v2(fname_stem, figs_save_path, subfig_fnames1, subfig_fnames2, subfig_captions, subfig_labels, main_caption)


# def compare_models_by_df_create_subfig(df, df_column, patent_ids=[], figs_save_path=None,  \
#                                     fname_stem='compare_', tex_base_path='./Figures/experiments', \
#                                     main_caption='', cwd_path=None, include_pid_in_caption_=False, verbose=False):

#     if cwd_path is None:
#         cwd_path = Path('/home/apollo/code/mproj7205')
    
#     if figs_save_path is not None:
#         figs_save_path = Path(figs_save_path)
#         figs_save_path.mkdir(parents=True, exist_ok=True)

#     subfig_fnames = []
#     subfig_captions = []
#     subfig_labels = []

#     for idx in df.index:
#         project_name = df.loc[idx]['project_name']
#         identifier = df.loc[idx][df_column].lower()
#         identifier = ''.join(i for i in identifier if i.isalnum())

#         config_fname = cwd_path / f"configs/{project_name}.json"
#         config = get_wandb_config_from_json(config_fname)
#         config = update_config_from_df(config, df_series = df.loc[idx])  

#         wa = WandbAnalysis(project_name=project_name) 

#         for patient_id in patent_ids:

#             filter_by_patients = [patient_id]

#             f, ax = plt.subplots(1,1, figsize=(16,10))

#             config['name'] = df.loc[idx]['name']
#             config['model_name'] = df.loc[idx]['model_name']
#             caption = wa.show_networks_predictions_v2(config, ax, filter_by_patients, verbose=verbose)

#             if caption is None:
#                 plt.close()
#                 continue

#             if figs_save_path is not None:

#                 fname = f'{fname_stem}_{patient_id}_{identifier}_{idx}.png'
#                 save_fname = figs_save_path / fname

#                 f.savefig(save_fname, bbox_inches='tight')

#                 subfig_fnames.append(f'{tex_base_path}/{fname}')
#                 if include_pid_in_caption_:
#                     caption = f'{patient_id}, {caption}'
#                 subfig_captions.append(caption)
#                 subfig_labels.append(f'fig:{fname}')

#                 if verbose: print(f'{tex_base_path}/{fname}')
#                 if verbose: print(caption)
#                 if verbose: print(f'fig:{fname}')
#                 if verbose: print(f'fig:{save_fname}')

#     if figs_save_path is not None:
#         create_latex_subfigure(fname_stem, figs_save_path, subfig_fnames, subfig_captions, subfig_labels, main_caption)

# def create_labeller_table(clear_statistics=True):

#     rm = ResultsManager(results_df_path=f'/home/apollo/data/report_results/labeller_statistics.csv')
#     if clear_statistics:
#         rm.clear_results_df()

#     multi_SN_RN_path = Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/')

#     npy_path_0 = multi_SN_RN_path / "0/angus/"
#     npy_path_1 = multi_SN_RN_path / "1/ben/"

#     npy_1 = os.listdir(npy_path_0 / 'labels')
#     npy_2 = os.listdir(npy_path_1 / 'labels')

#     validation_patient_ids = get_validation_patient_ids()

#     both_ = [x for x in npy_1 if x in npy_2]

#     # Remove 3607 from this list
#     both_ = [x for x in both_ if '3828' not in str(x)]
#     both_ = [x for x in both_ if '3607' not in str(x)]

#     get_pid = lambda x: int(re.search(r'label_(.*)_(.*)', str(x)).group(1))

#     phases = ['Train', 'Validation', 'Combined']
#     train_val_dict={}
#     for phase in phases:

#         if phase == 'Train':
#             both = [x for x in both_ if get_pid(x) not in validation_patient_ids]
#         elif phase == 'Validation':
#             both = [x for x in both_ if get_pid(x) in validation_patient_ids]
#         else:
#             both = both_[:]

#         labeller1_only = [x for x in npy_1 if x not in both]
#         labeller2_only = [x for x in npy_2 if x not in both]

#         data = get_labeller_data(both)

#         df = pd.DataFrame(data=data, columns=["patient_id", "TP", "FN", "FP", "DSC", "IOU"])
#         df = df.sort_values(['DSC'], ascending=False)
#         df = df.astype({'patient_id': 'int32'})

#         df['DSC'].mean()
#         df['DSC'].median()

#         rm.add_to_results_df(phase = phase, \
#                              labeller1_total=len(labeller1_only)+len(both), \
#                              labeller1_only=len(labeller1_only), \
#                              labeller2_only=len(labeller2_only), \
#                              both=len(both),
#                              baseline_mean = df['DSC'].mean(), \
#                              baseline_median = df['DSC'].median())



# def write_latex_table2D(df,
#                         table_name='table1',
#                         caption='Caption of the table',
#                         column_format=None,
#                         multirow=False,
#                         multicolumn=False,
#                         multicolumn_format=None,
#                         float_format='{:d}'.format):
#     tables_path = Path('/home/apollo/data/report_results/Tables')
#     tables_path.mkdir(parents=True, exist_ok=True)

#     with open( tables_path / f'{table_name}.tex', 'w') as file: 
#         file.write(df.to_latex(index=True,
#             caption=caption,
#             escape=True,
#             column_format=column_format,
#             label=f'tab:{table_name}',
#             float_format=float_format,                                             
#             multirow=multirow,
#             multicolumn=multicolumn,
#             multicolumn_format=multicolumn_format)
#         )


# def get_labeller_table():
#     rm = ResultsManager(results_df_path=f'/home/apollo/data/report_results/labeller_statistics.csv')
#     df = rm.load_results_df().T
#     df = df.rename(columns=df.iloc[0])
#     df = df.drop(index='phase')
#     return df

# def get_labeller_data(both):

#     multi_SN_RN_path = Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/')

#     npy_path_0 = multi_SN_RN_path / "0/angus/"
#     npy_path_1 = multi_SN_RN_path / "1/ben/"

#     data = None
#     for label_name in both:
        
#         label_ = npy_path_0 / 'labels' / label_name
#         prediction_ = npy_path_1 / 'labels' / label_name

#         regex = re.search(r'label_(.*)_(.*)', label_name)
#         if regex:
#             patient_id = int(regex.group(1))
#         else:
#             patient_id = ''

#         title_prefix=patient_id
#         model_name=''


#         if label_.is_file() and prediction_.is_file():
            
#             label_ = torch.from_numpy(np.load(label_))
#             prediction_ = torch.from_numpy(np.load(prediction_))

#             label_ = np.squeeze(label_.int().cpu().numpy())
#             prediction_ = np.squeeze(prediction_.int().cpu().numpy())
                
#             true_pred, false_pred = label_ == prediction_, label_ != prediction_
#             pos_pred, neg_pred = prediction_ == 1, prediction_ == 0

#             tp = (true_pred * pos_pred)
#             fp = (false_pred * pos_pred)

#             tn = (true_pred * neg_pred)
#             fn = (false_pred * neg_pred)
            
#             epsilon=0
#             dsc =  2 * (tp.sum() + epsilon) / ( prediction_.sum()+label_.sum()+epsilon )
#             iou =  (tp.sum() + epsilon) / ( tp.sum()+fn.sum()+fn.sum()+epsilon )

#             if dsc < 0.5:
#                 print(f'{title_prefix}, TP:{tp.sum()}, FN:{fn.sum()}, FP:{fp.sum()}, DSC:{dsc:.2f}')

#             to_append = np.array([int(patient_id), tp.sum(), fn.sum(), fp.sum(), dsc, iou])
#             if isinstance(data, np.ndarray):
#                 data = np.vstack([data, to_append])
#             else:
#                 data = to_append

#     return data


def get_compare_labellers_df_v3(patent_ids=[], multi_SN_RN_path=None, verbose=False):
    data = compare_labellers_statistics_v3(patent_ids=patent_ids, multi_SN_RN_path=multi_SN_RN_path, verbose=verbose)

    df = pd.DataFrame(data=data, columns=["patient_id", "TP", "FN", "FP", "DSC", "IOU"])
    df = df.sort_values(['DSC'], ascending=False)
    df = df.astype({'patient_id': 'int32'})

    return df

def compare_labellers_statistics_v3(patent_ids=[], multi_SN_RN_path=None, verbose=False):
    """
    patient_ids: list of patient ids. If not set, all examples with two labels will be printed.

    """

    if multi_SN_RN_path is None:
        multi_SN_RN_path = Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/')

    npy_path_0 = multi_SN_RN_path / "0/angus/"
    npy_path_1 = multi_SN_RN_path / "1/ben/"

    npy_1 = os.listdir(npy_path_0 / 'labels')
    npy_2 = os.listdir(npy_path_1 / 'labels')

    both = [x for x in npy_1 if x in npy_2]

    # Remove 3607 from this list
    both = [x for x in both if '3828' not in str(x)]
    both = [x for x in both if '3607' not in str(x)]

    pid2label_dict = {}
    for i, k in enumerate(both):
        regex = re.search(r'label_(.*)_(.*)', k)
        if regex:
            patient_id = int(regex.group(1))
            pid2label_dict[patient_id] = i

    if patent_ids:
        # Only get the patient ids that exist in both
        patent_ids = [x for x in patent_ids if x in both]

        # Get the index from patient ids
        index_list = [pid2label_dict[i] for i in patent_ids]
    else:
        index_list = list(pid2label_dict.values())
        if verbose: print(f'total number of images with multiple labellers: {len(both)}')


    data = None
    for idx in index_list:
        
        label_ = npy_path_0 / 'labels' / both[idx]
        prediction_ = npy_path_1 / 'labels' /  both[idx]
        original_ = npy_path_0 / 'images' /  both[idx].replace('label', 'image')

        regex = re.search(r'label_(.*)_(.*)', both[idx])
        if regex:
            patient_id = int(regex.group(1))
        else:
            patient_id = ''

        title_prefix=patient_id
        model_name=''


        if label_.is_file() and prediction_.is_file() and original_.is_file():
            
            label_ = torch.from_numpy(np.load(label_))
            prediction_ = torch.from_numpy(np.load(prediction_))
            original_ = torch.from_numpy(np.load(original_))

            original_ = np.squeeze(original_.cpu().numpy())
            prediction_ = np.squeeze(prediction_.int().cpu().numpy())
            label_ = np.squeeze(label_.int().cpu().numpy())
                
            true_pred, false_pred = label_ == prediction_, label_ != prediction_
            pos_pred, neg_pred = prediction_ == 1, prediction_ == 0

            tp = (true_pred * pos_pred)
            fp = (false_pred * pos_pred)

            tn = (true_pred * neg_pred)
            fn = (false_pred * neg_pred)
            
            epsilon=0
            dsc =  2 * (tp.sum() + epsilon) / ( prediction_.sum()+label_.sum()+epsilon )
            iou =  (tp.sum() + epsilon) / ( tp.sum()+fn.sum()+fn.sum()+epsilon )

            if dsc < 0.5:
                print(f'{title_prefix}, TP:{tp.sum()}, FN:{fn.sum()}, FP:{fp.sum()}, DSC:{dsc:.2f}')

            to_append = np.array([int(patient_id), tp.sum(), fn.sum(), fp.sum(), dsc, iou])
            if isinstance(data, np.ndarray):
                data = np.vstack([data, to_append])
            else:
                data = to_append

    return data


# def get_labeller_data(both):

#     multi_SN_RN_path = Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/')

#     npy_path_0 = multi_SN_RN_path / "0/angus/"
#     npy_path_1 = multi_SN_RN_path / "1/ben/"

#     data = None
#     for label_name in both:
        
#         label_ = npy_path_0 / 'labels' / label_name
#         prediction_ = npy_path_1 / 'labels' / label_name

#         regex = re.search(r'label_(.*)_(.*)', label_name)
#         if regex:
#             patient_id = int(regex.group(1))
#         else:
#             patient_id = ''

#         title_prefix=patient_id
#         model_name=''


#         if label_.is_file() and prediction_.is_file():
            
#             label_ = torch.from_numpy(np.load(label_))
#             prediction_ = torch.from_numpy(np.load(prediction_))

#             label_ = np.squeeze(label_.int().cpu().numpy())
#             prediction_ = np.squeeze(prediction_.int().cpu().numpy())
                
#             true_pred, false_pred = label_ == prediction_, label_ != prediction_
#             pos_pred, neg_pred = prediction_ == 1, prediction_ == 0

#             tp = (true_pred * pos_pred)
#             fp = (false_pred * pos_pred)

#             tn = (true_pred * neg_pred)
#             fn = (false_pred * neg_pred)
            
#             epsilon=0
#             dsc =  2 * (tp.sum() + epsilon) / ( prediction_.sum()+label_.sum()+epsilon )
#             iou =  (tp.sum() + epsilon) / ( tp.sum()+fn.sum()+fn.sum()+epsilon )

#             if dsc < 0.5:
#                 print(f'{title_prefix}, TP:{tp.sum()}, FN:{fn.sum()}, FP:{fp.sum()}, DSC:{dsc:.2f}')

#             to_append = np.array([int(patient_id), tp.sum(), fn.sum(), fp.sum(), dsc, iou])
#             if isinstance(data, np.ndarray):
#                 data = np.vstack([data, to_append])
#             else:
#                 data = to_append

#     return data

# def get_validation_patient_ids(multi_SN_RN_path=None, verbose=False):
#     """
#     patient_ids: list of patient ids. If not set, all examples with two labels will be printed.

#     """

#     if multi_SN_RN_path is None:
#         multi_SN_RN_path = Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/')

#     npy_path_0 = multi_SN_RN_path / "0/angus/"
#     npy_path_1 = multi_SN_RN_path / "1/ben/"

#     npy_1 = os.listdir(npy_path_0 / 'labels')
#     npy_2 = os.listdir(npy_path_1 / 'labels')

#     both = [x for x in npy_1 if x in npy_2]

#     # Remove 3607 from this list
#     both = [x for x in both if '3828' not in str(x)]
#     both = [x for x in both if '3607' not in str(x)]

#     pid2label_dict = {}
#     for i, k in enumerate(both):
#         regex = re.search(r'label_(.*)_(.*)', k)
#         if regex:
#             patient_id = int(regex.group(1))
#             pid2label_dict[patient_id] = i

#     wa = WandbAnalysis() 

#     index_list = list(pid2label_dict.values())
#     if verbose: print(f'total number of images with multiple labellers: {len(both)}')

#     config_path = Path('/home/apollo/code/mproj7205/configs2/template.json')
#     config = read_json(config_path)
#     config['valid_batch_size']=1
#     config = Bunch(config)
#     _, val_loader = build_datasets(config, verbose=False)

#     validation_patient_ids=[]
#     for _, data in enumerate(val_loader):
#         patient_id_ = int(data['patient_id'][0])
#         if patient_id_ in pid2label_dict.keys():
#             validation_patient_ids.append(patient_id_)

#     return validation_patient_ids


# def compute_metrics_by_df_v3(df, x_column):
    
#     df_all = None
    
#     for idx in df.index:
#         df1 = compute_metrics_by_run(df, idx=idx, sweep_name=None)
#         #df1['include_T1_'] = df['include_T1'].replace({False: inlcude_T1_labels[0], True : inlcude_T1_labels[1]})
#         df1['include_T1_'] = df.loc[idx]['include_T1_']
#         df1['include_T1'] = df.loc[idx]['include_T1']
#         df1['loss_function'] = df.loc[idx]['loss_function']
#         df1['network_name'] = df.loc[idx]['network_name']
#         df1[x_column] = df.loc[idx][x_column]
#         if isinstance(df_all, pd.DataFrame):
#             df_all = pd.concat([df_all, df1], axis=0, ignore_index=True)
#         else:
#             df_all = df1
                        
#     return df_all



# def filter_invalid_project_names(project_names):
#     project_names = list(set(project_names))

#     project_names_ = project_names[:]
#     for project_name in project_names_:
#         try:
#             wa = WandbAnalysis(project_name=project_name)
#             df = wa.all_df      
#         except:
#             print(f'removing project_name {project_name}')
#             project_names.pop(project_name)

#     return project_names


# def create_columns_for_results(df):

#     df = df.copy()

#     inlcude_T1_labels=['T2 only', 'T1 and T2']  
#     df_keys = list(df.keys())
#     if 'include_T1' in df_keys:
#         df.reset_index(inplace=True, drop=True)
#         df['include_T1_'] = df['include_T1'].replace({False: inlcude_T1_labels[0], True : inlcude_T1_labels[1]})
    
#     replace_dict = {'TverskyLoss': 'Tversky Loss', 
#                     'TverskyFocalLoss': 'Tversky Focal Loss', 
#                     'BCELoss': 'BCE Loss', 
#                     'FocalLoss': 'Focal Loss', 
#                     'WeightedDiceLoss': 'Weighted Dice Loss', 
#                     'FBetaLoss' : 'F-Beta Loss', 
#                     'DiceLoss' : 'Dice Loss'}

#     if 'loss_function' in df_keys:
#         df.reset_index(inplace=True, drop=True)
#         df['loss_function_'] = df['loss_function'].replace(replace_dict)

#     replace_dict = {'UNet_2Plus': 'UNet ++', 'UNet_3Plus': 'UNet 3+', 'smp.Unet' : 'UNet (smp)'}
    
#     if 'network_name' in df_keys:
#         df.reset_index(inplace=True, drop=True)
#         df['network_name_'] = df['network_name'].replace(replace_dict)

#     return df


#     #df = df.drop_duplicates(subset=['loss_function', 'include_T1'], keep='first').copy()

# def annotate_patches(ax):
#     for bar in ax.patches:
#         #ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))    
        
#         ax.annotate(format(bar.get_height(), '.2f'),
#                        (bar.get_x() + bar.get_width() / 2,
#                         bar.get_height()), ha='center', va='center',
#                        size=15, xytext=(0, 8),
#                        textcoords='offset points')        

# def plot_models(df, x_column='name', hue_col='include_T1_', \
#                 sns_plot='sns.barplot', fig_fname=None, \
#                 x_label=None, y_label=None, label_fontsize=20, legend_fontsize=20, \
#                 axhline=None, show_annotations=True):

#     # e.g., x_label = r'Beta ($\beta$)'
    

#     if sns_plot != 'sns.histplot': df = create_columns_for_results(df)

#     #inlcude_T1_labels=['T2 only', 'T1 and T2']    

    
#     f, ax = plt.subplots(1,1, figsize=(16,10))
#     #ax = sns.lineplot(data=df_fbeta, x="beta", y="dsc", hue='include_T1')
#     if sns_plot == 'sns.barplot':
#         ax = sns.barplot(data=df, x=x_column, y="dsc", hue=hue_col)

#     elif sns_plot == 'sns.boxplot':
#         ax = sns.boxplot(data=df, x=x_column, y="dsc", hue=hue_col)

#     elif sns_plot == 'sns.histplot':
#         ax = sns.histplot(data=df, x=x_column, stat='density', kde=True)

#     if axhline:
#         ax.axhline(axhline, color='r', linestyle='--', linewidth=4)

#     if x_label:
#         ax.set_xlabel(x_label, fontsize=label_fontsize)
#     else:
#         ax.xaxis.label.set_size(label_fontsize)

#     if y_label:
#         ax.set_ylabel(y_label, fontsize=label_fontsize)
#     else:
#         ax.set_ylabel('Dice Score (DSC)', fontsize=label_fontsize)

#     if show_annotations:
#         annotate_patches(ax)
    

#     if sns_plot != 'sns.histplot': 
#         plt.legend(title='MRI Channels', loc='upper left', fontsize=legend_fontsize)
#         plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legend_fontsize)
    
#     if fig_fname:
#         f.savefig(fig_fname, bbox_inches='tight')

# def bring_specific_df_to_top(all_dfs_dict, df_name):
#     # append all other defs to end of df_name
    
#     df = all_dfs_dict[df_name]
#     for df_name_, df2 in all_dfs_dict.items():
        
#         if df_name_ != df_name:
#             df = pd.concat([df, df2], axis=0, ignore_index=True) 
            
#     return df

# # def append_all_to_end(all_dfs_dict, df_name):
# #     all_dfs_list = [all_dfs_dict[df_name]]
# #     for df_name_ in all_dfs_dict.keys():
# #         print(df_name_, df_name)
# #         if df_name_ != df_name:
# #             print('true')
# #             all_dfs_list.append(all_dfs_dict[df_name_])
            
# #     return all_dfs_list

# def save_all_dfs_dict(all_dfs_dict):
#     dfs_path = Path('/home/apollo/data/report_results/dfs')
#     dfs_path.mkdir(parents=True, exist_ok=True)

#     for df_name_, df in all_dfs_dict.items():
#         # Save manual changes to the df
#         df_path = dfs_path / f'{df_name_}.csv'
#         df.to_csv(df_path, index=True)

#     all_dfs_dict_path = dfs_path / 'all_dfs_dict.json'
#     all_dfs_dict = all_dfs_dict.copy()
#     all_dfs_dict = {key: None for key in all_dfs_dict}

#     write_json(all_dfs_dict, dfs_path / all_dfs_dict_path)


# def load_all_dfs_dict(keep='first'):
#     # keeps either the first (order of model folders) or keep='best' uses dsc

#     if keep == 'best':
#         keep_best = True
#     else:
#         keep_best = False

#     dfs_path = Path('/home/apollo/data/report_results/dfs')
#     all_dfs_dict_path = dfs_path / 'all_dfs_dict.json'
#     all_dfs_dict = read_json(all_dfs_dict_path)
#     for df_name_, _ in all_dfs_dict.items():
#         df_path = dfs_path / f'{df_name_}.csv'
#         if df_path.is_file():
#             df = pd.read_csv(df_path, index_col=0)

#             if keep_best:
#                 # Re-order by best dsc (by default re-orders by the order which rowss were added)
#                 df = df.sort_values(['dsc'], ascending=False)

#             m = df.shape[0]
#             df = df.drop_duplicates(subset=['project_name', 'name'], keep='first').copy()
#             if (m-df.shape[0] > 0): print(f'Dropping {m-df.shape[0]} duplicate rows (models)')

#             all_dfs_dict[df_name_] = df

#     return all_dfs_dict

# def append_all_to_end_of_df(all_dfs_dict, df_name):
#     # append all other defs to end of df_name
    
#     df = all_dfs_dict[df_name]
#     for df_name_, df2 in all_dfs_dict.items():
        
#         if df_name_ != df_name:
#             df = pd.concat([df, df2], axis=0, ignore_index=True) 
            
#     return df

# def reorder_all_dfs_dict(all_dfs_dict, desired_key_order, append_all=False):
#     valid_keys = list(all_dfs_dict.keys())

#     for k in desired_key_order:
#         assert k in valid_keys
#         valid_keys.remove(k)

#     # Add keys to the end
#     if append_all: desired_key_order += valid_keys
    
#     return {k: all_dfs_dict[k] for k in desired_key_order}


# def combine_dfs_in_desired_order(all_dfs_dict, desired_order, append_all=False):
#     # append all other defs to end of df_name
#     all_dfs_dict = all_dfs_dict.copy()

#     all_dfs_dict = reorder_all_dfs_dict(all_dfs_dict, desired_key_order=desired_order, append_all=append_all)

#     df_name_0 = desired_order[0]
#     df = all_dfs_dict[df_name_0]
#     for df_name_, df2 in all_dfs_dict.items():
#         if df_name_ != df_name_0:
#             df = pd.concat([df, df2], axis=0, ignore_index=True) 
            
#     return df

# def get_label_uncertainty_caption(patient_id=3300):
#     patient_id = int(patient_id)

#     labels = os.listdir(Path('/home/apollo/data/multi_SN_RN/npy_2d/SN/0/angus/labels'))

#     fname = [x for x in labels if str(patient_id) in x][0]
#     labeller_0 = Path(f'/home/apollo/data/multi_SN_RN/npy_2d/SN/0/angus/labels/{fname}')
#     labeller_1 = Path(f'/home/apollo/data/multi_SN_RN/npy_2d/SN/1/ben/labels/{fname}')
#     if labeller_0.is_file() and labeller_1.is_file():
#         label_ = np.squeeze(np.load(labeller_0))
#         prediction_ = np.squeeze(np.load(labeller_1))

#         true_pred, false_pred = label_ == prediction_, label_ != prediction_
#         pos_pred, neg_pred = prediction_ == 1, prediction_ == 0

#         tp = (true_pred * pos_pred)
#         fp = (false_pred * pos_pred)

#         tn = (true_pred * neg_pred)
#         fn = (false_pred * neg_pred)
        
#         epsilon=0
#         dsc =  2 * (tp.sum() + epsilon) / ( prediction_.sum()+label_.sum()+epsilon )
        
#         title = f'TP:{tp.sum()}, FN:{fn.sum()}, FP:{fp.sum()}, DSC:{dsc:.2f}'
#         caption = f'DSC={dsc:.2f} ({tp.sum()}/{fn.sum()}/{fp.sum()})'

#     return caption

# def create_augmentation_columns(df):
#     #df['augmentation'] = df['network_name'] + df['RandomAffine'].isna().astype('str')
    
#     df['augmentation'] = df['RandomAffine'].isna()
    
#     replace_dict = {False: 'No augmentation', True : 'Augmentation'}
    
#     df.reset_index(inplace=True, drop=True)
#     df['augmentation_'] = df['augmentation'].replace(replace_dict)

#     replace_dict = {'UNetFalse': 'UNet (no augmentation)', 'UNetTrue' : 'UNet (augmentation)',
#                     'UNet_2PlusFalse': 'UNet ++ (no augmentation)', 'UNet_2PlusTrue': 'UNet ++ (augmentation)', 
#                     'UNet_3PlusFalse': 'UNet 3+ (no augmentation)', 'UNet_3PlusTrue': 'UNet 3+ (augmentation)', 
#                     'smp.UnetFalse' : 'UNet (smp) (no augmentation)', 'smp.UnetTrue' : 'UNet (smp) (augmentation)'
#                     }

#     df.reset_index(inplace=True, drop=True)
#     df['augmentation_network_'] = ( df['network_name'] + df['RandomAffine'].isna().astype('str')).replace(replace_dict)

#     return df

