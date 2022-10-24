import wandb
api = wandb.Api()
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

from utils.utils import Bunch
from model.train import *
from utils.results_manager import ResultsManager
# from model.build_networks import *
# from model.build_loss_criterion import *

class WandbAnalysis(object):

    def __init__(self, project_name=None):

        # Change oreilly-class/cifar to <entity/project-name>
        if project_name is None:
            return

        self.runs = api.runs(project_name)
        summary_list = [] 
        config_list = [] 
        name_list = [] 
        for run in self.runs: 
            # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
            summary_list.append(run.summary._json_dict) 

            # run.config is the input metrics.  We remove special values that start with _.
            config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 

            # run.name is the name of the run.
            name_list.append(run.name)       

        self.summary_df = pd.DataFrame.from_records(summary_list) 
        self.config_df = pd.DataFrame.from_records(config_list) 
        self.name_df = pd.DataFrame({'name': name_list}) 
        df = pd.concat([self.name_df, self.config_df, self.summary_df], axis=1)
        self.all_df = self.append_random_noise_column(df)

        self.name2id = {}        
        self.set_wandb_runid()
        self.all_df['wandb_id'] = self.all_df['name'].apply(lambda x: self.name2id[x])

        self.run_dir = Path('/Users/hao/MRI_Research_Project/mproj7205/wandb')
        self.dataset_dir = Path(self.all_df.iloc[0]['dataset_dir'])

        # Used in dice calculation in practice
        self.smooth = 1.0

        self.crop_shape = [160,160]

    def get_run_idx(self, run_name):
        return int(self.name_df[self.name_df['name']==run_name].index.values)

    def set_wandb_runid(self):
        for run in self.runs: 
            self.name2id.update({run.name: run.id})

    def append_random_noise_column(self, df):
        """ Add random noise column
            We have incorporated the Random Noise field inside the Normalization Method. 
            We need to extract wherever that occurs into a seperate field.
        """

        df['RandomNoise'] = None
        #row_idx = 0
        for row_idx in range(df.shape[0]):
            for k, v in df.loc[row_idx, 'Normalization'].items():
                if k=='RandomNoise':
                    #print(f'Random noise used in {df.loc[row_idx, "name"]}')
                    df.loc[row_idx,'RandomNoise'] = str(v)
            
        return df
    
    def get_history(self, run_idx=None, run_name=None):
    
        assert run_idx != run_name # Cant both be set
        
        if run_name is not None:
            run_idx = self.get_run_idx(run_name)

        df = self.runs[run_idx].history(pandas=True)
        
        return df[df['dice score (val)'].notna()]

    def plot_history(self, h, ax, y_col='loss', title='', fontsize=12, xlim=None):

        ax.plot(h['epoch'], h[y_col])
        #ax.legend((f"train {train_}", f"test {train_}"))
        ax.set_xlabel('epoch', fontsize=fontsize)
        ax.set_ylabel(y_col, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        if xlim is not None:
            ax.set_xlim(xlim)


    def plot_history_from_runs(self, runs_list, dataset='train', save_fname=None, \
                               annotate_xy=None, xlim=None, fontsize=12, **kwargs):

        f, ax = plt.subplots(1, 1, **kwargs)

        y_col = f'dice score ({dataset})'
            
        labels = []
        for run_name in runs_list:
            run_idx = self.get_run_idx(run_name)
            h = self.get_history(run_idx)
            self.plot_history(h, ax, y_col=y_col, fontsize=fontsize, xlim=xlim)
            labels.append(f'{run_name}')
        
        ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize-4)
        if annotate_xy is not None:
            ax.annotate('selected baseline experiment', size=15, color='red', xy=(70, 0.22), xytext=(160, 0.6), \
                        arrowprops=dict(facecolor='red', shrink=0.05),)
            #ax.annotate('Text 1', xy=annotate_xy, xytext=(500,0.8), arrowprops=dict(arrowstyle='->'))
        #ax.legend(labels, fontsize=fontsize-2)
        f.tight_layout()
        f.show()
        
        if save_fname is not None:
            f.savefig(save_fname)
        

    def get_available_patient_ids(self, run_name):
        
        run_idx = self.get_run_idx(run_name)
        run = self.runs[run_idx]

        run_summary_df = pd.DataFrame.from_records(run.summary._json_dict) 
        
        return run_summary_df.loc['captions']['Predicted masks (validation)']


    # def get_original_nii(self, run_name ='chocolate-sweep-7', show_plot=True):
        
    #     sub_dir = 'images'
    #     nii_image = self.get_nii_image_by_patient_id(patient_id=patient_id)
        
    #     img = nib.load(nii_image)
    #     image = img.get_fdata()
        
    #     data = np.rot90(image.squeeze().T)
        
    #     if show_plot:
    #         f, ax = plt.subplots(1,1, figsize=(16,10))
    #         ax.imshow(data, cmap="gray")
            
    #     return data

    def get_nii_image_by_patient_id(self, patient_id):

        sub_dir = 'images'
        regex_compile = re.compile("(?<=PPMI_)(.*)(?=_MR)")

        images_dir = self.dataset_dir / sub_dir
        for image_path in images_dir.iterdir():
            lookup_patient_id = regex_compile.search(str(image_path)).group(1)
            if lookup_patient_id == patient_id:
                return image_path

    def get_masks_from_run(self, run):

        for image_path in self.run_dir.iterdir():
            if str(image_path).endswith(run.id):
                return image_path            
                
    def get_run_masks(self, run_name, patient_id):
        
        run_idx = self.get_run_idx(run_name)
        run = self.runs[run_idx]

        summary_df = pd.DataFrame.from_records(run.summary._json_dict) 
        lookup_id = summary_df.loc['captions']['Predicted masks (validation)'].index(patient_id)
        results_dict = summary_df.loc['all_masks']['Predicted masks (validation)'][lookup_id]

        intersection = results_dict['intersection']['path'] 
        ground_truth= results_dict['ground truth']['path'] 
        prediction = results_dict['prediction']['path']  
        original = prediction.replace('0.mask', '').replace('/mask','')
        
        assert prediction.replace('0.mask', '') == ground_truth.replace('01.mask', '')
        assert ground_truth.replace('01.mask', '') == intersection.replace('012.mask', '')
            
        return original, intersection, ground_truth, prediction


    def get_cropped_window_dims(self, mask, pad=4):
        df = pd.DataFrame(mask)
        df = df.loc[(df!=0).any(axis=1)]
        df = df.T
        df = df.loc[(df!=0).any(axis=1)]
        df = df.T
        
        return df.columns[0]-pad, df.columns[-1]+pad, df.index[0]-pad, df.index[-1]+pad
        

    def plot_three_results_from_run(self, run_name, patient_id):

        sub_dir = 'files'
        original, intersection, ground_truth, prediction = self.get_run_masks(run_name, patient_id) 
        run_idx = self.get_run_idx(run_name) 
        run = self.runs[run_idx]          
        run_dir = self.get_masks_from_run(run)

        original_path = run_dir / sub_dir / original
        intersection_path = run_dir / sub_dir /  intersection
        ground_truth_path = run_dir / sub_dir /  ground_truth
        prediction_path = run_dir / sub_dir /  prediction

        original_ = plt.imread(original_path)
        intersection_ = plt.imread(intersection_path)
        ground_truth_ = plt.imread(ground_truth_path)
        prediction_ = plt.imread(prediction_path)

        class_labels = {
        0: "Negative",
        2: "Intersection",
        3: "Ground truth (only)",
        4: "Predicted (only)"
        }

        _map_masks = lambda x, val: np.rot90(np.ma.masked_where(np.round(x*255) != val, x))

        TPs = (~ _map_masks(intersection_, 2).mask).sum()
        FNs = (~ _map_masks(intersection_, 3).mask).sum()
        FPs = (~ _map_masks(intersection_, 4).mask).sum()
        TNs = original_.shape[0]*original_.shape[1] - TPs - FNs - FPs
        
        dice = 1- (2*(TPs)+self.smooth) / (2*(TPs) + FNs + FPs+self.smooth)
        title = f'TPs:{TPs}, FNs:{FNs}, FPs:{FPs}, Loss:{dice:.2f}'

        intersection_=intersection_*255/4

        colour_list = ['white', 'green', 'orange', 'red']
        cmap = colors.ListedColormap(colour_list)
        bounds=[0., 0.5, 0.75, 1., 1.25]
        norm = colors.BoundaryNorm(bounds, cmap.N) 

        alphas = np.ones(np.rot90(intersection_).shape)*.4
        alphas[np.rot90(intersection_)==0.] = 0

        f, ax = plt.subplots(1,3, figsize=(16,10))

        ax[0].imshow(np.rot90(original_), cmap="gray")
        ax[0].imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)
        ax[0].set_title(f'id: {patient_id}, dice:{dice:.2f}')
        ax[0].axis('off')

        ax[1].imshow(np.rot90(original_), cmap="gray")
        ax[1].imshow( _map_masks(ground_truth_, 1), cmap='RdYlGn', alpha=0.5)
        ax[1].set_title(f'Ground truth mask: id: {patient_id}, dice:{dice:.2f}')
        ax[1].axis('off')

        ax[2].imshow(np.rot90(original_), cmap="gray")
        ax[2].imshow( _map_masks(prediction_, 1), cmap='RdYlGn', alpha=0.5)
        ax[2].set_title(f'Predicted mask: id: {patient_id}, dice:{dice:.2f}')
        ax[2].axis('off')

    def non_linear_map(self, input):
        input = np.round(input*255).astype(int)
        map_val = 0
        for val in np.unique(input):
            input = np.where(input ==  val, map_val, input)
            map_val+=1

        return input

    def plot_results_from_run2(self, run_name, patient_id, show_both_masks=False, \
                              zoom_inset=None, save_fname=None):

        sub_dir = 'files'
        original, intersection, ground_truth, prediction = self.get_run_masks(run_name, patient_id) 
        run_idx = self.get_run_idx(run_name) 
        run = self.runs[run_idx]          
        run_dir = self.get_masks_from_run(run)

        original_path = run_dir / sub_dir / original
        intersection_path = run_dir / sub_dir /  intersection
        ground_truth_path = run_dir / sub_dir /  ground_truth
        prediction_path = run_dir / sub_dir /  prediction

        original_ = plt.imread(original_path)
        intersection_ = plt.imread(intersection_path)
        ground_truth_ = plt.imread(ground_truth_path)
        prediction_ = plt.imread(prediction_path)

        class_labels = {
        0: "Negative",
        2: "Intersection",
        3: "Ground truth (only)",
        4: "Predicted (only)"
        }

        _map_masks = lambda x, val: np.rot90(np.ma.masked_where(np.round(x*255) != val, x))

        TPs = (~ _map_masks(intersection_, 2).mask).sum()
        FNs = (~ _map_masks(intersection_, 3).mask).sum()
        FPs = (~ _map_masks(intersection_, 4).mask).sum()
        TNs = original_.shape[0]*original_.shape[1] - TPs - FNs - FPs
        
        #dice = 1- 2*(TPs) / (2*(TPs) + FNs + FPs)
        dice = 1- (2*(TPs)+self.smooth) / (2*(TPs) + FNs + FPs+self.smooth)

        title = f'TPs:{TPs}, FNs:{FNs}, FPs:{FPs}, Loss:{dice:.2f}'

        intersection_ = self.non_linear_map(intersection_)

        colour_list = ['white', 'green', 'orange', 'red']
        cmap = colors.ListedColormap(colour_list)
        bounds=[0, 10, 20, 30, 40]
        norm = colors.BoundaryNorm(bounds, cmap.N) 

        alphas = np.ones(np.rot90(intersection_).shape)*1.
        alphas[np.rot90(intersection_)==0.] = 0
                    
        #bounds=[0., 0.00784314, 0.01176471, 0.01568628]
        
        f, ax = plt.subplots(1,1, figsize=(16,10))

        if zoom_inset is None:            

            ax.imshow(np.rot90(original_), cmap="gray")
            ax.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)
            ax.set_title(title)
            ax.axis('off')

        else:
            ax.imshow(np.rot90(original_), cmap="gray")

            if show_both_masks:
                ax.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)

            _flip_rot = lambda x: np.flipud(np.rot90(x))

            pop_a = mpatches.Patch(color=colour_list[1], label='true positives')
            pop_b = mpatches.Patch(color=colour_list[2], label='false negatives')
            pop_c = mpatches.Patch(color=colour_list[3], label='false positives')

            plt.legend(handles=[pop_a,pop_b,pop_c], loc=4)

            x1, x2, y1, y2 = self.get_cropped_window_dims(_flip_rot(intersection_))

            axins = zoomed_inset_axes(ax, zoom_inset, loc=9) # zoom-factor: 2.5, location: upper-left
            axins.imshow(_flip_rot(original_), cmap="gray")
            axins.imshow(_flip_rot(intersection_), alpha=np.flipud(alphas), cmap=cmap, norm=norm)

            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y1, y2) # apply the y-limits
            plt.yticks(visible=False)
            plt.xticks(visible=False)

            mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="blue")

            ax.set_title(title)
            ax.axis('off')

            if save_fname is not None:
                f.savefig(save_fname)


    def plot_all_results_from_run(self, run_name, show_both_masks=False, \
                                  transparency=0.5, zoom_inset=None, save_fname=None, \
                                  save_fname_path=None, legend_loc=4):

        patient_ids_by_run = self.get_available_patient_ids(run_name)

        for pid in patient_ids_by_run:
            if save_fname_path is not None:
                save_fname = save_fname_path / f"{run_name.replace('-','_')}_{pid}.png"

            self.plot_results_from_run(run_name, patient_id=pid, zoom_inset=zoom_inset, \
                                       transparency=transparency, show_both_masks=show_both_masks, save_fname=save_fname, legend_loc=legend_loc)


    def get_statistics_from_run(self, run_name, patient_id):

        sub_dir = 'files'
        original, intersection, ground_truth, prediction = self.get_run_masks(run_name, patient_id)

        run_idx = self.get_run_idx(run_name)
        run = self.runs[run_idx]
        run_dir = self.get_masks_from_run(run)

        original_path = run_dir / sub_dir / original
        intersection_path = run_dir / sub_dir /  intersection
        ground_truth_path = run_dir / sub_dir /  ground_truth
        prediction_path = run_dir / sub_dir /  prediction

        original_ = plt.imread(original_path)
        intersection_ = plt.imread(intersection_path)
        ground_truth_ = plt.imread(ground_truth_path)
        prediction_ = plt.imread(prediction_path)

        class_labels = {
        0: "Negative",
        2: "Intersection",
        3: "Ground truth (only)",
        4: "Predicted (only)"
        }

        _map_masks = lambda x, val: np.rot90(np.ma.masked_where(np.round(x*255) != val, x))

        TPs = (~ _map_masks(intersection_, 2).mask).sum()
        FNs = (~ _map_masks(intersection_, 3).mask).sum()
        FPs = (~ _map_masks(intersection_, 4).mask).sum()
        TNs = original_.shape[0]*original_.shape[1] - TPs - FNs - FPs
        
        dice = 1- (2*(TPs)+self.smooth) / (2*(TPs) + FNs + FPs+self.smooth)
        #dice = 1- 2*(TPs) / (2*(TPs) + FNs + FPs)
        
        return TPs, FNs, FPs, TNs, dice


    def get_statistics(self, df):
        
        df_columns = ['run_name', 'patient_id', 'TPs', 'FNs', 'FPs', 'TNs', 'dice']
        results_df = pd.DataFrame(columns=df_columns)
        
        for run_idx in df.index:
            run_name = df.loc[run_idx, 'name']
            patient_ids_by_run = self.get_available_patient_ids(run_name)
            for patient_id in patient_ids_by_run:
                
                TPs, FNs, FPs, TNs, dice = self.get_statistics_from_run(run_name, patient_id)
                values = [run_name, str(patient_id), float(TPs), float(FNs), float(FPs), float(TNs), float(dice)]
                row = pd.Series(values, index = df_columns)
                results_df = results_df.append(row, ignore_index=True)
                
            #print(df.loc[row_idx, 'name'])
            
        return results_df


    def display_statistics(self, results_df):

        pd.set_option('display.float_format','{:.2f}'.format)
        for run_name in results_df['run_name'].unique():
            #list_of_results, index_labels, column_labels, title

            print(f'Statistics for run:{run_name}')
            print(results_df[results_df['run_name'] == run_name].describe().loc[['mean','std','min','max']])
            print()
                
            #plot_results_as_table(results_df[results_df['run_name'] == run_name].describe().loc[['mean','std','min','max']],
            #                 title=f'{run_name}')

    def get_statistics_by_run(self, results_df, run_name):
        return results_df[results_df['run_name'] == run_name].describe().loc[['mean','std','min','max']]


    def plot_results_from_run(self, run_name, patient_id, show_both_masks=False, \
                              transparency=0.5, zoom_inset=None, save_fname=None, legend_loc=4):

        sub_dir = 'files'
        original, intersection, ground_truth, prediction = self.get_run_masks(run_name, patient_id) 
        run_idx = self.get_run_idx(run_name) 
        run = self.runs[run_idx]          
        run_dir = self.get_masks_from_run(run)

        original_path = run_dir / sub_dir / original
        intersection_path = run_dir / sub_dir /  intersection
        ground_truth_path = run_dir / sub_dir /  ground_truth
        prediction_path = run_dir / sub_dir /  prediction

        original_ = plt.imread(original_path)
        intersection_ = plt.imread(intersection_path)
        ground_truth_ = plt.imread(ground_truth_path)
        prediction_ = plt.imread(prediction_path)

        class_labels = {
        0: "Negative",
        2: "Intersection",
        3: "Ground truth (only)",
        4: "Predicted (only)"
        }

        _map_masks = lambda x, val: np.rot90(np.ma.masked_where(np.round(x*255) != val, x))

        TPs = (~ _map_masks(intersection_, 2).mask).sum()
        FNs = (~ _map_masks(intersection_, 3).mask).sum()
        FPs = (~ _map_masks(intersection_, 4).mask).sum()
        TNs = original_.shape[0]*original_.shape[1] - TPs - FNs - FPs
        
        #dice = 1- 2*(TPs) / (2*(TPs) + FNs + FPs)
        dice = 1- (2*(TPs)+self.smooth) / (2*(TPs) + FNs + FPs+self.smooth)

        title = f'{run_name}: TPs:{TPs}, FNs:{FNs}, FPs:{FPs}, Loss:{dice:.2f}'

        intersection_ = self.non_linear_map(intersection_)

        colour_list = ['white', 'green', 'orange', 'red']
        cmap = colors.ListedColormap(colour_list)
        bounds=[0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N) 

        alphas = np.ones(np.rot90(intersection_).shape)*transparency
        alphas[np.rot90(intersection_)==0] = 0
                    
        #bounds=[0., 0.00784314, 0.01176471, 0.01568628]
        
        f, ax = plt.subplots(1,1, figsize=(16,10))

        if zoom_inset is None:            

            ax.imshow(np.rot90(original_), cmap="gray")
            ax.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)
            ax.set_title(title, fontsize=30)
            ax.axis('off')

        else:
            ax.imshow(np.rot90(original_), cmap="gray")

            if show_both_masks:
                ax.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)

            #_flip_rot = lambda x: np.flipud(np.rot90(x))

            pop_a = mpatches.Patch(color=colour_list[1], label='true positives')
            pop_b = mpatches.Patch(color=colour_list[2], label='false negatives')
            pop_c = mpatches.Patch(color=colour_list[3], label='false positives')

            if isinstance(legend_loc, int):
                plt.legend(handles=[pop_a,pop_b,pop_c], loc=legend_loc)
            else:
                 plt.legend(handles=[pop_a,pop_b,pop_c], bbox_to_anchor=(1, 1), loc='upper left')

            x1, x2, y1, y2 = self.get_cropped_window_dims(np.rot90(intersection_))
            #x1, x2, y1, y2 = self.get_cropped_window_dims(np.rot90(intersection_))

            axins = zoomed_inset_axes(ax, zoom_inset, loc=9) # zoom-factor: 2.5, location: upper-left
            ax.imshow(np.rot90(original_), cmap="gray")

            if show_both_masks:            
                ax.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)

            axins.imshow(np.rot90(original_), cmap="gray")
            axins.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)
            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y2, y1) # apply the y-limits
            plt.yticks(visible=False)
            plt.xticks(visible=False)

            mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="1.0")

            ax.set_title(title, fontsize=26)
            ax.axis('off')

        if save_fname is not None:
            f.savefig(save_fname)

    def plot_ground_truth_only(self, run_name, patient_id, show_both_masks=False, \
                              transparency=0.5, zoom_inset=None, save_fname=None):

        sub_dir = 'files'
        original, intersection, ground_truth, prediction = self.get_run_masks(run_name, patient_id) 
        run_idx = self.get_run_idx(run_name) 
        run = self.runs[run_idx]          
        run_dir = self.get_masks_from_run(run)

        original_path = run_dir / sub_dir / original
        intersection_path = run_dir / sub_dir /  intersection
        ground_truth_path = run_dir / sub_dir /  ground_truth
        prediction_path = run_dir / sub_dir /  prediction

        original_ = plt.imread(original_path)
        intersection_ = plt.imread(ground_truth_path)
        ground_truth_ = plt.imread(ground_truth_path)
        prediction_ = plt.imread(prediction_path)

        class_labels = {
        0: "Negative",
        2: "Intersection",
        3: "Ground truth (only)",
        4: "Predicted (only)"
        }

        _map_masks = lambda x, val: np.rot90(np.ma.masked_where(np.round(x*255) != val, x))

        TPs = (~ _map_masks(intersection_, 2).mask).sum()
        FNs = (~ _map_masks(intersection_, 3).mask).sum()
        FPs = (~ _map_masks(intersection_, 4).mask).sum()
        TNs = original_.shape[0]*original_.shape[1] - TPs - FNs - FPs
        
        #dice = 1- 2*(TPs) / (2*(TPs) + FNs + FPs)
        dice = 1- (2*(TPs)+self.smooth) / (2*(TPs) + FNs + FPs+self.smooth)

        title = f'TPs:{TPs}, FNs:{FNs}, FPs:{FPs}, Loss:{dice:.2f}'

        intersection_ = self.non_linear_map(intersection_)

        colour_list = ['white', 'green', 'orange', 'red']
        cmap = colors.ListedColormap(colour_list)
        bounds=[0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N) 

        alphas = np.ones(np.rot90(intersection_).shape)*transparency
        alphas[np.rot90(intersection_)==0] = 0
                    
        #bounds=[0., 0.00784314, 0.01176471, 0.01568628]
        
        f, ax = plt.subplots(1,1, figsize=(16,10))

        if zoom_inset is None:            

            ax.imshow(np.rot90(original_), cmap="gray")
            ax.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)
            ax.set_title(title, fontsize=16)
            ax.axis('off')

        else:
            ax.imshow(np.rot90(original_), cmap="gray")

            if show_both_masks:
                ax.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)

            #_flip_rot = lambda x: np.flipud(np.rot90(x))

            pop_a = mpatches.Patch(color=colour_list[1], label='ground truth')
            # pop_b = mpatches.Patch(color=colour_list[2], label='false negatives')
            # pop_c = mpatches.Patch(color=colour_list[3], label='false positives')

            plt.legend(handles=[pop_a], loc=4, fontsize=14)

            x1, x2, y1, y2 = self.get_cropped_window_dims(np.rot90(intersection_))
            #x1, x2, y1, y2 = self.get_cropped_window_dims(np.rot90(intersection_))

            axins = zoomed_inset_axes(ax, zoom_inset, loc=9) # zoom-factor: 2.5, location: upper-left
            ax.imshow(np.rot90(original_), cmap="gray")

            if show_both_masks:            
                ax.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)

            axins.imshow(np.rot90(original_), cmap="gray")
            axins.imshow(np.rot90(intersection_), alpha=alphas, cmap=cmap, norm=norm)
            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y2, y1) # apply the y-limits
            plt.yticks(visible=False)
            plt.xticks(visible=False)

            mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="1.0")

            ax.set_title('2D MRI with ground truth labels', fontsize=30)
            ax.axis('off')

        if save_fname is not None:
            f.savefig(save_fname)


    def plot_original_and_binary_mask(self, original_, mask_, 
                                      color_ = 'green',
                                      transparency=0.5, 
                                      zoom_inset=None, 
                                      patient_id_=None,
                                      model_name_=None,
                                      save_fname=None, 
                                      legend_loc=4):
        
        original_ = np.squeeze(original_.cpu().numpy())
        mask_ = np.squeeze(mask_.int().cpu().numpy())
        
        title = f'{model_name_}: patient: {patient_id_}'

        colour_list = ['white', color_]
        cmap = colors.ListedColormap(colour_list)
        bounds=[0, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N) 

        alphas = np.ones(np.rot90(mask_).shape)*transparency
        alphas[np.rot90(mask_)==0] = 0

        #bounds=[0., 0.00784314, 0.01176471, 0.01568628]

        f, ax = plt.subplots(1,1, figsize=(16,10))

        if zoom_inset is None:            

            ax.imshow(np.rot90(original_), cmap="gray")
            ax.imshow(np.rot90(mask_), alpha=alphas, cmap=cmap, norm=norm)
            ax.set_title(title, fontsize=30)
            ax.axis('off')

        else:
            # ax.imshow(np.rot90(original_), cmap="gray")

            # ax.imshow(np.rot90(mask_), alpha=alphas, cmap=cmap, norm=norm)

            #_flip_rot = lambda x: np.flipud(np.rot90(x))

            # pop_a = mpatches.Patch(color=colour_list[1], label='true positives')
            # pop_b = mpatches.Patch(color=colour_list[2], label='false negatives')
            # pop_c = mpatches.Patch(color=colour_list[3], label='false positives')

            # if isinstance(legend_loc, int):
            #     plt.legend(handles=[pop_a,pop_b,pop_c], loc=legend_loc)
            # else:
            #     plt.legend(handles=[pop_a,pop_b,pop_c], bbox_to_anchor=(1, 1), loc='upper left')

            x1, x2, y1, y2 = self.get_cropped_window_dims(np.rot90(mask_))
            #x1, x2, y1, y2 = self.get_cropped_window_dims(np.rot90(combined_))

            axins = zoomed_inset_axes(ax, zoom_inset, loc=9) # zoom-factor: 2.5, location: upper-left
            ax.imshow(np.rot90(original_), cmap="gray")

            #ax.imshow(np.rot90(mask_), alpha=alphas, cmap=cmap, norm=norm)

            axins.imshow(np.rot90(original_), cmap="gray")
            axins.imshow(np.rot90(mask_), alpha=alphas, cmap=cmap, norm=norm)
            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y2, y1) # apply the y-limits
            plt.yticks(visible=False)
            plt.xticks(visible=False)

            mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="1.0")

            ax.set_title(title, fontsize=26)
            ax.axis('off')

    def crop_npy_to_shape(self, input_, crop_shape=(160,160)):
            max_x, max_y = input_.shape
            crop_x, crop_y = int((max_x - crop_shape[0])/2), int((max_y- crop_shape[1])/2)
            crop_y_30 = -int(crop_y*.8)
            extra_x = crop_shape[0] - (max_x - crop_x-crop_x)
            extra_y = crop_shape[1] - (max_y - crop_y+crop_y_30-crop_y-crop_y_30)

            #input_ = input_[crop_x:-crop_x+extra_x, crop_y-crop_y_30:-crop_y-crop_y_30+extra_y]
            input_ = input_[crop_x:crop_x+crop_shape[0], crop_y-crop_y_30:crop_y-crop_y_30+crop_shape[1]]

            return input_

    def plot_results_from_prediction(self, 
                                     original_, 
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
            original_ = self.crop_npy_to_shape(original_)
            prediction_ = self.crop_npy_to_shape(prediction_)
            label_ = self.crop_npy_to_shape(label_)

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
                x1, x2, y1, y2 = self.get_cropped_window_dims(np.rot90(combined_))
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
            
    def show_networks_predictions(self, load_model_config, **kwargs):

        load_model_config['batch_size']=1
        load_model_config['valid_batch_size']=1

        plot_title_prefix = kwargs.pop('plot_title_prefix', None)
        filter_by_patients = kwargs.pop('filter_by_patients', [])
        plot_T2_only = kwargs.pop('plot_T2_only', False)
        verbose = kwargs.pop('verbose', False)

        load_model_config = Bunch(load_model_config)
        
        network = load_model(load_model_config)

        device = load_model_config.device

        if 'include_T1' in load_model_config.keys():
            include_T1 = load_model_config.include_T1
        else:
            include_T1 = False        
        
        _, val_loader = build_datasets(load_model_config, verbose=verbose)

        network.float()
        is_train=False
        use_sigmoid = False

        for _, data in enumerate(val_loader):

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
                if int(patient_id) == 3607 or int(patient_id) == 3828:
                    continue

                if filter_by_patients:
                    # If we are only viewing a list of particular patients, check if the patient is in the list
                    if int(patient_id) not in filter_by_patients:
                        continue

                model_name = load_model_config.model_name

                if plot_title_prefix:
                    title_prefix=f'{plot_title_prefix} ({patient_id}) '
                else:
                    title_prefix=f'{model_name}: patient: {patient_id} '

                if include_T1 and not plot_T2_only:
                    f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,10))

                    inputs_1C = inputs[:,0,:,:].unsqueeze(1)
                    inputs_2C = inputs[:,1,:,:].unsqueeze(1)

                    self.plot_results_from_prediction(original_=inputs_1C, 
                                                    label_=labels, 
                                                    prediction_=predictions,
                                                    ax=ax1,
                                                    title_prefix_=title_prefix,
                                                    model_name_ = model_name,
                                                    patient_id_ = patient_id,
                                                    zoom_inset=2.5,
                                                    legend_loc='upper-left')


                    self.plot_results_from_prediction(original_=inputs_2C, 
                                                    label_=labels, 
                                                    prediction_=predictions,
                                                    ax=ax2,
                                                    title_prefix_=title_prefix,
                                                    model_name_ = model_name,
                                                    patient_id_ = patient_id,
                                                    zoom_inset=2.5,
                                                    legend_loc=None)


                else:

                    f, ax = plt.subplots(1,1, figsize=(16,10))

                    inputs = inputs[:,0,:,:].unsqueeze(1)

                    self.plot_results_from_prediction(original_=inputs, 
                                                    label_=labels, 
                                                    prediction_=predictions,
                                                    ax=ax,
                                                    title_prefix_=title_prefix,
                                                    model_name_ = model_name,
                                                    patient_id_ = patient_id,
                                                    zoom_inset=2.5)
                      

    def show_networks_predictions_v2(self, load_model_config, ax, filter_by_patients, **kwargs):
        """
        This function is essentially the same as show_networks_predictions, except it only finds and plots a single result on ax
        
        """

        load_model_config['batch_size']=1
        load_model_config['valid_batch_size']=1
        plot_T2_only = False
        verbose = kwargs.pop('verbose', False)

        load_model_config = Bunch(load_model_config)
        
        network = load_model(load_model_config)

        device = load_model_config.device

        if 'include_T1' in load_model_config.keys():
            include_T1 = load_model_config.include_T1
        else:
            include_T1 = False        
        
        _, val_loader = build_datasets(load_model_config, verbose=verbose)

        network.float()
        is_train=False
        use_sigmoid = False

        for _, data in enumerate(val_loader):

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

                #print(patient_id)
                if int(patient_id) == 3607 or int(patient_id) == 3828:
                    continue

                if int(patient_id) not in filter_by_patients:
                        continue

                model_name = load_model_config.model_name

                inputs = inputs[:,0,:,:].unsqueeze(1)

                metrics = self.plot_results_from_prediction(original_=inputs, 
                                                            label_=labels, 
                                                            prediction_=predictions,
                                                            ax=ax,
                                                            show_title=False,
                                                            title_prefix_='',
                                                            model_name_ = model_name,
                                                            patient_id_ = patient_id,
                                                            return_metrics=True,
                                                            zoom_inset=2.5)

                return metrics


    def compute_metrics_from_saved_model(self, load_model_config, phase='validation', save_by_patient_id=False, **kwargs):

        load_model_config['batch_size']=1
        load_model_config['valid_batch_size']=1

        plot_title_prefix = kwargs.pop('plot_title_prefix', None)
        filter_by_patients = kwargs.pop('filter_by_patients', [])

        #load_model_config = Bunch(load_model_config)
        
        network = load_model(load_model_config)

        device = load_model_config.device

        if 'include_T1' in load_model_config.keys():
            include_T1 = load_model_config.include_T1
        else:
            include_T1 = False        
        
        if phase == 'validation':
            _, loader = build_datasets(load_model_config, verbose=False)
        elif phase == 'train':
            loader, _ = build_datasets(load_model_config, verbose=False)
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
        #use_sigmoid = False
        use_sigmoid = get_use_sigmoid(load_model_config)
        #     if 'network_name' in load_model_config.keys():
        #         if load_model_config.network_name.startswith('smp.'):
        #             use_sigmoid = True
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
                rm.add_to_results_df(model_name = f'{load_model_config.model_name}', 
                                    patient_id=patient_id,
                                    phase=phase,
                                    dsc=dsc,
                                    tp=tp,
                                    fp=fp,
                                    tn=tn,
                                    fn=fn)
            
        return dsc, tp_sum, fp_sum, tn_sum, fn_sum  
