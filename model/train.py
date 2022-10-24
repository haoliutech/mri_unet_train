"""
This file comes from build_v4.py.
The purpose of this is to refactor that function and separate out networks, loss functions etc.

"""

import os

import numpy as np
import torch
import torch.optim as optim
import torchio as tio
from tqdm import tqdm

from model.build_networks import *
from model.build_loss_criterion import *
from model.tio_data import *

def get_use_sigmoid(config):
    # Some networks need to use sigmoid after passing through the network

    use_sigmoid = False
    if 'network_name' in config.keys():
        if config['network_name'].startswith('smp.'):
            use_sigmoid = True
            
    return use_sigmoid

def build_datasets(config, verbose=True):
    tio_data = TioData(config, verbose=verbose)
    loader_train, loader_valid = tio_data.tio_data_loaders(config)
    
    return loader_train, loader_valid
        
def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def train(config):
    # Initialize a new wandb run
    
    log_images = False
    best_loss = 10e7
    epochs_since_best = 0
    
    es_compare_loss = 10e6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Training on device:{device}')

    #print(config.keys())
    loader_train, loader_valid = build_datasets(config)

    print(len(loader_train))


    network = build_network(config)
    optimizer = build_optimizer(network, config['optimizer'], config['learning_rate'])

    # To compare results using different loss functions, we will log metrics on the same base metric
    criterion = build_criterion(config)

    # Some networks need to apply sigmoid after network
    use_sigmoid = get_use_sigmoid(config)

    if 'include_T1' in config.keys():
        include_T1 = config['include_T1']
    else:
        include_T1 = False

    for epoch in tqdm(range(config['epochs'])):
        avg_loss_train = train_epoch(network, loader_train, criterion, optimizer, device, is_train=True, use_sigmoid=use_sigmoid, include_T1=include_T1)
        
        #log_images = log_predicted_masks(config, epoch)
        avg_loss_val = train_epoch(network, loader_valid, criterion, optimizer, device, is_train=False, log_images=log_images, use_sigmoid=use_sigmoid, include_T1=include_T1)

        if config['log_dce_loss'] is None:
            es_compare_loss=avg_loss_val
        
        if (epoch % config['log_dce_loss'] == 0):

            avg_dice_score_train, tp_train, fp_train, tn_train, fn_train = get_metrics(network, loader_train, device, include_T1, use_sigmoid)
            avg_dice_score_val, tp_val, fp_val, tn_val, fn_val = get_metrics(network, loader_valid, device, include_T1, use_sigmoid)
                
            es_compare_loss = 1-avg_dice_score_val 
            
        else:              
            pass  

        #best_loss, epochs_since_best = save_model(network, config, epoch, es_compare_loss, best_loss, epochs_since_best)

        best_loss, epochs_since_best = save_model(network, config, epoch, avg_loss_val, best_loss, epochs_since_best)

        print(f"avg_loss_train: {avg_loss_train}")
        print(f"avg_loss_val: {avg_loss_val}")
        if epochs_since_best == 0:
            break
            
        

def train_epoch(network, loader, criterion, optimizer, device, is_train=True, log_images=False, use_sigmoid=False, include_T1=False):

    cumu_loss = 0
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
        
        if is_train:
            optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(is_train):
            if use_sigmoid:
                predictions = torch.sigmoid(network(inputs))
            else:
                predictions = network(inputs)
                
            #predictions = network(inputs)
            #predictions = network.forward(inputs)
            loss = criterion(predictions, labels)
            cumu_loss += loss.item()

            if is_train:
                # Backward pass + weight update
                loss.backward()
                optimizer.step()

                        
    return cumu_loss / len(loader)



def load_model(config):
    
    new_model = build_network(config)
    new_model.load_state_dict(torch.load(os.path.join(config['model_dir'], config['model_name'])))
    new_model.eval()
    
    return new_model

def save_model(model, config, epoch, epoch_loss, best_loss, epochs_since_best):
    
    # CALLBACK: callback_model_save_epoch_freq_after_valid 
    # Design choice to log after validation and after X many epochs
    # Also make this callback on the final epoch 
    # Essentially the same as callback_log_images_freq, but can be set differently if required

    # Decrement epochs since best counter
    cwd_path = directory = os.getcwd()
    save_path = os.path.join(cwd_path,"data/save_model//trained_unet.pt")
    
    epochs_since_best+= -1
    
    if (epoch % config['callback_log_model_freq'] == 0) or (epoch == config['epochs'] - 1):
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("best loss: "+str(best_loss))
            # torch.save(model.state_dict(), save_path)
            epochs_since_best = config['early_stopping']
            
            if best_loss < config['min_val_loss']:
                if config['verbose']: print(f'Saving model with best mean dice Loss: {epoch_loss:4f}')
                torch.save(model.state_dict(), save_path)
                print(f'Saving model with best mean dice Loss: {epoch_loss:4f}')

    return best_loss, epochs_since_best    

def log_predicted_masks(config, epoch):
    # CALLBACK: callback_log_images_freq 
    # Design choice to log after validation and after X many epochs
    # Also make this callback on the final epoch
    if (epoch % config['callback_log_images_freq'] == 0) or (epoch == config['epochs'] - 1):
        return True
    else:
        return False



def get_metrics(network, loader, device, include_T1, use_sigmoid):

    log_metrics = LogMetrics()

    cumu_loss=0
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
        with torch.set_grad_enabled(False):
            if use_sigmoid:
                predictions = torch.sigmoid(network(inputs))
            else:
                predictions = network(inputs)
                            
            loss, (tp, fp, tn, fn) = log_metrics(predictions, labels)

            cumu_loss += loss.item()
            tp_sum+=tp
            fp_sum+=fp
            tn_sum+=tn
            fn_sum+=fn
        
            
    return cumu_loss / len(loader), tp_sum/ len(loader), fp_sum/ len(loader), tn_sum/ len(loader), fn_sum/ len(loader)

