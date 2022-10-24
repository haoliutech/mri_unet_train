import os

import numpy as np
import torch.optim as optim

import segmentation_models_pytorch as smp

from networks.unet import UNet
from networks.UNet_3Plus import *
from networks.UNet_2Plus import *

def build_network(config):
    device = config['device']

    if 'include_T1' in config.keys():
        in_channels = 1+int(config['include_T1'])
    else:
        in_channels = 1

    out_channels = 1
    
    SegmentationModels = {'UNet' : UNet,
                          'UNet_3Plus' : UNet_3Plus,
                          'UNet_2Plus' : UNet_2Plus,
                          'smp.Unet' : smp.Unet,
                          'smp.UnetPlusPlus' : smp.UnetPlusPlus,
                          'smp.PSPNet' : smp.PSPNet,
                          'smp.Linknet' : smp.Linknet,
                          'smp.MAnet' : smp.MAnet,
                          'smp.FPN' : smp.FPN,
                          'smp.PAN' : smp.PAN,
                         }

    if 'network_name' in config.keys():
        network_name = config['network_name']
    else:
        network_name = 'network_name'

    if network_name not in SegmentationModels.keys():
        raise ValueError(f'Invalid network_name:{network_name}')
            
    kwargs = {'in_channels': in_channels}

    if network_name == 'UNet':
        kwargs['out_channels'] = out_channels
        
    if network_name.startswith('smp.'):
        kwargs['classes'] = 1

        if 'encoder_name' in config.keys():
            kwargs['encoder_name'] = config['encoder_name']
    
        if 'encoder_weights' in config.keys():
            kwargs['encoder_weights'] = config['encoder_weights']

    network = SegmentationModels[network_name](**kwargs)
    
    return network.to(device)