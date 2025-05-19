import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
import numpy as np
import torchvision
from PIL import Image
import random
import time
import math
#from GradCAM import GradCAM
import cv2

#from data_viz import test_transform_reverse
from collections import Counter
from datetime import datetime
import os
from utils import  CONFIG_PATHS, get_cfg_custom, AverageMeter, OPT_PATH, MODEL_TO_CKPTS
from gluoncv.torch.model_zoo import get_model
import pandas as pd
import json
import logging

ucf_mean = [0.485, 0.456, 0.406]
ucf_std = [0.229, 0.224, 0.225]

def norm_grads(grads, frame_level=True):
    # frame level norm
    # clip level norm
    assert len(grads.shape) == 5 and grads.shape[2] == 32
    if frame_level:
        norm = torch.mean(torch.abs(grads), [1,3,4], keepdim=True)
    else:
        norm = torch.mean(torch.abs(grads), [1,2,3,4], keepdim=True)
    # norm = torch.norm(grads, dim=[1,2,3,4], p=1)
    return grads / norm

class Attack(object):
    """
    # refer to https://github.com/Harry24k/adversarial-attacks-pytorch
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the model's training mode to `test`
        by `.eval()` only during an attack process.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device
        
        self._targeted = 1
        self._attack_mode = 'default'
        self._return_type = 'float'
        self._target_map_function = lambda images, labels:labels

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, *input):
        r"""
        It defines the computation performed at every call (attack forward).
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
    def set_attack_mode(self, mode, target_map_function=None):
        r"""
        Set the attack mode.
  
        Arguments:
            mode (str) : 'default' (DEFAULT)
                         'targeted' - Use input labels as targeted labels.
                         'least_likely' - Use least likely labels as targeted labels.
                         
            target_map_function (function) :
        """
        if self._attack_mode == 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        if (mode == 'targeted') and (target_map_function == None):
            raise ValueError("Please give a target_map_function, e.g., lambda images, labels:(labels+1)%10.")
            
        if mode=="default":
            self._attack_mode = "default"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode=="targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._target_map_function = target_map_function
            self._transform_label = self._get_target_label
        elif mode=="least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(mode + " is not a valid mode. [Options : default, targeted, least_likely]")
            
    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str) : 'float' or 'int'. (DEFAULT : 'float')
        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str) : save_path.
            data_loader (torch.utils.data.DataLoader) : data loader.
            verbose (bool) : True for displaying detailed information. (DEFAULT : True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print('- Save Progress : %2.2f %% / Accuracy : %2.2f %%' % ((step+1)/total_batch*100, acc), end='\r')

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print('\n- Save Complete!')

        self._switch_model()
    
    def _transform_video(self, video, mode='forward'):
        r'''
        Transform the video into [0, 1]
        '''
        dtype = video.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=self.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=self.device)
        if mode == 'forward':
            # [-mean/std, mean/std]
            video.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        elif mode == 'back':
            # [0, 1]
            video.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
        return video

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels
        
    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels
    
    def _get_target_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return self._target_map_function(images, labels)
    
    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels
    
    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()
        
        del_keys = ['model', 'attack']
        
        for key in info.keys():
            if key[0] == "_" :
                del_keys.append(key)
                
        for key in del_keys:
            del info[key]
        
        info['attack_mode'] = self._attack_mode
        if info['attack_mode'] == 'only_default' :
            info['attack_mode'] = 'default'
            
        info['return_type'] = self._return_type
        
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images
    
model_names = ['i3d_resnet101', 'i3d_resnet50', 'slowfast_resnet101', 'slowfast_resnet50', 'tpn_resnet101', 'tpn_resnet50']

class BTG(Attack):
    def __init__(self, model, attack_params):
        super(BTG, self).__init__("BTG", model)
        
        self.epsilon = attack_params['epsilon'].view(1, 3, 1, 1, 1)
        self.steps = attack_params['steps']
        self.step_size = self.epsilon / self.steps
        self.delay = attack_params['delay']
        self.lamb = attack_params['lamb']
        self.gamma = attack_params['gamma']
        
        self.frames = 32
        self.device = next(model.parameters()).device
        self.loss_fn = nn.CrossEntropyLoss()
        
        
    def forward(self, videos, labels):
        """
        Overridden.
        """
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') 
        adv_videos = videos.float().detach()
        batch_size, channels, frames, height, width = adv_videos.shape
        
        for step in range(self.steps):
            grad_directions = []
            
            cls_loss = self.model(adv_videos)
            
            x_adv = adv_videos.clone().requires_grad_(True)
            x_adv_expanded = x_adv.unsqueeze(2)
            if (self.model.__class__.__name__ == 'I3D_ResNetV1'):
                x_adv_expanded.repeat(x_adv_expanded.shape[0], x_adv_expanded.shape[1], 16, x_adv_expanded.shape[3], x_adv_expanded.shape[4])
            output_t = self.model(x_adv_expanded)
            loss_t = self.loss_fn(output_t, labels)
            self.model.zero_grad()
            loss_t.backward()
                
            for t in range(frames):
                x_t_adv = x_adv[:, :, t, :, :]
                grad = x_t_adv.grad.data.clone()
                grad_direction = grad.view(batch_size, -1)
                grad_direction = grad_direction / (grad_direction.norm(p=2, dim=1, keepdim=True) + 1e-8)
                grad_directions.append(grad_direction)
                x_t_adv.grad.zero_()

            cos_sim_loss = 0.0
            for t in range(frames - 1):
                cos_sim = F.cosine_similarity(grad_directions[t], grad_directions[t + 1], dim=1)
                cos_sim_loss += (1 - cos_sim.mean())
            
            adv_videos.requires_grad = True
            outputs = self.model(adv_videos)
            loss_total = self.loss_fn(outputs, labels)
            loss_total += self.gamma * cos_sim_loss 
            self.model.zero_grad()
            loss_total.backward()
            grad = adv_videos.grad.data.clone()

            adv_videos = adv_videos + self.step_size.view(adv_videos.shape[0], adv_videos.shape[1], 1, 1, 1) * grad.sign()

            
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
        
        
        return adv_videos