import argparse
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import json
import torch
import random
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn

from models.Mixer import mixer
from attack_methods import BTG
from dataset.ucf101 import get_dataset
from gluoncv.torch.model_zoo import get_model

from utils import CONFIG_PATHS, OPT_PATH, get_cfg_custom, MODEL_TO_CKPTS, AverageMeter

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Set random seed for reproducibility
def set_random_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Argument parser
def arg_parse():
    parser = argparse.ArgumentParser(description='Attack and Evaluate Models')
    parser.add_argument('--gpu', type=str, default='1', help='GPU device ID.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--model', type=str, default='tpn_resnet101', help='Surrogate model name.')
    parser.add_argument('--epsilon', type=int, default=16, help='Attack epsilon.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--attack_method', type=str, default='BTG', help='Attack method to use.')
    parser.add_argument('--dataset_csv', type=str, default='./data/ucf_all_info.csv', help='Dataset CSV file path.')
    parser.add_argument('--used_idxs_pkl', type=str, default='./data/used_idxs.pkl', help='Used indices pickle file path.')
    parser.add_argument('--adv_save_path', type=str, default='', help='Path to save adversarial examples.')
    parser.add_argument('--datasets_root', type = str, default= '')
    parser.add_argument('--name', type = str, default= 'test')
    parser.add_argument('--mixer_ckpt', type = str)
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = arg_parse()
    print (args)
        # Set random seed
    set_random_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # Create directory to save adversarial examples
    adv_save_path = os.path.join(args.adv_save_path, args.name)
    if not os.path.exists(adv_save_path):
        os.makedirs(adv_save_path)

    # loading cfg
    cfg_path = CONFIG_PATHS[args.model]
    
    cfg = get_cfg_custom(cfg_path, args.batch_size)
    cfg.CONFIG.MODEL.PRETRAINED = False

    ckpt_path = MODEL_TO_CKPTS[args.model]
    if args.mixer_ckpt:
        ckpt_path = args.mixer_ckpt
    surrogate_model = get_model(cfg)
    surrogate_ckpt = torch.load(ckpt_path)
    if args.mixer_ckpt:
        num_classes = 202
        surrogate_model.fc = nn.Linear(surrogate_model.fc.in_features, num_classes)
    surrogate_model.load_state_dict(surrogate_ckpt['state_dict'])
    surrogate_model.to(device)
    surrogate_model.eval()

    # loading dataset
    dataset_loader = get_dataset('./ucf_all_info.csv', './used_idxs.pkl', args.datasets_root, args.batch_size)
    attack_methods = {
        'BTG': BTG
    }
    
    std = [0.229, 0.224, 0.225]
    std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(device)
    epsilon_normalized = args.epsilon / 255.0 / std_tensor
    attack_params = {
        'epsilon': epsilon_normalized,
        'steps': 10,
        'delay': 1.0,
        'lamb': 0.8,
        'gamma': 0.1,
    }
    attack_class = attack_methods[args.attack_method]
    attack = attack_class(surrogate_model, attack_params)

    for step, data in enumerate(dataset_loader):
        if step %1 == 0:
            print ('Running {}, {}/{}'.format(args.attack_method, step+1, len(dataset_loader)))
        val_batch = data[0].cuda()
        val_label = data[1].cuda()
        adv_batches = attack(val_batch, val_label)
        val_batch = val_batch.detach()
        for ind,label in enumerate(val_label):
            ori = val_batch[ind].cpu().numpy()
            adv = adv_batches[ind].cpu().numpy()
            np.save(os.path.join(adv_save_path, '{}-adv'.format(label.item())), adv)
            np.save(os.path.join(adv_save_path, '{}-ori'.format(label.item())), ori)
