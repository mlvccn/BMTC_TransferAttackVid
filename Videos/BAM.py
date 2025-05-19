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

def set_random_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def arg_parse():
    parser = argparse.ArgumentParser(description='Attack and Evaluate Models')
    parser.add_argument('--gpu', type=str, default='1', help='GPU device ID.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--model', type=str, default='tpn_resnet101', help='Surrogate model name.')
    parser.add_argument('--mixer_path', type=str, default="/", help='Path to mixer checkpoint.')
    parser.add_argument('--background_sets', type=str, default='', help='Background images directory.')
    parser.add_argument('--epsilon', type=int, default=16, help='Attack epsilon.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--attack_method', type=str, default='BTG', help='Attack method to use.')
    parser.add_argument('--dataset_csv', type=str, default='', help='Dataset CSV file path.')
    parser.add_argument('--used_idxs_pkl', type=str, default='./data/used_idxs.pkl', help='Used indices pickle file path.')
    parser.add_argument('--mix_save_path', type=str, default='', help='Path to save adversarial examples.')
    args = parser.parse_args()
    
    return args

def process_background_sets(directory):
    tensors = []
    subdirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.jpg')])
        if not files:
            continue
        random_file = random.choice(files)
        random_file_path = os.path.join(subdir_path, random_file)
        image = Image.open(random_file_path).convert('RGB')
        transformed_tensor = transform_image(image)
        tensors.append(transformed_tensor)
    final_tensor = torch.stack(tensors, dim=0)
    return final_tensor

def transform_image(image):
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=default_mean, std=default_std)
    ])
    return transform_pipeline(image)

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))
    correct_k = correct[:1].view(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size), pred.squeeze()


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_random_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    mix_save_path = os.path.join(args.mix_save_path, current_time)
    if not os.path.exists(mix_save_path):
        os.makedirs(mix_save_path)

    num_classes = 101
    std = [0.229, 0.224, 0.225]
    std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(device)
    epsilon_normalized = args.epsilon / 255.0 / std_tensor

    mixer_model = mixer(epsilon_normalized, num_classes, surrogate_model = None, target_models=None, gamma=0.8, w1=1, w2=0.2, w3=0.1)
    mixer_ckpt = torch.load(args.mixer_path)
    mixer_model.load_state_dict(mixer_ckpt['state_dict'], strict=False)
    mixer_model.to(device)
    mixer_model.eval()
    background_images = process_background_sets(args.background_sets)
    background_images = background_images.to(device)

    dataset_loader = get_dataset('./ucf_all_info.csv', './used_idxs.pkl', args.batch_size)
    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]]
    )
    to_pil = transforms.ToPILImage()
    
    for step, data in enumerate(dataset_loader):
        print(f'Running {args.attack_method}, {step+1}/{len(dataset_loader)}')
        val_batch = data[0].to(device)
        val_label = data[1].to(device)
        dic = data[2]
        
        loss, mixed_batch = mixer_model(val_batch, val_label, background_images)
        batch_size = mixed_batch.size(0)
        num_frames = mixed_batch.size(2)

        for b in range(batch_size):
            save_path = os.path.join(mix_save_path, dic[0].split('/')[-1])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for n in range(num_frames):
                frame_tensor = mixed_batch[b, :, n, :, :]  # 
                untransformed_frame = unnormalize(frame_tensor.cpu())
                untransformed_frame = torch.clamp(untransformed_frame, 0, 1)
                image = to_pil(untransformed_frame)
                image_path = os.path.join(save_path, f"image_{n:05d}.jpg")
                image.save(image_path)
                
            