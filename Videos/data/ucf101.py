import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms


class ucf101_loader(data.Dataset):
    def __init__(self, root, img_size, data_augmentation=False, background_dir = 'backgrounds_batch_1', batch_size = 48):
        self.img_size = img_size
        self.images, self.backgrounds, self.gts = [], [], []
        
        self.video_frames = []
     
        seq_dir = os.path.join(root,'frames')
        print("seq_dir:", seq_dir) 
        for seq_name in os.listdir(seq_dir):
            print("Processing:", seq_name)
            seq_path = os.path.join(seq_dir, seq_name)
            for video in os.listdir(seq_path):
                video_path = os.path.join(seq_path,video)
                frames = sorted([os.path.join(video_path, frame) for frame in os.listdir(video_path) if frame.endswith('.jpg') or frame.endswith('.png')])
                self.images += [frames]
            if not frames:
                print(f"No image files found in {seq_path}. Skipping...")

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.backgrounds = self.images
        
        for i in range(len(self.backgrounds)):
            for ii in range(len(self.backgrounds[i])):
                self.backgrounds[i][ii].replace('frames', background_dir)
            if not os.path.exists(self.backgrounds[i][-1]):
                self.backgrounds[i].pop()
                
        self.video_frames = self.images + self.backgrounds
                    
        self.size = sum(len(video_frames) for video_frames in self.video_frames)
        
    def __getitem__(self, idx):
        video_index = 0
        frame_within_video_idx = idx
        for frames in self.images:
            if frame_within_video_idx < len(frames):
                break
            frame_within_video_idx -= len(frames)
            video_index += 1
        img_path = self.images[video_index][frame_within_video_idx]
        
        last_img = self.images[video_index][-1]
        if img_path == last_img:
            flow_path = self.zero_flow
        else:
            flow_path = img_path.replace('frames', 'flows').replace('image_0','flow_').replace('jpg','png')
            if not os.path.exists(flow_path):
                flow_path = flow_path.replace('jpg', 'png')
            if not os.path.exists(flow_path):
                flow_path = flow_path.replace('png', 'jpg')
        image = self.img_transform(self.rgb_loader(img_path))
        
        flow = self.img_transform(self.rgb_loader(flow_path))

        return image, flow, img_path

    
    def __len__(self):
        return self.size
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')