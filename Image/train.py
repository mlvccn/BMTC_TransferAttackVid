import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import transferattack
from transferattack.utils import *
from models.Mixer import mixer
import random


def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='mifgsm', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=100, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=16, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--background_sets', default="", type=str, help='the path to store the adversarial patches')
    parser.add_argument('--save_path', default="", type=str)
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='0', type=str)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--mixer_surrogate', type=str, default= 'resnet101')
    parser.add_argument('--mixer', type=str, default= 'resnet101')
    return parser.parse_args()

args = get_parser()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

output_dir = os.path.join(args.output_dir, args.attack, args.mixer)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
device = torch.device("cuda:{}".format(args.GPU_ID) if torch.cuda.is_available() else "cpu")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset = AdvDataset(input_dir=args.input_dir, output_dir=output_dir, targeted=args.targeted, eval=args.eval)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

## mixer surrogate model load
surrogate_model = models.__dict__[args.mixer_surrogate](pretrained = True)
surrogate_model.eval().to(device)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

std = torch.tensor(std).view(1, 3, 1, 1).to(device)
mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)

eps = (args.eps - mean)/std.view(1,3,1,1)
normalize = transforms.Normalize(mean, std)
torch.nn.Sequential(normalize, surrogate_model)
target_name = ['resnet18', 'resnet101', 'resnext50_32x4d', 'densenet121']

mix_model = mixer(eps, 1000, surrogate_name=args.mixer_surrogate, target_name = target_name, gamma=0.8, w1=1, w2=0.3, w3=0.1, device = device)
mix_model.train().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(mix_model.parameters(), lr=args.lr, momentum=0.9)
    
def pil_loader(path):
    # Open image and convert to RGB
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def transform_image(image):
    # Define the transform pipeline (resize and normalize)
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]

    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=default_mean, std=default_std)  # Normalize
    ])

    return transform_pipeline(image)

def process_background_sets(directory):
    tensors = []

    files = sorted([f for f in os.listdir(directory) if f.endswith('.JPEG')])
    
    for file_name in files:

        file_path = os.path.join(directory, file_name)

        image = pil_loader(file_path)
        transformed_tensor = transform_image(image)

        tensors.append(transformed_tensor)

    final_tensor = torch.stack(tensors, dim=0)

    return final_tensor

background_images = process_background_sets(args.background_sets)
background_images = background_images.to(device)
num_epochs = args.epoch

for epoch in range(num_epochs):
    mix_model.train()
    total_loss = 0.0
    for index, (x, y_true, filenames) in enumerate(dataloader):
        x = x.to(device)
        y_true = y_true.to(device)
        loss = mix_model(x, y_true, background_images, index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    save_path = os.path.join(args.save_path, f'mixer_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'state_dict': mix_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f'Model saved to {save_path}')
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
