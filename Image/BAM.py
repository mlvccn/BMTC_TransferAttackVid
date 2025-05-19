import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import transferattack
from transferattack.utils import *
from models.Mixer import mixer
import random
import thop

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='mifgsm', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=100, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=16, type=int, help='the bacth size')
    parser.add_argument('--eps', default=8 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model')
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
    parser.add_argument('--mixer_path', type=str, default="", help='Path to save adversarial examples.')
    parser.add_argument('--mix_save_path', type=str, default="", help='Path to save adversarial examples.')
    return parser.parse_args()

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
    # Sort subdirectories by name

    files = sorted([f for f in os.listdir(directory) if f.endswith('.JPEG')])
    
    for file_name in files:

        file_path = os.path.join(directory, file_name)

        # Load and transform the image
        image = pil_loader(file_path)
        transformed_tensor = transform_image(image)

        # Append the tensor (reshape to match desired output format if needed)
        tensors.append(transformed_tensor)

    # Stack all tensors along a new dimension to create (N, C, H, W) tensor
    final_tensor = torch.stack(tensors, dim=0)

    return final_tensor

if __name__ == '__main__':
    args = get_parser()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    device = torch.device("cuda:{}".format(args.GPU_ID) if torch.cuda.is_available() else "cpu")

    # Create directory to save adversarial examples
    mix_save_path = os.path.join(args.mix_save_path, current_time)
    if not os.path.exists(mix_save_path):
        os.makedirs(mix_save_path)

    # Load mixer model
    num_classes = 1000  # Update based on your dataset
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    #mean_tensor = torch.tensor(mean).view(1,3,1,1).to(device)
    #std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(device)
    epsilon_normalized = args.eps #-mean_tensor / std_tensor

    mixer_model = mixer(epsilon_normalized, num_classes, surrogate_name = args.model, target_name=None, gamma=0.8, w1=1, w2=0.2, w3=0.1, device=device)
    mixer_ckpt = torch.load(args.mixer_path)
    mixer_model.load_state_dict(mixer_ckpt['state_dict'], strict=False)
    mixer_model.to(device)
    mixer_model.eval()
    
    

    # Process background images
    background_images = process_background_sets(args.background_sets)
    background_images = background_images.to(device)

    # Load dataset


    dataset = AdvDataset(input_dir=args.input_dir, output_dir=None, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)
    # Define unnormalize transform
    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]]
    )
    to_pil = transforms.ToPILImage()
    """
    background_save_path = os.path.join(args.mix_save_path, current_time)
    if not os.path.exists(background_save_path):
        os.makedirs(background_save_path)

    for idx in range(background_images.size(0)):
        bg_tensor = background_images[idx]  # shape (C, H, W)
        # Unnormalize
        untransformed_bg = unnormalize(bg_tensor.cpu())
        # Clip values to [0,1]
        untransformed_bg = torch.clamp(untransformed_bg, 0, 1)
        # Convert to PIL image
        image = to_pil(untransformed_bg)
        # Save image
        image_path = os.path.join(background_save_path, f"background_{idx}.jpg")
        image.save(image_path)
    """
    
    # Perform the attack and save adversarial examples
    for index, (x, y_true, filenames) in enumerate(dataloader):
        x = x.to(device) #b,c,h,w
        y_true = y_true.to(device)
        # 前向传播并计算损失
        loss, mixed_batch = mixer_model(x, y_true, background_images, index) #b,c,h,w

        batch_size = mixed_batch.size(0)

        for b in range(batch_size):
            save_path = os.path.join(mix_save_path, filenames[b])
            mixed_img = mixed_batch[b,:,:,:]
            #untransformed_img = unnormalize(mixed_img.cpu())
            # Clip values to [0,1]
            untransformed_img = mixed_img.cpu()
            untransformed_img = torch.clamp(untransformed_img, 0, 1)
            # Convert to PIL image
            image = to_pil(untransformed_img)
            # Save image
            image_path = os.path.join(save_path)
            image.save(image_path)
                
            