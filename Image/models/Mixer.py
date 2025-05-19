import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet import resnet18, resnet50, resnet101
import logging
from transferattack.utils import *
# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def adversarial_fusion(x, B, gamma):

    return gamma * x + (1 - gamma) * B

class mixer(nn.Module):
    def __init__(self, epsilon, action_dim, surrogate_name, target_name, gamma, w1, w2, w3, device):
        super(mixer, self).__init__()
        resnet = resnet50()
        self.epsilon = epsilon
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, action_dim)
        self.surrogate_name = surrogate_name
        self.target_name = target_name
        self.gamma_value = gamma
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.device = device
        

    def forward(self, x, y_true, background_images, idx):

        batch_size, channels, height, width = x.shape
        x_frames = x
        features = self.feature_extractor(x_frames)  
        features = features.view(features.size(0), -1)  
        action_logits = self.fc(features) 
        action_probs = F.softmax(action_logits, dim=-1) 
        m = torch.distributions.Categorical(action_probs)
        actions = m.sample() 
        log_probs = m.log_prob(actions)  
        B_frames = background_images[actions]

        x_adv_frames = adversarial_fusion(x_frames, B_frames, self.gamma_value)
        
        x_adv_frames = torch.clamp(x_adv_frames, x_frames - self.epsilon, x_frames + self.epsilon)

        x_adv = x_adv_frames.view(batch_size, channels, height, width).permute(0, 1, 2, 3)
        R_attack = self.compute_attack_reward(x, y_true, x_adv)
        
        if idx % 5 == 0 and self.target_name:
            R_transfer = self.compute_transfer_reward(x_adv, y_true)
        else:
            R_transfer = 0.0
        if isinstance(R_attack, torch.Tensor):
            R_attack = R_attack.item()

        R_total = self.w1 * R_attack + self.w2 * R_transfer 
        
        if isinstance(R_total, torch.Tensor):
            R_total = R_total.item()
        total_log_prob = log_probs.sum()
        loss = -total_log_prob * R_total
        return loss, x_adv

    def compute_attack_reward(self, x, y_true, x_adv):

        surrogate_model = models.__dict__[self.surrogate_name](pretrained = True)
        surrogate_model.eval().to(self.device)
        with torch.no_grad():
            output_orig = surrogate_model(x)
            output_adv = surrogate_model(x_adv)
            
            output_orig = output_orig[:, :1000]
            output_adv = output_adv[:, :1000]

            z_y = output_orig[torch.arange(output_orig.size(0)), y_true]
            z_hat_y = output_adv[torch.arange(output_adv.size(0)), y_true]

            R_attack = z_hat_y - z_y
            
        return R_attack.mean().item()


    def compute_transfer_reward(self, x_adv, y_true):
        R_attack_list = []
        batch_size = x_adv.size(0)
        for model_name, model in load_pretrained_model(self.target_name):
            model = wrap_model(model.eval().to(self.device))
            with torch.no_grad():
                output = model(x_adv)
                output = output[:, :1000]
                z_y = output[torch.arange(output.size(0)), y_true]
                mask = torch.ones_like(output, dtype=torch.bool)
                mask[torch.arange(output.size(0)), y_true] = False
                output_masked = output.masked_fill(~mask, float('-inf'))
                z_hat_y, _ = output_masked.max(1)
                R_attack_i_per_sample = z_hat_y - z_y
                R_attack_list.append(R_attack_i_per_sample)
                logger.info(f"Model: {model_name}, R_attack_i: {R_attack_i_per_sample}")
                del model
        R_transfer = (sum(R_attack_list) / len(self.target_name)).mean().item()
        logger.info(f"Total R_transfer: {R_transfer}")
        return R_transfer


    def compute_temporal_consistency_reward(self, x_adv, y_true):
        surrogate_model = models.__dict__[self.surrogate_name](pretrained = True)
        surrogate_model.eval().to(self.device)
        batch_size, channels, frames, height, width = x_adv.shape
        R_consis = 0.0
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        for t in range(frames - 1):
            x_adv.requires_grad_(True)
            output = surrogate_model(x_adv)
            loss = F.cross_entropy(output, y_true)
            surrogate_model.zero_grad()
            loss.backward(retain_graph=True)
            grad = x_adv.grad
            grad_t = grad[:, :, t, :, :].view(batch_size, -1) 
            grad_t1 = grad[:, :, t+1, :, :].view(batch_size, -1)
            sim = cosine_sim(grad_t, grad_t1) 
            R_consis += (1 - sim.mean().item()) 
            x_adv.grad.zero_()
        R_consis /= (frames - 1)
        return R_consis

