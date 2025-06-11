import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Add this import
import lpips
# from trainer import LD3Trainer, ModelConfig, TrainingConfig, DiscretizeModelWrapper
from utils import get_solvers, move_tensor_to_device, parse_arguments, set_seed_everything

from dataset import load_data_from_dir, LTTDataset

# Fully connected neural network with one hidden layer
class LTT_model(nn.Module):
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0):
        super().__init__()

        self.unet = SimpleUNet_Encoding(
            in_channels=3
        )
        self.mlp = SimpleMLP(
            input_size=256,
            output_size=steps + 1,
            hidden_size=512,
            dropout=mlp_dropout
        )

        self.l1_norm = SimpleMLP.L1NormLayer()
    
    def forward(self, x):
        out = self.unet(x)
        out = self.mlp(out)
        out = F.softplus(out)
        # x = torch.sigmoid(x) 
        out = self.l1_norm(out)
        # x = torch.softmax(x, dim=1)  # Apply softmax along the class dimension

        return out

#https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114
class SimpleUNet_Encoding(torch.nn.Module):
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, dropout=0.0,num_groups=32):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            # self.bn1 = nn.BatchNorm2d(out_channels)
            self.gn1 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
            self.dropout1 = nn.Dropout2d(dropout)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            # self.bn2 = nn.BatchNorm2d(out_channels)
            self.gn2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
            self.dropout2 = nn.Dropout2d(dropout)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        def forward(self, x):
            residual = self.shortcut(x)
            # x = F.leaky_relu(self.conv1(x), 0.1) #F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
            x = F.leaky_relu(self.gn1(self.conv1(x)), 0.1)
            x = self.dropout1(x)
            # x = self.conv2(x) #self.bn2(self.conv2(x))
            x = self.gn2(self.conv2(x))
            x = self.dropout2(x)
            return F.leaky_relu(x + residual, 0.1)
    class DownSample(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = SimpleUNet_Encoding.DoubleConv(in_channels, out_channels)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            down = self.conv(x)
            p = self.pool(down)

            return down, p

    def __init__(self, in_channels):
        super().__init__()
        self.down_convolution_1 = SimpleUNet_Encoding.DownSample(in_channels, 64)
        self.down_convolution_2 = SimpleUNet_Encoding.DownSample(64, 128)
        self.bottle_neck = SimpleUNet_Encoding.DoubleConv(128, 256)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)

        b = self.bottle_neck(p2)
        b = F.adaptive_avg_pool2d(b, (1, 1)).squeeze(-1).squeeze(-1)
        return b


class AdaUNet_Encoding(torch.nn.Module):

    class AdaNorm(nn.Module):
        def __init__(self, num_channels, emb_dim, num_groups=32):
            super().__init__()
            self.norm = nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)
            self.fc = nn.Sequential(
                nn.Linear(emb_dim, num_channels * 2),
                nn.SiLU()
            )

        def forward(self, x, t_emb):
            out = self.norm(x)
            gamma_beta = self.fc(t_emb)  # shape: [B, 2 * C]
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
            return out * gamma + beta
        
    class TimestepEmbedding(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.embedding = nn.Sequential(
                nn.Linear(1, emb_dim,),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim)
            )

        def forward(self, t):
            return self.embedding(t.float().unsqueeze(0).unsqueeze(1))  # Expecting t shape: [B]
        
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, dropout=0.0, num_groups=32, emb_dim=128):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.adanorm1 = AdaUNet_Encoding.AdaNorm(out_channels, emb_dim, num_groups)
            self.dropout1 = nn.Dropout2d(dropout)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.adanorm2 = AdaUNet_Encoding.AdaNorm(out_channels, emb_dim, num_groups)
            self.dropout2 = nn.Dropout2d(dropout)

            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        def forward(self, x, t_emb):
            residual = self.shortcut(x)
            x = F.leaky_relu(self.adanorm1(self.conv1(x), t_emb), 0.1)
            x = self.dropout1(x)
            x = self.adanorm2(self.conv2(x), t_emb)
            x = self.dropout2(x)
            return F.leaky_relu(x + residual, 0.1)
        


    class DownSample(nn.Module):
        def __init__(self, in_channels, out_channels, emb_dim):
            super().__init__()
            self.conv = AdaUNet_Encoding.DoubleConv(in_channels, out_channels, emb_dim=emb_dim)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x, t_emb):
            down = self.conv(x, t_emb)
            p = self.pool(down)
            return down, p

    def __init__(self, in_channels, emb_dim=128):
        super().__init__()
        self.timestep_embedding = AdaUNet_Encoding.TimestepEmbedding(emb_dim)
        self.down_convolution_1 = AdaUNet_Encoding.DownSample(in_channels, 64, emb_dim)
        self.down_convolution_2 = AdaUNet_Encoding.DownSample(64, 128, emb_dim)
        self.bottle_neck = AdaUNet_Encoding.DoubleConv(128, 256, emb_dim=emb_dim)

    def forward(self, x, timestep):
        t_emb = self.timestep_embedding(timestep)
        down_1, p1 = self.down_convolution_1(x, t_emb)
        down_2, p2 = self.down_convolution_2(p1, t_emb)
        b = self.bottle_neck(p2, t_emb)
        b = F.adaptive_avg_pool2d(b, (1, 1)).squeeze(-1).squeeze(-1)
        return b


class SimpleMLP(nn.Module):
    class L1NormLayer(nn.Module):
        def forward(self, x):
            return x / x.abs().sum(dim=1, keepdim=True)
    

    def __init__(self, input_size, output_size, hidden_size=100, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Add dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Add dropout
        x = self.fc3(x)

        return x

class Complicated_Delta_LTT_model(nn.Module):
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image : bool = False):
        super().__init__()

        self.unet = AdaUNet_Encoding(
            in_channels=3
        )

        self.unet2 = AdaUNet_Encoding(
            in_channels=3
        )
        self.mlp = SimpleMLP(
            input_size=512+1+1,
            output_size=2,
            hidden_size=512,
            dropout=mlp_dropout
        )


    def forward(self, x, x_prev, current_timestep, steps_left):
        out = self.unet(x, current_timestep)
        out_prev = self.unet2(x_prev, current_timestep)
        out = torch.cat([out, out_prev, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
        out = self.mlp(out)
        # out = torch.sigmoid(out)
        return out



class Delta_LTT_model_using_Bottleneck(nn.Module):

    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0):
        super().__init__()
        # Add processing for additional features
        self.mlp = SimpleMLP(
            input_size=256+1+1,
            output_size=1,
            hidden_size=512,
            dropout=mlp_dropout
        )

    def forward(self, bottleneck, current_timestep, steps_left):
        out = torch.cat([bottleneck, current_timestep.expand((bottleneck.shape[0])).unsqueeze(1), steps_left.expand((bottleneck.shape[0])).unsqueeze(1)], dim=1)
        out = self.mlp(out)
        out = torch.sigmoid(out)

        return out

class Delta_LTT_model(nn.Module):

    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image : bool = False):
        super().__init__()

        self.unet = SimpleUNet_Encoding(
            in_channels=3
        )
        self.just_image = just_image
        # Add processing for additional features
        if self.just_image:
            self.mlp = SimpleMLP(
                input_size=256,
                output_size=1,
                hidden_size=512,
                dropout=mlp_dropout
            )
        else:
            self.mlp = SimpleMLP(
                input_size=256+1+1,
                output_size=1,
                hidden_size=512,
                dropout=mlp_dropout
            )


    def forward(self, x, current_timestep, steps_left):
        out = self.unet(x)

        if not self.just_image:
            out = torch.cat([out, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
        out = self.mlp(out)
        out = torch.sigmoid(out)

        return out


class Delta_LTT_model_using_Bottleneck(nn.Module):

    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0):
        super().__init__()
        # Add processing for additional features
        self.mlp = SimpleMLP(
            input_size=256+1+1,
            output_size=1,
            hidden_size=512,
            dropout=mlp_dropout
        )

    def forward(self, bottleneck, current_timestep, steps_left):
        out = torch.cat([bottleneck, current_timestep.expand((bottleneck.shape[0])).unsqueeze(1), steps_left.expand((bottleneck.shape[0])).unsqueeze(1)], dim=1)
        out = self.mlp(out)
        out = torch.sigmoid(out)

        return out

class Tiny_Delta_LTT_model(nn.Module):
    """
    Much smaller, parameter-efficient model for timestep prediction.
    - Lightweight CNN: 2 conv layers, max 32 channels, with GroupNorm and LeakyReLU
    - Small MLP: 1 hidden layer, 32 units
    - Supports just_image and full input modes
    """
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image: bool = False):
        super().__init__()
        self.just_image = just_image
        # Tiny CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Output of CNN is 32-dim
        mlp_in = 32 if just_image else 32 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(32, 1),
        )
    def forward(self, x, current_timestep=None, steps_left=None):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)  # Flatten
        if not self.just_image:
            # current_timestep and steps_left expected as shape [B] or [B,1]
            if current_timestep is not None and steps_left is not None:
                if current_timestep.dim() == 1:
                    current_timestep = current_timestep.unsqueeze(1)
                if steps_left.dim() == 1:
                    steps_left = steps_left.unsqueeze(1)
                out = torch.cat([out, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
            else:
                raise ValueError("current_timestep and steps_left must be provided in non-just_image mode")
        out = self.mlp(out)
        out = torch.sigmoid(out)
        return out

# Scaled-up Tiny_Delta_LTT_model: deeper CNN, larger MLP, still <100k params
class Medium_Delta_LTT_model(nn.Module):
    """
    Scaled-up version of Tiny_Delta_LTT_model, still parameter-efficient (<100k params).
    - CNN: 3 conv layers, up to 64 channels, GroupNorm, LeakyReLU
    - MLP: 2 hidden layers, 64 units each
    - Supports just_image and full input modes
    """
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image: bool = False):
        super().__init__()
        self.just_image = just_image
        # Slightly larger CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Output of CNN is 64-dim
        mlp_in = 64 if just_image else 64 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x, current_timestep=None, steps_left=None):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)  # Flatten
        if not self.just_image:
            if current_timestep is not None and steps_left is not None:
                if current_timestep.dim() == 1:
                    current_timestep = current_timestep.unsqueeze(1)
                if steps_left.dim() == 1:
                    steps_left = steps_left.unsqueeze(1)
                out = torch.cat([out, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
            else:
                raise ValueError("current_timestep and steps_left must be provided in non-just_image mode")
        out = self.mlp(out)
        out = torch.sigmoid(out)
        return out
# Parameter count for this model (just_image=False):
# CNN: (3*32*3*3)+(32*48*3*3)+(48*64*3*3)+norms+biases ≈ 30k
# MLP: (66*64)+(64*64)+(64*1)+biases ≈ 8k
# Total: ~38k (well under 100k)

# Scaled-up Medium_Delta_LTT_model: 3x wider/deeper, still parameter-efficient
class Large_Delta_LTT_model(nn.Module):
    """
    Scaled-up version of Medium_Delta_LTT_model (about 3x wider/deeper, still <100k params).
    - CNN: 3 conv layers, up to 192 channels, GroupNorm, LeakyReLU
    - MLP: 2 hidden layers, 192 units each
    - Supports just_image and full input modes
    """
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image: bool = False):
        super().__init__()
        self.just_image = just_image
        # Much larger CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 144, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(12, 144),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(144, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(12, 192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Output of CNN is 192-dim
        mlp_in = 192 if just_image else 192 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(192, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(192, 1),
        )
    def forward(self, x, current_timestep=None, steps_left=None):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)  # Flatten
        if not self.just_image:
            if current_timestep is not None and steps_left is not None:
                if current_timestep.dim() == 1:
                    current_timestep = current_timestep.unsqueeze(1)
                if steps_left.dim() == 1:
                    steps_left = steps_left.unsqueeze(1)
                out = torch.cat([out, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
            else:
                raise ValueError("current_timestep and steps_left must be provided in non-just_image mode")
        out = self.mlp(out)
        out = torch.sigmoid(out)
        return out
# Parameter count for this model (just_image=False):
# CNN: (3*96*3*3)+(96*144*3*3)+(144*192*3*3)+norms+biases ≈ 270k
# MLP: (194*192)+(192*192)+(192*1)+biases ≈ 74k
# Total: ~344k (over 100k, but you can reduce channels to 64/96/128 and MLP to 64 to stay <100k)

# Eighth-size Huge_Delta_LTT_model: reduce MLP width from 512 to 256, 2 hidden layers
class Huge_Delta_LTT_model(nn.Module):
    """
    Much larger model than Delta_LTT_model, but with 1/16 the MLP parameters of the original Huge_Delta_LTT_model.
    - CNN: deeper and wider, outputs [B, 256, 8, 8] bottleneck
    - MLP: 2 hidden layers, 256 units each (was 3x1024)
    - Supports just_image and full input modes
    """
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image: bool = False):
        super().__init__()
        self.just_image = just_image
        # Large CNN encoder: outputs [B, 256, 8, 8]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.1, inplace=True),
            # Output: [B, 256, 4, 4] for input 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B, 256, 8, 8]
        )
        mlp_in = 256*8*8 if just_image else 256*8*8 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 1),
        )
    def forward(self, x, current_timestep=None, steps_left=None):
        out = self.cnn(x)  # [B, 256, 8, 8]
        out_flat = out.view(out.size(0), -1)
        if not self.just_image:
            if current_timestep is not None and steps_left is not None:
                if current_timestep.dim() == 1:
                    current_timestep = current_timestep.unsqueeze(1)
                if steps_left.dim() == 1:
                    steps_left = steps_left.unsqueeze(1)
                out_flat = torch.cat([out_flat, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
            else:
                raise ValueError("current_timestep and steps_left must be provided in non-just_image mode")
        out_mlp = self.mlp(out_flat)
        out_mlp = torch.sigmoid(out_mlp)
        return out_mlp  # Return both prediction and bottleneck

class Huge_Delta_LTT_CNN_Bigger(nn.Module):
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image: bool = False):
        super().__init__()
        self.just_image = just_image
        # Larger CNN encoder: outputs [B, 512, 8, 8]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B, 512, 8, 8]
        )
        mlp_in = 512*8*8 if just_image else 512*8*8 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 1),
        )
    def forward(self, x, current_timestep=None, steps_left=None):
        out = self.cnn(x)
        out_flat = out.view(out.size(0), -1)
        if not self.just_image:
            if current_timestep is not None and steps_left is not None:
                if current_timestep.dim() == 1:
                    current_timestep = current_timestep.unsqueeze(1)
                if steps_left.dim() == 1:
                    steps_left = steps_left.unsqueeze(1)
                out_flat = torch.cat([out_flat, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
            else:
                raise ValueError("current_timestep and steps_left must be provided in non-just_image mode")
        out_mlp = self.mlp(out_flat)
        out_mlp = torch.sigmoid(out_mlp)
        return out_mlp


class Huge_Delta_LTT_MLP_Bigger(nn.Module):
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image: bool = False):
        super().__init__()
        self.just_image = just_image
        # Original CNN encoder: outputs [B, 256, 8, 8]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B, 256, 8, 8]
        )
        mlp_in = 256*8*8 if just_image else 256*8*8 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, 1),
        )
    def forward(self, x, current_timestep=None, steps_left=None):
        out = self.cnn(x)
        out_flat = out.view(out.size(0), -1)
        if not self.just_image:
            if current_timestep is not None and steps_left is not None:
                if current_timestep.dim() == 1:
                    current_timestep = current_timestep.unsqueeze(1)
                if steps_left.dim() == 1:
                    steps_left = steps_left.unsqueeze(1)
                out_flat = torch.cat([out_flat, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
            else:
                raise ValueError("current_timestep and steps_left must be provided in non-just_image mode")
        out_mlp = self.mlp(out_flat)
        out_mlp = torch.sigmoid(out_mlp)
        return out_mlp
# Half-size Huge_Bottleneck_Delta: reduce MLP width from 1024 to 512


class Ginormous_Delta_LTT(nn.Module):
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image: bool = False):
        super().__init__()
        self.just_image = just_image
        # Larger CNN encoder: outputs [B, 512, 8, 8]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B, 512, 8, 8]
        )
        mlp_in = 512*8*8 if just_image else 512*8*8 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(512, 1),
        )
    def forward(self, x, current_timestep=None, steps_left=None):
        out = self.cnn(x)
        out_flat = out.view(out.size(0), -1)
        if not self.just_image:
            if current_timestep is not None and steps_left is not None:
                if current_timestep.dim() == 1:
                    current_timestep = current_timestep.unsqueeze(1)
                if steps_left.dim() == 1:
                    steps_left = steps_left.unsqueeze(1)
                out_flat = torch.cat([out_flat, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
            else:
                raise ValueError("current_timestep and steps_left must be provided in non-just_image mode")
        out_mlp = self.mlp(out_flat)
        out_mlp = torch.sigmoid(out_mlp)
        return out_mlp
class Huge_Bottleneck_Delta(nn.Module):
    """
    Takes [B, 256, 8, 8] bottleneck, predicts timestep ratio.
    - MLP: 2 hidden layers, 256 units each (matches Huge_Delta_LTT_model)
    """
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256*8*8 + 1 + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 1),
        )
    def forward(self, bottleneck, current_timestep, steps_left):
        out = bottleneck.view(bottleneck.size(0), -1)
        if current_timestep.dim() == 1:
            current_timestep = current_timestep.unsqueeze(1)
        if steps_left.dim() == 1:
            steps_left = steps_left.unsqueeze(1)
        out = torch.cat([out, current_timestep.expand((out.shape[0])).unsqueeze(1), steps_left.expand((out.shape[0])).unsqueeze(1)], dim=1)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        return out

# --- Fancy_Delta_LTT_Model: Multi-scale, SE-blocks, Self-Attention, Dynamic Conv ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SelfAttention2d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)  # B, HW, C//8
        proj_key = self.key_conv(x).view(B, -1, H*W)  # B, C//8, HW
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, H*W)  # B, C, HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dynamic_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels * in_channels * kernel_size * kernel_size, 1)
        )
    def forward(self, x):
        B, C, H, W = x.size()
        weight = self.dynamic_gen(x)
        # [B, out_channels, in_channels, k, k]
        weight = weight.view(B, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        # Unfold input: [B, in_channels * k * k, H*W]
        x_unf = F.unfold(x, self.kernel_size, padding=self.kernel_size//2)
        # [B, in_channels, k*k, H*W]
        x_unf = x_unf.view(B, self.in_channels, self.kernel_size * self.kernel_size, H*W)
        # Reshape weights for matmul: [B, out_channels, in_channels * k*k]
        weight = weight.view(B, self.out_channels, self.in_channels * self.kernel_size * self.kernel_size)
        # [B, out_channels, H*W] = [B, out_channels, in_channels*k*k] x [B, in_channels*k*k, H*W]
        out = torch.bmm(weight, x_unf.view(B, self.in_channels * self.kernel_size * self.kernel_size, H*W))
        # [B, out_channels, H, W]
        out = out.view(B, self.out_channels, H, W)
        return out

class Fancy_Delta_LTT_Model(nn.Module):
    """
    Multi-scale CNN with SE-blocks, self-attention, and dynamic convolution for timestep prediction.
    Fancier MLP: 3 hidden layers, GELU, LayerNorm, Dropout, residual connection.
    """
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0, just_image: bool = False):
        super().__init__()
        self.just_image = just_image
        # Multi-scale encoder with SE and dynamic conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(32)
        )
        self.conv2 = nn.Sequential(
            DynamicConv2d(32, 64, 3),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(256)
        )
        self.attn = SelfAttention2d(256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        mlp_in = 256 if just_image else 256 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 1),
        )
    def forward(self, x, current_timestep=None, steps_left=None):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.attn(out)
        out = self.pool(out).view(out.size(0), -1)
        if not self.just_image:
            if current_timestep is not None and steps_left is not None:
                if current_timestep.dim() == 1:
                    current_timestep = current_timestep.unsqueeze(1)
                if steps_left.dim() == 1:
                    steps_left = steps_left.unsqueeze(1)
                out = torch.cat([
                    out,
                    current_timestep.expand((out.shape[0])).unsqueeze(1),
                    steps_left.expand((out.shape[0])).unsqueeze(1)
                ], dim=1)
            else:
                raise ValueError("current_timestep and steps_left must be provided in non-just_image mode")
        out_mlp = self.mlp(out)
        out_mlp = torch.sigmoid(out_mlp)
        return out_mlp

if __name__ == "__main__":

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # Dataset
    # data_dir = 'train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0'
    # steps = 5
    # num_train = 160000 #40000 if clever
    # num_valid = 50
    # train_batch_size = 50
    # optimal_params_path = "opt_t" #opt_t_clever_initialisation

    # # Initialize TensorBoard writer
    # learning_rate = 1e-4



    # def custom_collate_fn(batch):
    #     collated_batch = []
    #     for samples in zip(*batch):
    #         if any(item is None for item in samples):
    #             collated_batch.append(None)
    #         else:
    #             collated_batch.append(torch.utils.data._utils.collate.default_collate(samples))
    #     return collated_batch
    
    # valid_dataset = LTTDataset(dir=os.path.join(data_dir, "validation"), size=num_valid, train_flag=False, use_optimal_params=True,optimal_params_path=optimal_params_path) 
    # train_dataset = LTTDataset(dir=os.path.join(data_dir, "train"), size=num_train, train_flag=True, use_optimal_params=True, optimal_params_path=optimal_params_path)

    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     collate_fn=custom_collate_fn,
    #     batch_size=train_batch_size,  # Adjust batch size as needed
    #     shuffle=True,
    # )

    # valid_loader = DataLoader(
    #     dataset=valid_dataset,
    #     collate_fn=custom_collate_fn,
    #     batch_size=50,  # Adjust batch size as needed
    #     shuffle=False,
    # )

    # model = LTT_model(steps = steps)
    # loss_fn = nn.MSELoss()#CrossEntropyLoss()
    # model = model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


    # img, latent, _ = train_dataset[0]
    # img = img.to(device)
    # latent = latent.to(device)
    
    # print(model(latent.unsqueeze(0)))
    Complicated_Delta_LTT_model = Complicated_Delta_LTT_model(steps=10, mlp_dropout=0.0)
    # number of parameters
    print(f"Number of parameters in Complicated_Delta_LTT_model: {sum(p.numel() for p in Complicated_Delta_LTT_model.parameters() if p.requires_grad)}")

    Delta_LTT_model = Delta_LTT_model(steps=10, mlp_dropout=0.0, just_image=False)

    # number of parameters
    print(f"Number of parameters in Delta_LTT_model: {sum(p.numel() for p in Delta_LTT_model.parameters() if p.requires_grad)}")


    Tiny_Delta_LTT_model = Tiny_Delta_LTT_model(steps=10, mlp_dropout=0.0, just_image=False)
    # number of parameters
    print(f"Number of parameters in Tiny_Delta_LTT_model: {sum(p.numel() for p in Tiny_Delta_LTT_model.parameters() if p.requires_grad)}")

    Medium_Delta_LTT_model = Medium_Delta_LTT_model(steps=10, mlp_dropout=0.0, just_image=False)
    # number of parameters
    print(f"Number of parameters in Medium_Delta_LTT_model: {sum(p.numel() for p in Medium_Delta_LTT_model.parameters() if p.requires_grad)}")

    Large_Delta_LTT_model = Large_Delta_LTT_model(steps=10, mlp_dropout=0.0, just_image=False)
    # number of parameters
    print(f"Number of parameters in Large_Delta_LTT_model: {sum(p.numel() for p in Large_Delta_LTT_model.parameters() if p.requires_grad)}")

    Huge_Delta_LTT_model = Huge_Delta_LTT_model(steps=10, mlp_dropout=0.0, just_image=False)
    # number of parameters
    print(f"Number of parameters in Huge_Delta_LTT_model: {sum(p.numel() for p in Huge_Delta_LTT_model.parameters() if p.requires_grad)}")

    Delta_LTT_model_using_Bottleneck = Delta_LTT_model_using_Bottleneck(steps=10, mlp_dropout=0.0)
    # number of parameters
    print(f"Number of parameters in Delta_LTT_model_using_Bottleneck: {sum(p.numel() for p in Delta_LTT_model_using_Bottleneck.parameters() if p.requires_grad)}")
    
    Huge_Bottleneck_Delta = Huge_Bottleneck_Delta(steps=10, mlp_dropout=0.0)
    # number of parameters
    print(f"Number of parameters in Huge_Bottleneck_Delta: {sum(p.numel() for p in Huge_Bottleneck_Delta.parameters() if p.requires_grad)}")

    Huge_Delta_LTT_CNN_Bigger = Huge_Delta_LTT_CNN_Bigger(steps=10, mlp_dropout=0.0, just_image=False)
    # number of parameters
    print(f"Number of parameters in Huge_Delta_LTT_CNN_Bigger: {sum(p.numel() for p in Huge_Delta_LTT_CNN_Bigger.parameters() if p.requires_grad)}")

    Huge_Delta_LTT_MLP_Bigger = Huge_Delta_LTT_MLP_Bigger(steps=10, mlp_dropout=0.0, just_image=False)
    # number of parameters
    print(f"Number of parameters in Huge_Delta_LTT_MLP_Bigger: {sum(p.numel() for p in Huge_Delta_LTT_MLP_Bigger.parameters() if p.requires_grad)}")

    Ginormous_Delta_LTT = Ginormous_Delta_LTT(steps=10, mlp_dropout=0.0, just_image=False)
    # number of parameters
    print(f"Number of parameters in Ginormous_Delta_LTT: {sum(p.numel() for p in Ginormous_Delta_LTT.parameters() if p.requires_grad)}")

    Fancy_Delta_LTT_Model = Fancy_Delta_LTT_Model(steps=10, mlp_dropout=0.0, just_image=False)
    # number of parameters
    print(f"Number of parameters in Fancy_Delta_LTT_Model: {sum(p.numel() for p in Fancy_Delta_LTT_Model.parameters() if p.requires_grad)}")




