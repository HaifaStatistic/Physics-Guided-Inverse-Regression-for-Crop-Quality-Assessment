
import torch
import torch.nn as nn
from torch.nn import MSELoss,CrossEntropyLoss,BCELoss,BCEWithLogitsLoss,Sequential,Dropout,Linear,Sigmoid,Flatten,Module
import torch.nn.functional as F
import numpy as np
import random
from torchvision.models import resnet18

seed_val = 10
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def gaussian_kernel(size: int, sigma: float):
    coords = np.linspace(-size // 2 + 1, size // 2, size)
    g = np.exp(-((coords / sigma) ** 2) / 2)
    g /= g.sum()
    g = np.outer(g, g)
    return torch.tensor(g, dtype=torch.float32)

kernel_size = 5  # Size of the Gaussian kernel
sigma =10     # Standard deviation, controls the spread of the Gaussian
num_channels=3



class DualResNet3(nn.Module):
    def __init__(self,model_type):
        super(DualResNet3, self).__init__()
  
        # Load two ResNet18 models
        self.resnet1 = resnet18(weights='ResNet18_Weights.DEFAULT')
        self.resnet2 = resnet18(weights='ResNet18_Weights.DEFAULT')

        # Remove the classification heads
        self.features1 = nn.Sequential(*list(self.resnet1.children())[:-1])

        self.fusion3 = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
        )


    def physical_loss(self, concentration):
        second_order_derivation_x = torch.tensor([[[[1, -2, 1],
                                                    [2, -4, 2],
                                                    [1, -2, 1]]]], dtype=torch.float32)
        second_order_derivation_y = torch.tensor([[[[1, 0, -1],
                                                    [0, 0, 0],
                                                    [-1, 0, 1]]]], dtype=torch.float32)

        # Expand the kernel to handle 3 channels
        second_order_derivation_kernel_x = second_order_derivation_x.repeat(3, 3, 1, 1)
        second_order_derivation_kernel_y = second_order_derivation_y.repeat(3, 3, 1, 1)

        # Move kernel to the same device as concentration
        second_order_derivation_kernel_x = second_order_derivation_kernel_x.to(concentration.device)
        second_order_derivation_kernel_y = second_order_derivation_kernel_y.to(concentration.device)

        # Apply convolution
        term = (F.conv2d(concentration, second_order_derivation_kernel_x, padding=1) +
                F.conv2d(concentration, second_order_derivation_kernel_y, padding=1))

        # Calculate mean and variance per image in the batch
        mean_term_per_image = torch.mean(term, dim=[1, 2, 3])  # Mean across channels and spatial dimensions
        var_loss_per_image = torch.mean((term - mean_term_per_image.unsqueeze(1).unsqueeze(2).unsqueeze(3)) ** 2, dim=[1, 2, 3])
        return var_loss_per_image


    def forward(self, x1, x2,model_type):
        if model_type=='e1': 
            x1 = self.features1(x1)
            x1 = x1.view(x1.size(0), -1)                    
            return self.fusion3(x1)
        elif model_type=='e2':
            weight1 = self.physical_loss(x2)
            weight1=weight1/10
            x2 = self.features1(x2)
            x2 = x2.view(x2.size(0), -1)   
            return self.fusion3(x2), weight1

          

class ImageTransformer11(nn.Module):
    def __init__(self,MT,sigma, noise_factor=0.0):
        super(ImageTransformer11, self).__init__()
        self.MT=MT
        self.sigma=sigma
        gaussian_kernel_tensor = gaussian_kernel(kernel_size, self.sigma)
        gaussian_kernel_tensor = gaussian_kernel_tensor.view(1, 1, kernel_size, kernel_size).repeat(num_channels, 1, 1, 1)

        # Blurring layer
        self.blur_conv = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=num_channels)
        self.blur_conv.weight.data = gaussian_kernel_tensor
        self.blur_conv.weight.requires_grad = False  # Freeze the weights

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)        
        self.trans_conv = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)#110x350 -> 220x700 Cucumbers
        # Applying He initialization
        self.leaky_relu = nn.LeakyReLU()

       # Apply Xavier Initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.trans_conv.weight)

        self.noise_factor = noise_factor
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        if not self.training:  # this checks if the module is in training mode
            x = x + self.noise_factor * torch.randn_like(x)          
        x = self.blur_conv(x)        
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = (self.trans_conv(x))
        return x


