import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import random
import torch.nn.functional as F
seed_val = 10
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def bucket_data_by_label_range(data, labels):
    """
    Bucket data based on label ranges.

    Parameters:
    - data: np.array of shape (num_samples, channels, height, width)
    - labels: np.array of shape (num_samples,)
    - ranges: List of tuples defining the ranges. E.g., [(0,1), (1,2), ...]

    Returns:
    - bucketed_data: Dictionary with keys as range and values as data corresponding to that range.
    """
    ranges = [(i, i+1) for i in np.arange(0, 25, 1)]
    bucketed_data = {}
    
    for r in ranges:
        start, end = r
        subset_indices = np.where((labels >= start) & (labels < end))[0]
        bucketed_data[r] = data[subset_indices]
    
    return bucketed_data
def rescal(transform_images):
    min_val = transform_images.min()
    max_val = transform_images.max()
    return (transform_images - min_val) / (max_val - min_val)

def visualize_samples(epoch,real_images,transform_images, num_examples=10):
    real_images=real_images.cpu().detach()#.numpy()
    transform_images=transform_images.cpu().detach()#.numpy()
    real_images=rescal(real_images)
    transform_images=rescal(transform_images)

    print("real_images.min(), real_images.max()",real_images.min(), real_images.max())
    print("transform_images.min(), transform_images.max()",transform_images.min(), transform_images.max())
    fig, axarr = plt.subplots(2, num_examples, figsize=(15, 6))
    for idx in range(num_examples):
        axarr[0, idx].imshow(transform_images[idx].permute(1,2,0).squeeze())
        axarr[0, idx].axis('off')
        axarr[1, idx].imshow(real_images[idx].permute(1,2,0).squeeze())
        axarr[1, idx].axis('off')

    axarr[0, 0].set_title('Transform Images')
    axarr[1, 0].set_title('Real Images')

    plt.tight_layout()
    plt.savefig(f'Train_output_epoch_{epoch}.png')

def variance_minimization_loss(concentration,model_type):
        second_order_derivation_x =torch.tensor([[[[1, -2, 1],
                                                    [2, -4, 2],
                                                    [1, -2, 1]]]], dtype=torch.float32) 

        second_order_derivation_y =torch.tensor([[[[1, 0, -1],
                                                    [0, 0, 0],
                                                    [-1, 0, 1]]]], dtype=torch.float32)             
        # Expand the kernel to handle 3 channels
        second_order_derivation_kernel_x = second_order_derivation_x.repeat(3, 3, 1, 1)  # Make it 3x3x3 to apply to RGB image
        second_order_derivation_kernel_y = second_order_derivation_y.repeat(3, 3, 1, 1)  # Make it 3x3x3 to apply to RGB image

        # Move kernel to the same device as concentration
        second_order_derivation_kernel_x = second_order_derivation_kernel_x.to(concentration.device)
        second_order_derivation_kernel_y = second_order_derivation_kernel_y.to(concentration.device)
        
        # Apply convolution
        term = (nn.functional.conv2d(concentration, second_order_derivation_kernel_x, padding=1) + nn.functional.conv2d(concentration, second_order_derivation_kernel_y, padding=1) )
        mean_term = torch.mean(term)
        return torch.mean((term - mean_term) ** 2)
    
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
def train_transform(model, train_loader,test_loader, num_epochs, learning_rate, device,model_tune):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    test_losses=[]
    plot_size=True
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_outputs=[]
        all_labels=[]        
        for batch_idx, (data,label_value,filename,_) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss=variance_minimization_loss(outputs,model_tune)
            
            loss.backward()
            optimizer.step()
            train_loss+= loss.item()
        train_loss /= len(train_loader.dataset)
        '''
        model.eval()
        test_loss = 0.0
        all_outputs=[]
        all_labels=[]
        with torch.no_grad():
            for data,label_value,filename,_ in test_loader:
                data = data.to(device)
                outputs = model(data)
                if plot_size:
                    plot_size=False
                    #print("data.shape,outputs.shape ",data.shape,outputs.shape)
            
                loss = variance_minimization_loss(outputs,model_tune)
                test_loss += loss.item()
        if (len(test_loader))!=0:
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            #visualize_samples(epoch,data,outputs, num_examples=10)
            #print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        else:
            test_losses=0
        '''
    return model
