import pandas as pd
import torch
from torchvision.transforms import Compose, ToTensor, Resize, RandomVerticalFlip, RandomHorizontalFlip,Normalize,RandomRotation
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD,Adam
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch.nn import MSELoss,CrossEntropyLoss,BCELoss,BCEWithLogitsLoss,Sequential,Dropout,Linear,Sigmoid,Flatten
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
from torch.utils.data import ConcatDataset
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import random

import sys

# Get the path to the directory containing the train.py script
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
# Add the scripts directory to sys.path
sys.path.append(scripts_dir)

# Now you can import the module
#from train import train_transform

from train import train_transform
from model import ImageTransformer11,DualResNet3
import torch.nn as nn
import csv



machine_seed=20


def regr_metrics(all_labels_np, all_outputs_np):
    #all_labels_np.extend(labels.detach().cpu().numpy())
    #all_outputs_np.extend(outputs.detach().cpu().numpy())
    #mse=mean_squared_error(all_labels_np, all_outputs_np)
    rmse = sqrt(mean_squared_error(all_labels_np, all_outputs_np))
    mae = mean_absolute_error(all_labels_np, all_outputs_np)
    r2 = r2_score(all_labels_np, all_outputs_np)
    errors = all_labels_np - all_outputs_np
    std_errors = np.std(errors)
    return rmse,mae,std_errors ,r2   
seed_val = 10
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###################
datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
#print("datapath ",datapath)
# Define the directories and the path to the labels CSV file
directories = [datapath+"\\shamp_first_round", datapath+"\\shamp_second_round"]
labels_csv_path =datapath+'\\mushrooms_fitness.csv'
local_path = os.path.dirname(__file__)  # Define local_path if it's not already defined
csv_file_path = os.path.join(local_path, 'mushrooms_result.csv')

# Read the labels CSV file
df_labels = pd.read_csv(labels_csv_path)

# Find all image files with .jpg or .png extensions in the specified directories
image_files = [
    os.path.join(dp, f) for directory in directories
    for dp, dn, filenames in os.walk(directory)
    for f in filenames if f.endswith(('.jpg', '.png'))
]
"""
# Print statements for debugging
print(f"Directories: {directories}")
print(f"Labels CSV Path: {labels_csv_path}")
print(f"CSV File Path: {csv_file_path}")
print(f"Number of Image Files Found: {len(image_files)}")
"""
                        
means = []
stds = []

# For each image file
for image_file in image_files:
    # Load image
    image = Image.open(image_file)

    # Convert to tensor
    image_tensor = ToTensor()(image)

    # Compute and store mean and std
    means.append(torch.mean(image_tensor))
    stds.append(torch.std(image_tensor))

# Compute overall mean and std
mean = np.mean(means)
std = np.mean(stds)

# Define the augmentation
transform = Compose([
    Resize((224,224)),  # Resize all images to the same size
    RandomVerticalFlip(p=0.5),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(360),
    ToTensor(),  # Convert images to PyTorch tensors
    Normalize(mean=mean, std=std)
])


class CucumberDataset(Dataset):
    def __init__(self, image_files, labels_df,transform=None):
        self.transform = transform
        self.labels_df = labels_df.dropna()
        self.labels_df['A'] = self.labels_df['A'].astype(str)
        self.labels_df['folder'] = self.labels_df['folder'].astype(str)
        valid_groups = labels_df['A'].unique().tolist()
        self.image_paths = image_files


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        # Extract group and cucumber_number from the image file name
        folder, group, cucumber_number = os.path.splitext(os.path.basename(image_path))[0].split('-')
        filename=f"{folder}-{group}-{cucumber_number}.png"
        try:
            label_value = self.labels_df.loc[(self.labels_df['folder'] == folder) & (self.labels_df['A'] == group), 'fitness'].values[0]   
            bin_label = self.labels_df.loc[(self.labels_df['folder'] == folder) & (self.labels_df['A'] == group), 'quartile_group'].values[0]   

            if 0<= label_value <= 5:

                label = torch.tensor(label_value).float()
            else:
                return None
        except IndexError:
            return None

        return image, label, filename,bin_label


# Define the Dataset
dataset1 = CucumberDataset(image_files, labels_df=df_labels, transform=transform)


# Filter out None values
dataset2= [item for item in dataset1 if item is not None]
spl2=[item[-1] for item in dataset2]

all_train_dataset, test_dataset = train_test_split(dataset2, test_size=0.3,stratify=spl2, random_state=1)

print("len(all_train_dataset),len(test_dataset)",len(all_train_dataset),len(test_dataset))

spl2=[item[-1] for item in all_train_dataset]

multipl=0.3
batch_size_test=50

firs_time=0
curent_loop=0
all_split_sizes=[ 15, 31, 47, 63,79,95, 111, 127, 143]

all_test_PDE=['d']
for model_tune in all_test_PDE:
    print("!!!!!!!!!!!!!!!!!!model_tune ",model_tune)
    firs_time+=1
    average_all_RMSE_c=[]
    average_all_RMSE_e=[]
    for split_size in all_split_sizes:
        print("split_size:  ",split_size)
        batch_size=int(split_size*multipl)
        for k in range(machine_seed,machine_seed+50):
            
            print("\n")
            print("random_state: ",k)

            train_dataset, _ = train_test_split(all_train_dataset, train_size=split_size,stratify=spl2, random_state=k)
            train_size1, test_size1=len(train_dataset),len(test_dataset)            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=0, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

            # Define the device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


            epoch_N=[18]
            sigma_N=[10]

            model_type=['e2']
            for num_epochsT in epoch_N:
                num_epochsT=18

                for model_T in model_type:
                    curent_loop+=1

                    sigma=10

                    MT='a'
                    
                    seed_val = 10
                    random.seed(seed_val)
                    np.random.seed(seed_val)
                    torch.manual_seed(seed_val)

                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    model_transformer=ImageTransformer11(MT,sigma).to(device)

                    learning_rate=0.001
                    
                    model_transformer=train_transform(model_transformer, train_loader, test_loader,num_epochsT, learning_rate, device,model_tune)
                    model_transformer.eval()
                    # Define the model


                    mdl1='e1'
                    mdl2='e2'

                    model1=DualResNet3(mdl1).to(device)
                    model2=DualResNet3(mdl2).to(device)
               

                    # Define the criterion
                    criterion = MSELoss()
                    learning_rate =0.0001

                    # Define the optimizer
                    optimizer1 = Adam(model1.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
                    optimizer2 = Adam(model2.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)


                    # Placeholder to save the loss for each epoch
                    train_losses = []
                    # Training loop
                    epoch_losses=[]
                    all_mse_test=[]
                    all_rmse_test=[]
                    all_mae_test=[]
                    all_r2_test=[]
                    std_errors_no_outliers_test=[]
                    std_errors_test=[]                            
                    total_test_losses=[]
                    num_epochs =55############################################
                    average_RMSE_loss=[]
                    average_R_Sq=[]
                    # Initialize data structures to store metrics
                    metrics_model1 = []
                    metrics_model2 = []

                    R2_1=[]
                    R2_2=[]
                    R2_3=[]
                    R2_4=[]

                    for epoch in range(num_epochs):
                        train_losses = 0.0
                        train_losses2=0.0
                        train_losses3=0.0
                        num_batches = 0    
                        avg_loss=0.0
                        all_outputs = []
                        all_labels = []  
                        model1.train()
                        model2.train()

                        model_transformer.eval()
                        for i, (images, labels,filename,_) in enumerate(train_loader):
                            # Move images and labels to the GPU if available
                            images = images.to(device)
                            labels = labels.to(device)

                            outputs = model_transformer(images).detach()

                            #print("images.shape, labels.shape",images.shape, labels.shape)
                            # Forward pass
                            outputs1= model1(images,outputs,mdl1)
                            outputs2,weight1= model2(images,outputs,mdl2)
                            

                            outputs1 = outputs1.view(outputs1.shape[0])  # flatten the outputs RESNET
                            outputs2 = outputs2.view(outputs2.shape[0])  # flatten the outputs RESNET

                            loss1 = criterion(outputs1, labels)
                            loss2 = criterion(outputs2, labels)

                            
                            total_loss = loss1 + loss2

                            # Backward and optimize
                            optimizer1.zero_grad()
                            optimizer2.zero_grad()


                            total_loss.backward()

                            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer1.step()
                            optimizer2.step()

                        epoch_losses.append(avg_loss)        
                        if epoch>25:
                            model1.eval()
                            model2.eval()

                            with torch.no_grad():
                                test_losses = []
                                all_outputs1 = []
                                all_outputs2 = []

                                all_labels = []  
                                #all_filename=[]            
                                for i, (images, labels,filename,_) in enumerate(test_loader):

                                    images = images.to(device)
                                    labels = labels.to(device)

                                    outputs = model_transformer(images).detach()

                                    outputs1= model1(images,outputs,mdl1)
                                    outputs2,weight1= model2(images,outputs,mdl2)
                                    outputs1 = outputs1.view(outputs1.shape[0]) 
                                    outputs2 = outputs2.view(outputs2.shape[0]) 

                                    all_labels.extend(labels.detach().cpu().numpy())
                                    all_outputs1.extend(outputs1.detach().cpu().numpy())
                                    all_outputs2.extend(outputs2.detach().cpu().numpy())
                                all_labels_np =(np.array(all_labels)) 
                                metrics1=regr_metrics(all_labels_np, all_outputs1)
                                metrics2=regr_metrics(all_labels_np, all_outputs2)####################
                                print("metrics1: ",metrics1)
                                print("metrics2: ",metrics2)
                                metrics_model1.append(metrics1)
                                metrics_model2.append(metrics2)
                    avg_metrics_model1 = np.mean(metrics_model1, axis=0)
                    avg_metrics_model2 = np.mean(metrics_model2, axis=0)
 
                    ################### #################
                    # Save the averages to a CSV file
                    headers = ['Seed split',  'Model', 'Average RMSE', 'Average MAE', 'Average Std Errors', 'Average R2','Train Size', 'Test Size', 'batch_size',"Seed",'dataset',"num_epochsT"]
                    # Check if file exists, if not, write the header
                    file_exists = os.path.isfile(csv_file_path)

                    with open(csv_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(headers)  # Write header
                        writer.writerow([k, 'ResNet',avg_metrics_model1[0], avg_metrics_model1[1], avg_metrics_model1[2], avg_metrics_model1[3],train_size1, test_size1, batch_size,seed_val,num_epochsT])
                        writer.writerow([k, 'PGNN', avg_metrics_model2[0], avg_metrics_model2[1], avg_metrics_model2[2], avg_metrics_model2[3],train_size1, test_size1, batch_size, seed_val,num_epochsT])
 