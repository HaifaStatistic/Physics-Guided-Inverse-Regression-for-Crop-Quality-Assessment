import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import synthetic_data
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, RandomVerticalFlip, RandomHorizontalFlip,Normalize
from torch.optim import SGD,Adam
from torchvision.models import resnet18
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import shutil
from math import sqrt
from train import train_transform
from model2 import ImageTransformer11,DualResNet3
import subprocess
from torch.nn import MSELoss,CrossEntropyLoss,BCELoss,BCEWithLogitsLoss,Sequential,Dropout,Linear,Sigmoid,Flatten
import csv
import numpy as np
import os

machine_seed=120
csv_file_path = '/home/ec2-user/PINN_Cucumber_NEW2.csv'

def change_shape(sigmoid_weights,outputs1):
    if len(sigmoid_weights.shape) == 1:
        sigmoid_weights = sigmoid_weights.unsqueeze(1)

    # Expand weights if necessary to match the dimensions of model outputs
    if sigmoid_weights.shape[1] != outputs1.shape[1]:
        sigmoid_weights = sigmoid_weights.expand(-1, outputs1.shape[1])
    return sigmoid_weights
def regr_metrics(all_labels_np, all_outputs_np):
    rmse = sqrt(mean_squared_error(all_labels_np, all_outputs_np))
    mae = mean_absolute_error(all_labels_np, all_outputs_np)
    r2 = r2_score(all_labels_np, all_outputs_np)
    errors = all_labels_np - all_outputs_np
    std_errors = np.std(errors)
    return rmse,mae,std_errors ,r2  



# Define the augmentation
transform = Compose([
    RandomVerticalFlip(p=0.5),
    RandomHorizontalFlip(p=0.5)#,
])

class CustomDataset(Dataset):
    def __init__(self, images, labels, int_labels, transform=None):
        self.images = images
        self.labels = labels
        self.int_labels = int_labels
        self.transform=transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        int_label = self.int_labels[idx]
        return image, torch.tensor(label, dtype=torch.float), torch.tensor(int_label, dtype=torch.int64)


num_states=300
batch_size=10
all_b_circ=['b']
all_split_sizes=[  30, 45,70]
 
#try:
for b_circ in all_b_circ:
    add_rand=0
    for seed_val in range(machine_seed,machine_seed+10):
        seed_val+=add_rand
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        all_check=True
        while all_check:
            images,normalized_labels=synthetic_data.create_sinth_data(num_states,b_circ)
            # Convert normalized labels to integer labels for stratification
            int_labels = [int(label) for label in normalized_labels]

            #print("int_labels",int_labels)
            class_distribution = Counter(int_labels)

            # Check if all classes have at least 2 samples
            if all(count > 1 for count in class_distribution.values()):
                all_check=False
            else:
                add_rand+=1
          
        train_size=30


        # Create the custom dataset
        dataset = CustomDataset(images, normalized_labels, int_labels, transform)

        train_indices, test_indices = train_test_split(range(len(dataset)), train_size=train_size, stratify=dataset.int_labels, random_state=seed_val)

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        train_size1, test_size1=len(train_dataset),len(test_dataset)     
        print("train_size1, test_size1",train_size1, test_size1)
        train_loader = DataLoader(train_dataset,num_workers=0,  batch_size=batch_size,shuffle=True)#
        test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        sigma=10
        num_epochsT=18

        MT='a'

        model_transformer=ImageTransformer11(MT,sigma).to(device)

        learning_rate=0.001

        model_transformer=train_transform(model_transformer, train_loader, test_loader,num_epochsT, learning_rate)
        model_transformer.eval()

        frz=False
        mdl1='e1'
        mdl2='e2'
        model1=DualResNet3(mdl1,frz).to(device)
        model2=DualResNet3(mdl2,frz).to(device)


        # Define the criterion
        criterion = MSELoss()
        learning_rate =0.0001

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
        metrics_model3 = []
        metrics_model4 = []
        metrics_model5 = []
        metrics_model6 = []

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
            #model3.train()
            #model4.train()
            model_transformer.eval()
            for i, (images, labels,_) in enumerate(train_loader):
                # Move images and labels to the GPU if available
                images = images.to(device)
                labels = labels.to(device)

                outputs = model_transformer(images).detach()

                # Forward pass
                outputs1= model1(images,outputs,mdl1)
                outputs2,weight1= model2(images,outputs,mdl2)

                outputs1 = outputs1.view(outputs1.shape[0])  # flatten the outputs RESNET
                outputs2 = outputs2.view(outputs2.shape[0])  # flatten the outputs RESNET

                loss1 = criterion(outputs1, labels)
                loss2 = criterion(outputs2, labels)

                total_loss = loss1 + loss2


                optimizer1.zero_grad()
                optimizer2.zero_grad()

                total_loss.backward()

                optimizer1.step()
                optimizer2.step()
    
            print("epoch",epoch)
            if epoch>25:
                model1.eval()
                model2.eval()

                with torch.no_grad():
                    test_losses = []
                    all_outputs1 = []
                    all_outputs2 = []
                    all_outputs3=[]
                    all_outputs4=[]
                    all_weighted_sum = []
                    all_weighted_sum1 = []
                    all_labels = []  
                    #all_filename=[]            
                    for i, (images, labels,_) in enumerate(test_loader):
                        #labels = labels.unsqueeze(1)
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model_transformer(images).detach()

                        outputs1= model1(images,outputs,mdl1)
                        outputs2,weight1= model2(images,outputs,mdl2)###############

                        outputs1 = outputs1.view(outputs1.shape[0])  # flatten the outputs RESNET
                        outputs2 = outputs2.view(outputs2.shape[0])  # flatten the outputs RESNET

                        all_labels.extend(labels.detach().cpu().numpy())
                        all_outputs1.extend(outputs1.detach().cpu().numpy())
                        all_outputs2.extend(outputs2.detach().cpu().numpy())


                    all_labels_np =(np.array(all_labels)) 
                    metrics1=regr_metrics(all_labels_np, all_outputs1)
                    metrics2=regr_metrics(all_labels_np, all_outputs2)

                    metrics_model1.append(metrics1)
                    metrics_model2.append(metrics2)


        avg_metrics_model1 = np.mean(metrics_model1, axis=0)
        avg_metrics_model2 = np.mean(metrics_model2, axis=0)

        ################### #################
        # Save the averages to a CSV file
        headers = [  'Model', 'Average RMSE', 'Average MAE', 'Average Std Errors', 'Average R2','Train Size', 'Test Size', 'batch_size',"Seed",'dataset',"num_epochsT",'b_circ']
        # Check if file exists, if not, write the header

        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)  # Write header
            writer.writerow([ 'ResNet',avg_metrics_model1[0], avg_metrics_model1[1], avg_metrics_model1[2], avg_metrics_model1[3], train_size1, test_size1, batch_size,seed_val,approach,num_epochsT,b_circ])
            writer.writerow([ 'PGNN', avg_metrics_model2[0], avg_metrics_model2[1], avg_metrics_model2[2], avg_metrics_model2[3], train_size1, test_size1, batch_size, seed_val,approach,num_epochsT,b_circ])
