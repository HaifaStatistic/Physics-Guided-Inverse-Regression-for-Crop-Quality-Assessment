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
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
boto3.client('s3', region_name='us-west-2')
boto3.client('ec2', region_name='us-west-2')
'''
with open('number_seed_new_4.txt') as f:
    number_str = f.read().strip()
    machine_seed=int(number_str)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
machine_seed=120

csv_file_path = '/home/ec2-user/PINN_Cucumber_NEW2.csv'
def upload_file_to_s3(file_name, bucket, final_tourn=False):
    object_name = 'Simulation/1/PINN__'+str(machine_seed)+'_.csv'#'+str(count_turns)+'
    if final_tourn:
        object_name = 'Simulation/1/finish/PINN__'+str(machine_seed)+'_.csv'
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False
    return True
def scheduled_upload(csv_file_path,final_tourn=False):
    print("Uploading file...")
    file_name = csv_file_path # Replace with the path to your file
    bucket = 'simulationdavidshulman'  # Your S3 bucket
    success = upload_file_to_s3(file_name, bucket,approach,final_tourn)
    if success:
        print(f"File {file_name} uploaded successfully.")
    else:
        print("File upload failed.")
def change_shape(sigmoid_weights,outputs1):
    if len(sigmoid_weights.shape) == 1:
        sigmoid_weights = sigmoid_weights.unsqueeze(1)

    # Expand weights if necessary to match the dimensions of model outputs
    if sigmoid_weights.shape[1] != outputs1.shape[1]:
        sigmoid_weights = sigmoid_weights.expand(-1, outputs1.shape[1])
    return sigmoid_weights
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



# Define the augmentation
transform = Compose([
    #Resize((110,350)),  # Resize all images to the same size
    RandomVerticalFlip(p=0.5),
    RandomHorizontalFlip(p=0.5)#,
    #ToTensor()#,  # Convert images to PyTorch tensors
    #Normalize(mean=mean, std=std)
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
all_b_circ=['b']#'a','b','c','d'
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
                #print("Stratification not possible due to class with fewer than 2 samples.")
              
        train_size=30
        #for train_size in all_split_sizes:
        '''
        # Display the count for each unique value
        for value, count in class_distribution.items():
            print(f"Value {value} appears in train_dataset {count} times")
        '''

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

        #print('model_T: ',model_T)
        #print('num_epochsT: ',num_epochsT)
        #print('sigma: ', sigma)
        #print('sm',sm)
        MT='a'

        model_transformer=ImageTransformer11(MT,sigma).to(device)

        learning_rate=0.001

        model_transformer=train_transform(model_transformer, train_loader, test_loader,num_epochsT, learning_rate)
        model_transformer.eval()

        frz=False
        mdl1='e1'
        mdl2='e2'
        #mdl3='c'
        #mdl4='b6'
        model1=DualResNet3(mdl1,frz).to(device)
        model2=DualResNet3(mdl2,frz).to(device)
        #model3=DualResNet3(mdl3,frz).to(device)
        #model4=DualResNet3(mdl4,frz).to(device)


        # Define the criterion
        criterion = MSELoss()
        learning_rate =0.0001
        #print("learning_rate ",learning_rate)
        # Define the optimizer
        optimizer1 = Adam(model1.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        optimizer2 = Adam(model2.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        #optimizer3 = Adam(model3.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        #optimizer4 = Adam(model4.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)


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

                #print("images.shape, labels.shape",images.shape, labels.shape)
                # Forward pass
                outputs1= model1(images,outputs,mdl1)
                outputs2,weight1= model2(images,outputs,mdl2)
                #print("weight1",weight1)
                #outputs3= model3(images,outputs,mdl3)
                #outputs4= model4(images,outputs,mdl4)
                # Apply sigmoid to the weights
                # Calculate the weighting factor b
                sigmoid_weights = (1 / (1 + weight1))#.unsqueeze(1)

                # Compute the weighted average of the predictions
                

                sigmoid_weights1 = torch.sigmoid(weight1)
                #print("labels.shape,outputs1.shape, outputs2.shape,weight1.shape,sigmoid_weights.shape",labels.shape,outputs1.shape, outputs2.shape,weight1.shape,sigmoid_weights.shape)

                sigmoid_weights=change_shape(sigmoid_weights,outputs1)
                sigmoid_weights1=change_shape(sigmoid_weights1,outputs1)
                # Compute the weighted sum of the predictions
                
                #(1 / (1 + weight1))
                #weighted_average = pred1 * (1 - b) + pred2 * b
                weighted_sum = outputs1 * (1 - sigmoid_weights) + outputs2 * sigmoid_weights

                #sigmoid:
                #weighted_average = pred1 * b + pred2 * (1 - b)
                weighted_sum1 = outputs1* sigmoid_weights1  + outputs2 * (1 - sigmoid_weights1)
                #outputs = (outputs1 + outputs2) / 2
                outputs1 = outputs1.view(outputs1.shape[0])  # flatten the outputs RESNET
                outputs2 = outputs2.view(outputs2.shape[0])  # flatten the outputs RESNET
                #outputs3 = outputs3.view(outputs3.shape[0])  # flatten the outputs RESNET
                #outputs4 = outputs4.view(outputs4.shape[0])  # flatten the outputs RESNET
                
                '''
                print("labels.shape,outputs1, outputs2,sigmoid_weights.shape",labels.shape,outputs1.shape, outputs2.shape,sigmoid_weights.shape)
                print("weighted_sum",weighted_sum)
                print("sigmoid_weights",sigmoid_weights)

                print("outputs1",outputs1)
                print("outputs2",outputs2)
                '''
                loss1 = criterion(outputs1, labels)
                loss2 = criterion(outputs2, labels)
                #loss3 = criterion(outputs3, labels)
                #loss4 = criterion(outputs4, labels)
                
                total_loss = loss1 + loss2# +  loss4

                # Backward and optimize
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                #optimizer3.zero_grad()
                #optimizer4.zero_grad()

                total_loss.backward()
                '''
                loss1.backward()
                loss2.backward()
                loss3.backward()
                '''
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer1.step()
                optimizer2.step()
                #optimizer3.step()
                #optimizer4.step()
                '''
                train_losses += loss1.item()
                train_losses2 += loss2.item()
                train_losses3 += loss3.item()
                num_batches += 1      
                '''  
            # Calculate the average loss for this epoch and add to list
            '''
            avg_loss = train_losses / num_batches
            avg_loss2 = train_losses2 / num_batches
            avg_loss3 = train_losses3 / num_batches
            
            print("avg_loss1 Train",avg_loss)
            print("avg_loss2 Train",avg_loss2)
            print("avg_loss3 Train",avg_loss3)
            
            epoch_losses.append(avg_loss)   
            '''    
            print("epoch",epoch)
            if epoch>25:
                model1.eval()
                model2.eval()
                #model3.eval()
                #model4.eval()
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
                        #outputs3= model3(images,outputs,mdl3)####################
                        #outputs4= model4(images,outputs,mdl4)####################
                        # Apply sigmoid to the weights
                        ###################################
                        sigmoid_weights = (1 / (1 + weight1))#.unsqueeze(1)

                        # Compute the weighted average of the predictions
                        

                        sigmoid_weights1 = torch.sigmoid(weight1)
                        #print("labels.shape,outputs1.shape, outputs2.shape,weight1.shape,sigmoid_weights.shape",labels.shape,outputs1.shape, outputs2.shape,weight1.shape,sigmoid_weights.shape)

                        sigmoid_weights=change_shape(sigmoid_weights,outputs1)
                        sigmoid_weights1=change_shape(sigmoid_weights1,outputs1)
                        # Compute the weighted sum of the predictions
                        
                        #(1 / (1 + weight1))
                        #weighted_average = pred1 * (1 - b) + pred2 * b
                        weighted_sum = outputs1 * (1 - sigmoid_weights) + outputs2 * sigmoid_weights

                        #sigmoid:
                        #weighted_average = pred1 * b + pred2 * (1 - b)
                        weighted_sum1 = outputs1* sigmoid_weights1  + outputs2 * (1 - sigmoid_weights1)
                        ##################################
                        #outputs = (outputs1 + outputs2) / 2
                        outputs1 = outputs1.view(outputs1.shape[0])  # flatten the outputs RESNET
                        outputs2 = outputs2.view(outputs2.shape[0])  # flatten the outputs RESNET
                        weighted_sum= weighted_sum.view(weighted_sum.shape[0])
                        weighted_sum1= weighted_sum1.view(weighted_sum1.shape[0])
                        #outputs3 = outputs3.view(outputs3.shape[0])  # flatten the outputs RESNET
                        #outputs4 = outputs4.view(outputs4.shape[0])  # flatten the outputs RESNET

                        all_labels.extend(labels.detach().cpu().numpy())
                        all_outputs1.extend(outputs1.detach().cpu().numpy())
                        all_outputs2.extend(outputs2.detach().cpu().numpy())
                        #all_outputs3.extend(outputs3.detach().cpu().numpy())
                        #all_outputs4.extend(outputs4.detach().cpu().numpy())
                        all_weighted_sum.extend(weighted_sum.detach().cpu().numpy())
                        all_weighted_sum1.extend(weighted_sum1.detach().cpu().numpy())


                        #all_outputs.extend(outputs.detach().cpu().numpy())

                    #if epoch>25:# or 1==1:
                    # Append metrics for each model (you'll need to modify this based on how you get outputs for each model)
                    all_labels_np =(np.array(all_labels)) 
                    metrics1=regr_metrics(all_labels_np, all_outputs1)
                    metrics2=regr_metrics(all_labels_np, all_outputs2)####################
                    metrics3=regr_metrics(all_labels_np, all_weighted_sum1)
                    #metrics4=regr_metrics(all_labels_np, all_outputs3)##################
                    metrics5=regr_metrics(all_labels_np, all_weighted_sum)
                    #metrics6=regr_metrics(all_labels_np, all_outputs4)
                    ''''''
                    print("Epoch",epoch)
                    print("rmse,mae,std_errors ,r2 ")
                    print("metrics1 e1",metrics1)
                    print("metrics2 e2",metrics2)
                    print("metrics3 sigmoid_sepate",metrics3)
                    #print("metrics4 c",metrics4)
                    print("metrics5 b5_separate",metrics5)
                    #print("metrics6 b6",metrics6)
                    
                    metrics_model1.append(metrics1)
                    metrics_model2.append(metrics2)
                    metrics_model3.append(metrics3)
                    #metrics_model4.append(metrics4)
                    metrics_model5.append(metrics5)
                    #metrics_model6.append(metrics6)
                    '''
                    all_outputs_np = (np.array(all_outputs))                    #np.expm1
                    all_labels_np =(np.array(all_labels))                       #np.expm1
                    mse,rmse,mae,r2,std_errors=regr_metrics(all_labels_np, all_outputs_np)
                    average_RMSE_loss.append(rmse)
                    average_R_Sq.append(r2)
                    all_mae_test.append(mae)
                    std_errors_test.append( std_errors)
                    
                    print(metrics_model1)
                    print(metrics_model2)
                    print(metrics_model3)
                    '''
            #print(f'Test: Epoch {epoch+1}, MSE: {mse:.4f},RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')     
            # Calculate average metrics for each model
        avg_metrics_model1 = np.mean(metrics_model1, axis=0)
        avg_metrics_model2 = np.mean(metrics_model2, axis=0)
        avg_metrics_model3 = np.mean(metrics_model3, axis=0)
        #avg_metrics_model4 = np.mean(metrics_model4, axis=0)
        avg_metrics_model5 = np.mean(metrics_model5, axis=0)
        #avg_metrics_model6 = np.mean(metrics_model6, axis=0)
        #print("avg_metrics_model1",avg_metrics_model1)
        #print("avg_metrics_model4",avg_metrics_model4)
        ################### #################
        # Save the averages to a CSV file
        headers = [  'Model', 'Average RMSE', 'Average MAE', 'Average Std Errors', 'Average R2','Train Size', 'Test Size', 'batch_size',"Seed",'dataset',"num_epochsT",'b_circ']
        # Check if file exists, if not, write the header

        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)  # Write header
            writer.writerow([ 'e1',avg_metrics_model1[0], avg_metrics_model1[1], avg_metrics_model1[2], avg_metrics_model1[3], train_size1, test_size1, batch_size,seed_val,approach,num_epochsT,b_circ])
            writer.writerow([ 'e2', avg_metrics_model2[0], avg_metrics_model2[1], avg_metrics_model2[2], avg_metrics_model2[3], train_size1, test_size1, batch_size, seed_val,approach,num_epochsT,b_circ])
            writer.writerow([ 'sigmoid_sepate', avg_metrics_model3[0], avg_metrics_model3[1], avg_metrics_model3[2], avg_metrics_model3[3], train_size1, test_size1, batch_size, seed_val,approach,num_epochsT,b_circ])
            #writer.writerow([ 'c', avg_metrics_model4[0], avg_metrics_model4[1], avg_metrics_model4[2], avg_metrics_model4[3], train_size1, test_size1, batch_size, seed_val,approach,num_epochsT,b_circ])
            writer.writerow([ 'b5_separate', avg_metrics_model5[0], avg_metrics_model5[1], avg_metrics_model5[2], avg_metrics_model5[3], train_size1, test_size1, batch_size, seed_val,approach,num_epochsT,b_circ])
            #writer.writerow([ 'b6', avg_metrics_model6[0], avg_metrics_model6[1], avg_metrics_model6[2], avg_metrics_model6[3], train_size1, test_size1, batch_size, seed_val,approach,num_epochsT,b_circ])
        scheduled_upload(csv_file_path,final_tourn=False)       



'''                
except:
    print("Something else went wrong")            
'''
'''
final_tourn=True
scheduled_upload(csv_file_path,approach,final_tourn=True)

except IndexError:
    print(f"No label found for file: {image_path}")

'''
'''
def terminate_instance():
    # Get the current instance ID
    instance_id = subprocess.getoutput("curl -s http://169.254.169.254/latest/meta-data/instance-id")
    
    # Initialize boto3 EC2 client
    ec2_client = boto3.client('ec2', region_name='us-west-2')

    # Terminate the instance
    try:
        ec2_client.terminate_instances(InstanceIds=[instance_id])
        print(f"Instance {instance_id} is scheduled for termination.")
    except Exception as e:
        print(f"Error terminating instance: {e}")

# Call the function at the end of your script
terminate_instance()
'''