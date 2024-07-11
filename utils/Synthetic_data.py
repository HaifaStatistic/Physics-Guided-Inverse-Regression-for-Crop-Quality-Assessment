import numpy as np
import os
from PIL import Image
import pandas as pd
import cv2

def bucket_data_by_label_range( labels):
    ranges = [(i, i+1) for i in np.arange(0, 10, 1)]
    bucketed_data = {}  
    for r in ranges:
        start, end = r
        subset_indices = np.where((labels >= start) & (labels < end))[0]
        
        bucketed_data[r] = len(subset_indices)#data[subset_indices]
        #print(bucketed_data)
        #print(subset_indices,r,len(subset_indices))
    return bucketed_data


def add_black_circles(image, num_circles=5, min_radius=3, max_radius=10):
    """
    Add random circles to the image.

    Parameters:
    - image: Input image (numpy array).
    - num_circles: Number of  circles to add.
    - min_radius: Minimum radius of the circles.
    - max_radius: Maximum radius of the circles.

    Returns:
    - Augmented image with circles.
    """
    augmented_image = image.copy()
    h, w = image.shape[:2]

    for _ in range(num_circles):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        radius = np.random.randint(min_radius, max_radius)
        cv2.circle(augmented_image, center, radius, (113, 124, 20), -1)  # -1 fills the circle  0, 0, 0

    return augmented_image

# Define the modified exponential function
def modified_exponential_func(z, A, B, alpha):
    return A * np.exp(alpha * z) + B * np.exp(-alpha * z)

def exponential_transformation(M, a, b, noise_trans=1.5):
    noisy_M = M + np.random.uniform(-noise_trans, noise_trans, size=M.shape)
    print("noisy_M,b,np.exp(b * noisy_M)",noisy_M,b,np.exp(b * noisy_M))
    return np.clip(a * np.exp(b * noisy_M), 0, 255).astype(np.uint8)

def logarithmic_transformation(M, a, b, noise_trans=15.5):
    noisy_M = M + np.random.uniform(-noise_trans, noise_trans, size=M.shape)
    return np.clip(a * np.log(b * noisy_M + 1), 0, 255).astype(np.uint8)

def power_law_transformation(M, a, gamma, noise_trans=15.5):
    noisy_M = M + np.random.uniform(-noise_trans, noise_trans, size=M.shape)
    return np.clip(a * np.power(noisy_M, gamma), 0, 255).astype(np.uint8)

def linear_transformation(M, a, b, noise_trans=15.5):
    """
    Perform a linear transformation on the moisture content to obtain color value.

    Parameters:
    - M: Moisture content (2D numpy array)
    - a: Constant multiplier for the linear transformation
    - b: Constant offset for the linear transformation

    Returns:
    - Transformed color channel (2D numpy array)
    """
    #return np.clip(a * M + b, 0, 255).astype(np.uint8)
    noisy_M = M + np.random.uniform(- 15.15, 15.15, size=M.shape)#np.random.normal(0, noise_trans, size=M.shape)
    #print("a,b,M",a,b,M)
    return np.clip(a * noisy_M + b, 0, 255).astype(np.uint8)

# Create synthetic images
def create_synt_data(num_synthetic_images,output_folder,seed_val,noise_trans,b_circ,spr):
    a_red = 0.5
    b_red = 30
    np.random.seed(seed_val)
    a_green = -0.8
    b_green = 220

    a_blue = -0.3
    b_blue = 90
    '''
    a_red = 70
    b_red = 0.03

    a_green = -70
    b_green = 0.2

    a_blue = -40
    b_blue = 0.1
    '''
    noise_std = 0.0051  # Standard deviation of the noise (adjust as needed)
    #noise_std_meas=0.002

    overall_A = [43.79071945216161, 18.69489910517041]
    overall_B = [69.34399028617138, 40.69081891623643]
    overall_alpha = [0.07933347960757842, 0.04965241306927144]
    mean_A,std_A=overall_A
    mean_B,std_B=overall_B
    mean_alpha,std_alpha=overall_alpha
    #print("synth,seed_val",seed_val)
    image_shape = (110, 350)  # Shape of the synthetic images
    z_values = np.linspace(0, 10, image_shape[1])
    quality_labels = []
    for i in range(num_synthetic_images):
        rand_su=0
        A_sample_overall = np.random.uniform(mean_A - std_A, mean_A + std_A)
        B_sample_overall = np.random.uniform(mean_B - std_B, mean_B + std_B)
        alpha_sample_overall = np.random.uniform(mean_alpha - std_alpha, mean_alpha + std_alpha)
        
        moisture_contents=[]
        # Initialize the synthetic RGB image
        synthetic_rgb_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        
        for j in range(image_shape[0]):
            # Add noise to the overall values
            rand_su+= np.random.normal(0, noise_std)#np.random.uniform(- 0.015, 0.015)#
            A_noisy = A_sample_overall + rand_su#np.random.normal(0, noise_std)
            B_noisy = B_sample_overall + rand_su#np.random.normal(0, noise_std)
            alpha_noisy = alpha_sample_overall + rand_su#np.random.normal(0, noise_std)
            
            moisture_content = modified_exponential_func(z_values, A_noisy, B_noisy, alpha_noisy)#/(4*10**2)
            moisture_contents.append(moisture_content)

            # Transform moisture content to RGB channels  linear_transformation exponential_transformation   logarithmic_transformation
            synthetic_rgb_image[j, :, 0] = linear_transformation(moisture_content, a_red, b_red,noise_trans)
            synthetic_rgb_image[j, :, 1] = linear_transformation(moisture_content, a_green, b_green,noise_trans)
            synthetic_rgb_image[j, :, 2] = linear_transformation(moisture_content, a_blue, b_blue,noise_trans)
        # Add black circles to the synthetic image
        if b_circ=='a':
            num_circles,min_radius,max_radius=(10,5,8)
        elif b_circ=='b':            
            num_circles,min_radius,max_radius=(20,5,8)
        elif b_circ=='c':    
            num_circles,min_radius,max_radius=(30,5,8)
        elif b_circ=='d':    
            num_circles,min_radius,max_radius=(40,5,8)
        #print('b_circ',b_circ)
        #print("num_circles,min_radius,max_radius",num_circles,min_radius,max_radius)
        synthetic_rgb_image = add_black_circles(synthetic_rgb_image, num_circles=num_circles, min_radius=min_radius, max_radius=max_radius)
        # Calculate the quality label for the synthetic image
        moisture_contents=np.array(moisture_contents)
        quality_label = np.sum(moisture_contents)/(4*10**6) #+ np.random.normal(0, noise_std_meas)
        quality_labels.append(quality_label)    
        # Save the synthetic RGB image
        image_filename = os.path.join(output_folder, f"synth_{i+1}.png")
        Image.fromarray(synthetic_rgb_image).save(image_filename)
    #spr=0.5
    # Normalize the quality labels to the range [1, 10]
    #print('num_circles,min_radius,max_radius',num_circles,min_radius,max_radius)
    min_label = min(quality_labels)
    max_label = max(quality_labels)
    normalized_labels1 =  ((np.array(quality_labels) - min_label) / (max_label - min_label))*9 
    normalized_labels=normalized_labels1 + np.random.normal(0, 0.4, size=normalized_labels1.shape)   #normal   uniform
    #print("normalized_labels1.shape",normalized_labels1.shape)
    #print("spr",spr)
    #print("np.mean(normalized_labels1-normalized_labels)",np.mean(normalized_labels1-normalized_labels))
    #print(f"Synthetic RGB images saved to {output_folder}")
    # Save quality labels to a CSV file
    #print('num_synthetic_images',num_synthetic_images)
    #print('normalized_labels', len(normalized_labels))
    labels_df = pd.DataFrame({'Image_Name': [f"synth_{i+1}" for i in range(num_synthetic_images)],
                              'Quality_Label': normalized_labels})
    #labels_df.to_csv('C:\\Cucumbers_quality\\Physic_inform_NN\\PINN_combined_Monte_Carlo\\cuted_ra_3C_syn\\quality_labels.csv', index=False)
    return labels_df,num_circles,min_radius,max_radius
'''
print(f"Quality labels saved to {output_folder}/quality_labels.csv")

bucketed_data = bucket_data_by_label_range(labels_df.Quality_Label)
print("bucketed_data1",bucketed_data)
'''