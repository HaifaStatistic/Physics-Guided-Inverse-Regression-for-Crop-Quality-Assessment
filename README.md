# Physics-Guided Inverse Regression for Crop Quality Assessment

## Overview
This repository contains the implementation and data for the paper "Physics-Guided Inverse Regression for Crop Quality Assessment." The repository is organized to facilitate the reproduction of the results and to provide a clear structure for further development and experimentation.

## Repository Structure
- **data/**: Contains datasets used in the study, including both raw and processed data.
- **code/**: Includes trained models and scripts for training new models.
- **utils/**: Contains various utility scripts for data preprocessing, model training, and evaluation.
- **README.md**: This file.
- **environment.yml**: List of dependencies required to run the code.


## Abstract
We present an innovative approach leveraging Physics-Guided Neural Networks (PGNNs) for enhancing agricultural quality assessments. Central to our methodology is the application of physics-guided inverse regression, a technique that significantly improves the model's ability to precisely predict quality metrics of crops. This approach directly addresses the challenges of scalability, speed, and practicality that traditional assessment methods face. By integrating physical principles, notably Fick’s second law of diffusion, into neural network architectures, our developed PGNN model achieves a notable advancement in enhancing both the interpretability and accuracy of assessments. Empirical validation conducted on cucumbers and mushrooms demonstrates the superior capability of our model in outperforming conventional computer vision techniques in postharvest quality evaluation. This underscores our contribution as a scalable and efficient solution to the pressing demands of global food supply challenges.

## Keywords
- Physics-Guided Neural Networks (PGNNs)
- Machine Learning in Agriculture
- Agricultural Quality Assessment
- Moisture Distribution Modeling
- Inverse Regression

## Introduction
### Background and Motivation
Advancing quality assessment methods in agriculture is critical due to its impact on supply chain management, economic outcomes, and consumer health. Traditional approaches based on laboratory measurements are limited by scalability, speed, and practicality, particularly in diverse agricultural contexts. Emerging technologies in computer vision and machine learning offer rapid and non-destructive evaluation of produce quality through advanced image analysis.

### Physics-Guided Neural Networks (PGNNs)
PGNNs integrate physical knowledge within neural network architectures to guide the learning process, enhancing the model’s capability to learn and replicate physical laws effectively. This approach is particularly useful in scenarios where physical knowledge can improve accuracy and reduce reliance on extensive training data.

## Methodology
### Problem Formulation
Our methodology employs a two-step physics-guided approach to estimate laboratory measurements from image data:
1. **Physics-Guided Inverse Regression**: Transform image data into basic features encapsulating intrinsic quality attributes using physical laws.
2. **Predictive Modeling**: Use the estimated features to predict laboratory measurements.

### Integration of Physical Models
The quality attribute of agricultural produce is linked to its resultant quality metric through physical laws. By modeling these attributes with differential equations, we ensure the predictions adhere to established physical principles.

## Application
### Cucumbers
Our dataset consists of 660 RGB images of cucumbers, categorized post-storage. The models were trained to predict quality labels based on laboratory measurements. The study compares Direct Prediction and Inverse Prediction models, with results showing the superiority of the Inverse Prediction model, especially with smaller training sizes.

### Mushrooms
The dataset includes 332 images of mushrooms post-harvest and cold storage. The quality was labeled based on chroma, luminosity, and additional specific properties. Similar to cucumbers, the Inverse Prediction model outperformed the Direct Prediction model across various training sizes.

## Results
The PGNN models consistently showed superior performance in both cucumber and mushroom datasets, particularly with smaller training sets. This highlights their potential in practical scenarios with limited data availability.

## Simulation
A comprehensive simulation study was conducted to evaluate the robustness of the PGNN model under various conditions, demonstrating its adaptability and effectiveness in agricultural data analysis.

## Conclusion
This research marks a significant step towards more advanced, precise, and reliable crop quality assessment systems. The PGNN approach integrates physical laws into neural network models, enhancing the interpretability and accuracy of predictions. Future work will explore extending this framework to other types of produce and incorporating additional data types for a more comprehensive assessment.

## Setup Instructions

### Conda Environment

To set up the environment using Anaconda/Conda, follow these steps:

1. **Install Anaconda/Miniconda**: If you haven't already installed Anaconda or Miniconda, you can download and install it from [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Clone the Repository**: Clone this repository to your local machine using:
   ```sh
   git clone https://github.com/HaifaStatistic/Physics-Guided-Inverse-Regression-for-Crop-Quality-Assessment.git
   cd Physics-Guided-Inverse-Regression-for-Crop-Quality-Assessment
   ```
3. **Create the Conda Environment**: Use the environment.yml file to create the Conda environment. Run the following commands:
    ```sh
    conda env create -f environment.yml
    ```
4. **Activate the Environment**: Activate the newly created environment:
    ```sh
    conda activate crop_quality_assessment
    ```
    This will install all necessary dependencies and activate the environment, allowing you to run the project code smoothly

## Authors
David Shulman, Assaf Israeli, Yael Botnaro, Ori Margalit, Oved Tamir, Shaul Naschitz, Dan Gamrasni, Ofer M. Shir and Itai Dattner

## Contact
Dr. David Shulman: dshulman@campus.haifa.ac.il

## Citation
If you find our work useful, can cite our paper using:
```
@article{Shulman2024,
  author    = {Shulman, D. and Israeli, A. and Botnaro, Y. and others},
  title     = {Physics-Guided Inverse Regression for Crop Quality Assessment},
  journal   = {Journal of Agricultural, Biological, and Environmental Statistics (JABES)},
  year      = {2024},
  doi       = {10.1007/s13253-024-00643-9},
  url       = {https://doi.org/10.1007/s13253-024-00643-9}
}
```
## Acknowledgments

The author(s) would like to express their gratitude for the financial support received for the research, authorship, and publication of this article:

- This project is supported by Israel Science Foundation grant (1755/22)
- This project is supported by Veterinary Services and Animal Health, Israel Ministry of Agriculture and Rural Development (21-06-0012)

