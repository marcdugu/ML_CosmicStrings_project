#########################################################################################################
#######                                                                                           #######
#######                             Neural network run code                                       #######
#######                                                                                           #######
#########################################################################################################

""" This script is for training a neural network using timeseries of GW data and a convolutional neural network (CNN) model.
It includes the following key steps:
1. Import necessary libraries: including PyTorch, CNN/CNN_utilities modules (within same folder), and other utilities for 
data handling and visualization such as numpy and matplotlib.
2. Define Neural Network Parameters:
   - `num_files_load`: Total number of files to load into the dataset.
   - `num_epochs`: Number of epochs for training the model.
   - `batch_size`: Number of samples in each batch for training.
   - `learning_rate`: Learning rate for the optimizer.
   - `weight_decay`: Regularization parameter to prevent overfitting.
3. Breakoff Parameters:
   - `min_validation_loss`: Initializing validation loss with a large value to track the minimum during training.
   - `patience`: Number of epochs with no improvement before stopping early.
   - `min_delta`: Minimum change to qualify as an improvement.
4. GPU Availability: Checks if a GPU is available and sets the device accordingly (CUDA or CPU).
5. Dataset Initialization: The dataset is created using the `CNN.signal_dataset` method.
   - Option to normalize the data with 'normalized=True'
6. Data Splitting: Divides the dataset into training, validation, and test sets with the specified sizes.
7. DataLoader Creation: 
   - Creates `DataLoader` objects for training, validation, and testing, using the specified batch sizes.
8. Running the Neural Network:
   - The `CNN.RunNeuralNetwork` method is invoked with the dataloaders and other parameters to begin training and evaluation.
   - Options for saving the training history and learning curve are included with `Save=True` and file names 'Hist' and 'Learning'
   in folder '/FinalPlots' """

#import packages
import os
import CNN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#Neural network parameters
num_files_load = 500
num_epochs = 50
batch_size = int(num_files_load/5)
learning_rate = 0.001
weight_decay = 0.0001

#breakoff parameters
min_validation_loss = float('inf') #initializing validation loss
patience = 10
min_delta = 0

#checking for GPU availabiltity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#initialize the dataset
local_root_dir_marc = r'C:\\Users\\marcd\\Desktop\\Master\\Courses\\Machine_Learning\\Project\\data\\CostmiStrings\\mock_data'
local_root_dir_bo = r'/Users/boribbens/Documents/Universiteit_Utrecht/EP_Master/Semester_1/Computational_aspects_of_Machine_Learning/ML_Project/Datafolder/mock_data'
main_root_dir = r'/Volumes/ML2024_data/GW2/Data'

dataset = CNN.signal_dataset(root_dir=main_root_dir, num_files_load = num_files_load, normalized = False)

#split data into train/validation/test
train_size = 0.7
validation_size = 0.2
test_size = 1 - (train_size + validation_size)

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

#create dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#########################################################################################################
#running the main neural network code

CNN.RunNeuralNetwork(train_loader, validation_loader, test_loader, learning_rate, weight_decay, num_epochs, patience, min_delta, Save=True, HistName='Hist', LearningName='Learning')

