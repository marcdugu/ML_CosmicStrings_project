#########################################################################################################
#######                                                                                           #######
#######                             Neural network main code                                      #######
#######                                                                                           #######
#########################################################################################################

#importing the packages
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import CNN_utilities as Utils
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
"""
Preprocess Your Data
Normalize: Normalize the signals to have zero mean and unit variance.
Reshape: Reshape the signals to fit the input requirements of a CNN ([batch_size, 1, 65536]).
Split: Split the dataset into training, validation, and test sets.
Augmentation (Optional): If the dataset is small, consider augmenting it by adding noise, jittering, or shifting the signals slightly.
"""

class signal_dataset(Dataset):
    
    def __init__(self, root_dir, num_files_load=5000, transform=None, normalized=False):
        self.root_dir = root_dir
        self.num_files_load = num_files_load
        self.transform = transform
        self.normalized = normalized
        self.data = []
        self.labels = []        
        #Load data and labels
        self._load_data()
    
    def _load_data(self):
        #Label-mapping
        label_map = {
            "Background": 0,
            "Injections": 1
        }
        #Get file
        for folder, label in label_map.items():
            folder_path = os.path.join(self.root_dir, folder, 'Whitened')
            file_count = 0
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    if file_count >= self.num_files_load:  
                        break
                    file_path = os.path.join(folder_path, file)
                    #Load data
                    signal_and_time_data = np.load(file_path)
                    signals_data = signal_and_time_data[:, 0:3]
                    if self.normalized:
                        signals_data = Utils.Normalize(signals_data)
                        self.data.append(signals_data)
                    else:
                        self.data.append(signals_data)
                    self.labels.append(label)
                    file_count += 1  

                    
    def __len__(self):
        return len(self.data) 
        
    def __getitem__(self, index):
        signals = self.data[index] # Shape: (65536, 3)
        label = self.labels[index] # Label: 0 (Background) or 1 (Injection)            
        if self.transform:
            signals = self.transform(signals)                
        #Convert to PyTorch tensor
        signals = torch.tensor(signals.T, dtype=torch.float32) # Transpose to [3, 65536] for CNN input
        label = torch.tensor(label, dtype=torch.int)
        return signals, label
        
#%% CNN Architecture
"""
Define the CNN Architecture
A simple CNN architecture for time series:
Input Layer: Accepts 1D input signals of size [1, 65536].
Convolutional Layers: Use 1D convolutions (nn.Conv1d) with small kernel sizes (e.g., 3, 5, or 7) to extract features.
Pooling Layers: Use 1D max-pooling (nn.MaxPool1d) to reduce the feature map size and add translation invariance.
Fully Connected Layers: Flatten the feature maps and use dense layers to classify the signal.
Output Layer: Use a softmax or sigmoid activation depending on whether it's a binary or multi-class problem.
"""

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        
        # First convolution: Combine 3 input signals and feature extraction
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=11, stride=1, padding=3) 
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  
        # Second convolution: feature extraction
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=2) 
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) 
        # Third convolution: feature extraction
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)  
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  
        # Fourth convolution: feature extraction
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)  
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) 
        # Fifth convolution: feature extraction
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1)  
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2) 
        # Sixth convolution: feature extraction
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)  
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2) 
        # Seventh convolution: feature extraction
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)  
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16, 256)  # Flatten and reduce to 256 features

        #more layers
        self.fc2 = nn.Linear(256, 64)  # Flatten and reduce to 64 features

        self.fc3 = nn.Linear(64, 1)  # Binary classification (0 = background, 1 = injection)
       

        
    def forward(self, x):
        # Pass through convolutional layers + ReLU + pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)        
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        x = F.relu(self.conv7(x))
         
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)       
        
                
        return x

#%% Dummy array

#DummyArray to study the shape transformation of the data though the CNN
#Make sure it is equal to the real CNN

#make subclass to CNN class (NEEDS TWEEKING):
class DummyCNN(ConvNN):
    def __init__(self):
        super(DummyCNN, self).__init__()
 
    def forward(self, x):
        print("Input shape:", x.shape)  
        
        x = F.relu(self.conv1(x))
        print("Conv1:", x.shape)
        x = self.pool1(x)
        print("Pool1:", x.shape)
        
        x = F.relu(self.conv2(x))
        print("Conv2:", x.shape)
        x = self.pool2(x)
        print("Pool2:", x.shape)
        
        x = F.relu(self.conv3(x))
        print("Conv3:", x.shape)
        x = self.pool3(x)
        print("Pool3:", x.shape)
        
        x = F.relu(self.conv4(x))
        print("Conv4:", x.shape)
        x = self.pool4(x)
        print("Pool4:", x.shape)
        
        x = F.relu(self.conv5(x))
        print("Conv5:", x.shape)
        x = self.pool5(x)
        print("Pool5:", x.shape)
        
        x = F.relu(self.conv6(x))
        print("Conv6:", x.shape)
        x = self.pool6(x)
        print("Pool6:", x.shape)         
        
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]
        print("Flatten:", x.shape)
                
        return x


#%% Train and Validation
"""
Train the model 
Evaluate the model on the validation/test set using accuracy, precision, recall, and F1-score.
"""

def RunNeuralNetwork(train_loader:DataLoader, validation_loader:DataLoader, test_loader:DataLoader, learning_rate:float, 
                     weight_decay:float, num_epochs:int, patience:int, min_delta:int, Save=False, HistName=None, LearningName=None):
    
    #COMMENTS
    #Model
    model = ConvNN()
    #Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #ADAM or SGD

    model.to(device)
    epochs = 0
    state = {'min_validation_loss': float('inf'), 'counter': 0}

    train_losses = []
    val_losses = []
    val_accuracies = []


    for epoch_ in tqdm(range(num_epochs), desc="Training Epochs", ncols=100):
        # Training phase
        model.train()
        train_loss = 0.0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device).float()  # Move to device        
            optimizer.zero_grad()                                            # Clear gradients
            outputs = model(signals)                                         # Forward pass
            loss = criterion(outputs.squeeze(), labels)                      # Compute loss
            loss.backward()                                                  # Backward pass
            optimizer.step()                                                 # Update weights        
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels in validation_loader:
                signals, labels = signals.to(device), labels.to(device).float()
                outputs = model(signals)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                # Compute accuracy
                preds = torch.sigmoid(outputs).squeeze() > 0.5  
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(validation_loader)
        val_accuracy = correct / total
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        epochs += 1
        
        #Early Stopper
        stop, state = Utils.EarlyStopper(val_loss, patience, min_delta, state)
        if stop:         
            print(f'Early Stop at Epoch: {epochs}')
            break
    
    #After completing all epochs, add the test phase
    print("\nEvaluating on Test Data...")
    model.eval()  #Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device).float()
            outputs = model(signals)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
            
            #Compute test accuracy
            preds = torch.sigmoid(outputs).squeeze() > 0.5  #Apply sigmoid and threshold
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        labels_array = labels.cpu().numpy()      # Cannot convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        preds_array = preds.cpu().numpy()

    countlist = Utils.histogram_counting(labels_array, preds_array)
    Utils.histogram_plot(countlist, Save=Save, HistName=HistName)


    #Compute average test loss and accuracy
    test_loss /= len(test_loader)
    test_accuracy = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

                
    # Save the trained model
    torch.save(model.state_dict(), './CNN/GW_signal_cnn.pth')

    #make plot
    Utils.MakePlot(epochs, train_losses, val_losses, val_accuracies, Save=Save, LearningName=LearningName)


                
'''
It learns to fast. Learning rate?
Look for Early Stopping

Recommendations
To fine-tune and optimize the model, consider the following approaches:

1. Hyperparameter Tuning
Learning Rate: Experiment with different learning rates to find the best value for convergence. Use a learning rate scheduler to reduce the learning rate as training progresses.
Batch Size: Try smaller or larger batch sizes to see how it affects training dynamics.
Epochs: Reduce the number of epochs to avoid overfitting since validation loss starts to increase after 4 epochs.
Optimizer: If you're using Adam, you might want to try SGD with momentum or other optimizers to observe their effects.
2. Regularization
Dropout: Add dropout layers between fully connected layers to prevent overfitting. Start with dropout rates like 0.2 or 0.3.
Weight Decay (L2 Regularization): Add a weight decay parameter to your optimizer to penalize large weights.
3. Data Augmentation
Augment your training data if the dataset size is small. For time-series data, you can:
Add noise.
Apply time-shifting or scaling.
Perform random cropping.
4. Early Stopping
Use early stopping to halt training when the validation loss stops improving. This will prevent overfitting and save training time.
'''
                   
"""
Fine-Tuning and Optimization
Hyperparameter Tuning: Adjust learning rate, batch size, number of layers, and filter sizes.
Regularization: Add dropout layers to prevent overfitting.
Visualization: Use tools like TensorBoard to visualize loss and accuracy trends during training.
""" 

# %%
