import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

#Hyper params
batch_size = 4

#Load data
class CosmicStrings_data():
    def __init__(self, directory, n=5000, transform=None):
        self.directory = directory #Directory where the data is stored
        self.transform = transform
        self.n = n
        #self.data = np.empty((2*n, 65536, 3), dtype=np.int32)
        self.data = []
        #self.labels = np.empty(2*n)
        self.labels = []
        self._load_data() #Adds the data to the two lists above

    def _load_data(self):
        data_dict = {
            "Background" : 0, #Background gets label 0
            "Injections" : 1  #Injections gets label 1
            }
        for folder_name, label in data_dict.items():
            folder_path = os.path.join(self.directory, folder_name, 'Whitened') #We want the whitened data
            for i, file in enumerate(os.listdir(folder_path)[:self.n]): #For testing, only get the first 50 files from each folder
                if file.endswith(".npy"): #Check if the file has the proper extension
                    file_path = os.path.join(folder_path, file) #Path to the specific file
                    full_data = np.load(file_path)
                    signal_data = full_data[:, 0:3] #First three rows are the signal data from each arm
                    #self.data[i] = signal_data #Add the data to the datalist AANPASSEN
                    self.data.append(signal_data)
                    #self.labels[i] = label   #Add the corresponding label to the datalist #AANPASSen
                    self.labels.append(label)

    #Function for checking the length of the dataset
    def len(self):
        return len(self.data) 
    def __len__(self):
        return len(self.data) 
    
    #Function for fetching the data at a specified index
    def __getitem__(self, index, asPytorchTensor=True):
        signals = self.data[index] # Shape: (65536, 3)
        label = self.labels[index] # Label: 0 (Background) or 1 (Injection)            
        if self.transform:
            signals = self.transform(signals)                
        #Convert to PyTorch tensor if necessary
        if asPytorchTensor == True: 
            signals = torch.tensor(signals.T, dtype=torch.float32) # Transpose to [3, 65536] for CNN input
            label = torch.tensor(label, dtype=torch.int)
            return signals, label
        else:
            return signals, label


#Create the CCN class
class CNN(nn.Module):
    def __init__(self, input_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3) 
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flattened_size = self._get_flattened_size(input_length)       
        self.fc1 = nn.Linear(self.flattened_size, 16)
        self.fc2 = nn.Linear(16 ,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #Flatten for fully connected layers
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]

        #Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _get_flattened_size(self, input_length):
        # Calculate the length after Conv1 + Pool1
        length_after_pool1 = (input_length + 2*3 - 7) // 1 + 1  # Conv1 output length
        length_after_pool1 = length_after_pool1 // 2  # After Pool1 (halving the length)

        # Calculate the length after Conv2 + Pool2
        length_after_pool2 = (length_after_pool1 + 2*3 - 3) // 1 + 1  # Conv2 output length
        length_after_pool2 = length_after_pool2 // 2  # After Pool2 (halving the length)

        # The flattened size will be the product of the remaining length and the output channels (64)
        flattened_size = 64 * length_after_pool2
        return flattened_size

    
def RunNeuralNetwork(dataset:CosmicStrings_data, n_train:float, n_val:float, learning_rate:float, num_epochs:int, plot=True, print_epoch=False):
    """
    dataset[CosmicStrings_data]: the data the nn has to learn
    n_train[float]: the fraction of datasamples you want to use for training
    n_val[float]: the fraction of datasamples you want to use for validation
    learning_rate[float]: how fast the nn learns
    num_epochs[int]: the number of epochs
    plot[bool](optional): allows to plot the loss for the training and validation phase
    print_epoch[bool](optional): when set to True prints the loss for each epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_size = dataset.len()                      #Dataset size
    train_size = int(data_size * n_train)                #Training set size
    val_size = int(data_size * n_val)                    #Validation set size
    test_size = int(data_size - (train_size + val_size)) #Test set size
    print(train_size, val_size, test_size)
    #Split the data randomly in training, validation and testing:
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
   
    #Create DataLoader
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)   

    #Model
    model = CNN(65536)
    #Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Training the CNN
    model.to(device)
    train_losses = []
    validation_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", ncols=100):
        #Training phase
        model.train()
        train_loss = 0.0  #Start value
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device).float()
            optimizer.zero_grad()                        #Clear gradients
            outputs = model(signals)                     #Forward pass
            loss = criterion(outputs.squeeze(), labels)  #Compute loss
            loss.backward()                              #Backward pass
            optimizer.step()                             #Update weights        
            train_loss += loss.item()

            
        #Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device).float()
                outputs = model(signals)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                #Compute accuracy
                preds = torch.sigmoid(outputs).squeeze() > 0.5  # Apply sigmoid and threshold
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        train_losses.append(train_loss)
        validation_losses.append(val_loss)

        if print_epoch == True:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # After completing all epochs, add the test phase
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

    #Compute average test loss and accuracy
    test_loss /= len(test_loader)
    test_accuracy = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    if plot == True:
        LossPlot('Training loss', train_losses, fig_num=0)
        LossPlot('Validation loss', validation_losses, fig_num=0)
        plt.show()

    #Save the trained model
    torch.save(model.state_dict(), 'telescope_signal_cnn.pth')

def LossPlot(name:str, data:list, fig_num:int):
    """
    name[string]: label name
    data[list]: x and y values for the plot
    fignum[int]: figure number
    """
    plt.figure(fig_num)
    plt.plot(data, label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    