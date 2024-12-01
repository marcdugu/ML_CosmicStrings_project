import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
2. Preprocess Your Data
Normalize: Normalize the signals to have zero mean and unit variance.
Reshape: Reshape the signals to fit the input requirements of a CNN ([batch_size, 1, 65536]).
Split: Split the dataset into training, validation, and test sets.
Augmentation (Optional): If the dataset is small, consider augmenting it by adding noise, jittering, or shifting the signals slightly.
"""

class signal_dataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []        
        #Load data and labels
        self._load_data()
    
    def _load_data(self):        
        #Label-mapping
        label_map = {
            "Background" : 0,
            "Injections" : 1
            }
        #Get file
        for folder, label in label_map.items():
            folder_path = os.path.join(self.root_dir,folder,'Whitened')
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    file_path = os.path.join(folder_path, file) 
                    #Load data
                    signal_and_time_data = np.load(file_path)
                    signals_data = signal_and_time_data[:, 0:3]
                    self.data.append(signals_data)
                    self.labels.append(label)
                    
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
        

"""
3. Define the CNN Architecture
A simple CNN architecture for time series data can look like this:

Input Layer: Accepts 1D input signals of size [1, 65536].
Convolutional Layers: Use 1D convolutions (nn.Conv1d) with small kernel sizes (e.g., 3, 5, or 7) to extract features.
Pooling Layers: Use 1D max-pooling (nn.MaxPool1d) to reduce the feature map size and add translation invariance.
Fully Connected Layers: Flatten the feature maps and use dense layers to classify the signal.
Output Layer: Use a softmax or sigmoid activation depending on whether it's a binary or multi-class problem.

Conv1d:
    L_out = (( L_in + 2pad - dilation x (kernel_size - 1) - 1 ) / stride ) +1 
"""

#Calculate Lout(data size) after applying 1d convolution
def Calc_Lout_conv1d(L_in, padding, dilation, kernel_size, stride):
    return ((L_in + (2*padding) - (dilation*(kernel_size-1)) - 1) / stride) + 1

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        
        # First convolution: Combine 3 input signals into 1      
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3) 
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  
        # Second convolution: Further feature extraction
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2) 
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) 
        # Third convolution: Deeper feature extraction
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  
        # Fourth convolution: Final feature extraction
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) 
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4096, 256)  # Flatten and reduce to 256 features
        self.fc2 = nn.Linear(256, 1)  # Binary classification (0 = background, 1 = injection)

        
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
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
                
        return x


"""
4. Prepare the Dataset and DataLoader
"""
#Initialize the Dataset
root_dir = r'C:\Users\marcd\Desktop\Master\Courses\Machine_Learning\Project\data\CostmiStrings\mock_data'
dataset = signal_dataset(root_dir=root_dir) # shape(1000,2,3,65536)-->(file, signal/label, telescope, time)

#Split into Train/Test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#Create DataLoader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



"""
5. Define Loss and Optimizer
Use binary cross-entropy loss (BCELoss) for binary classification
With nn.BCEWithLogitsLoss()--> not use sigmoid in CNN
"""

#Model
model = ConvNN()
#Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #ADAM or SGD


"""
6. Train the Model
Train the model by iterating over epochs, feeding batches of data, and updating weights using backpropagation.
"""
"""
7. Evaluate the Model
Evaluate the model on the validation/test set using accuracy, precision, recall, and F1-score.
"""

# Training the CNN
num_epochs = 3
model.to(device)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for signals, labels in train_loader:
        signals, labels = signals.to(device), labels.to(device).float()  # Move to device        
        optimizer.zero_grad()  # Clear gradients
        outputs = model(signals)  # Forward pass
        loss = criterion(outputs.squeeze(), labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights        
        train_loss += loss.item()
        
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device).float()
            outputs = model(signals)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
            # Compute accuracy
            preds = torch.sigmoid(outputs).squeeze() > 0.5  # Apply sigmoid and threshold
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    val_accuracy = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            


"""
8. Fine-Tuning and Optimization
Hyperparameter Tuning: Adjust learning rate, batch size, number of layers, and filter sizes.
Regularization: Add dropout layers to prevent overfitting.
Visualization: Use tools like TensorBoard to visualize loss and accuracy trends during training.
"""


# Save the trained model
torch.save(model.state_dict(), 'telescope_signal_cnn.pth')
                    
                    
                    
                    
        
        
        
    
        
        