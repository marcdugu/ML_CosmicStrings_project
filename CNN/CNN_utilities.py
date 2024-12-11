import matplotlib.pyplot as plt

"""
Pythonfile where we put some utility functions such as:

"""

def EarlyStopper(validation_loss, patience, min_delta):

    min_validation_loss = float('inf')
    if validation_loss < min_validation_loss:
        min_validation_loss = validation_loss
        counter = 0
    elif validation_loss > (min_validation_loss + min_delta):
        counter += 1
        if counter >= patience:
            return True
    return False

#Calculate Manually L_out after applying 1d convolution
def Calc_Lout_conv1d(L_in, padding, dilation, kernel_size, stride):
    return ((L_in + (2*padding) - (dilation*(kernel_size-1)) - 1) / stride) + 1


def MakePlot(epochs, train_losses, val_losses, val_accuracies):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # Plot Train and Validation Loss
    axs[0].plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    axs[0].plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train and Validation Loss')
    axs[0].legend()
    axs[0].grid(True)
    # Plot Validation Accuracy
    axs[1].plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', marker='o', color='green')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_title('Validation Accuracy')
    axs[1].legend()
    axs[1].grid(True)
    # Show the plot
    plt.tight_layout()
    plt.show()