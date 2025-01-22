#########################################################################################################
#######                                                                                           #######
#######                           Neural network utility code                                     #######
#######                                                                                           #######
#########################################################################################################

#importing packages
import os
import numpy as np
import matplotlib.pyplot as plt

#########################################################################################################

def EarlyStopper(validation_loss, patience, min_delta, state):

    '''
    EarlyStopper is a function that implements early stopping for model training.
    It monitors the validation loss and stops training if the loss does not improve for a specified number of epochs (patience).
    The function returns a boolean indicating whether training should stop and updates the state dictionary.
    Parameters:
    - validation_loss: The current validation loss.
    - patience: The number of epochs to wait for improvement before stopping.
    - min_delta: The minimum change in loss to count as an improvement.
    - state: A dictionary storing the current minimum validation loss and a counter for epochs without improvement.
    Returns:
    - A tuple (stop_training, updated_state), where stop_training is a boolean and updated_state is the modified state dictionary.
    '''
    
    min_validation_loss = state.get('min_validation_loss')
    counter = state.get('counter')
    
    if validation_loss < min_validation_loss:
        min_validation_loss = validation_loss
        counter = 0
    elif validation_loss > (min_validation_loss + min_delta):
        counter += 1
        if counter >= patience:
            return True, {'min_validation_loss': min_validation_loss, 'counter': counter}
    
    #update the state
    state['min_validation_loss'] = min_validation_loss
    state['counter'] = counter
    return False, state

#########################################################################################################

def Calc_Lout_conv1d(L_in, padding, dilation, kernel_size, stride):
    '''
    Small code that calculates the output length (L_out) of a 1D convolution layer.
    '''
    return ((L_in + (2*padding) - (dilation*(kernel_size-1)) - 1) / stride) + 1

#########################################################################################################

def MakePlot(epochs, train_losses, val_losses, val_accuracies, Save=False, LearningName=None):
    
    """ 
    MakePlot is a function that generates plots to visualize the learning process during model training.
    It plots the training and validation losses over epochs, as well as the validation accuracy.
    The function can also save the generated plots to a specified location if the Save flag is set to True. 
    """

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

    if Save:
        dir = "./CNN/FinalPlots"
        location = f"./CNN/FinalPlots/{LearningName}.png"
        os.makedirs(dir, exist_ok=True)
        plt.savefig(location, dpi=300, bbox_inches='tight')
    
    #plt.show()

#########################################################################################################

def Normalize(dataset):
    '''
    Normalize is a simple function to normalize a specific dataset
    input: array (np.array) of 3 datasets with the same length size [3,n]
    returns: normalized dataset size [3,n]
    '''
    datasetmax = max(np.max(dataset[0]), np.max(dataset[1]), np.max(dataset[2]))
    for i in range(len(dataset)):
        dataset[i] = dataset[i]/datasetmax
    
    return dataset

#########################################################################################################

def overlap_plot(data: np.array, x=1, alpha=1, Save=False, Name=None):

    """
    data[n.parray]: 4x65536 array -> column 0-2 are E1, E2 and E3, column 3 is time
    x[int](optional): the number of the figure
    alpha[float](optional): the opacity of the plot

    Returns the figure of the combined strains as function of time
    """
    plt.figure(x, figsize=(30, 10))
    plt.grid(color='grey')

    E1 = data[:, 0]
    E2 = data[:, 1]
    E3 = data[:, 2]

    E = [E1, E2, E3]

    time = data[:, 3]
    t0 = time[0]
    time = time-time[0]

    n_arms = 3
    colors = ['mediumpurple', 'mediumaquamarine', 'cornflowerblue']
    for arm in range(n_arms):
        plt.plot(time, E[arm], alpha=alpha, label=f'arm {arm+1}', color=colors[arm])
    plt.xlim(0, time[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    if Save:
        dir = "./FinalPlots"
        location = f"./FinalPlots/{Name}.png"
        os.makedirs(dir, exist_ok=True)
        plt.savefig(location, dpi=300, bbox_inches='tight')

    plt.legend()

#########################################################################################################

def histogram_counting(labels, predictions):
    
    '''
    histogram_counting is a utility function to plot of how good the CNN performed. 
    Parameters:
    - Labels (the correct labels of the dataset (1 or 0)
    - Predictions (the predicted labels of the dataset (True of False)
    '''

    #running count with g (signal) and f (glitch) with the first letter corresponds to the data and the second to the prediction
    gg_count = 0
    gf_count = 0
    fg_count = 0
    ff_count = 0

    if len(labels)==len(predictions):
        for i in range(len(labels)):
            if labels[i] == 1 and predictions[i] == True:
                gg_count += 1
            elif labels[i] == 1 and predictions[i] == False:
                gf_count += 1
            elif labels[i] == 0 and predictions[i] == True:
                fg_count += 1
            elif labels[i] == 0 and predictions[i] == False:
                ff_count += 1
        
        countlist = [gg_count, ff_count, fg_count, gf_count]
        return countlist
    else:
        print("The length of the labels is not the same as the length of the predictions")
        return

#########################################################################################################

def histogram_plot(countlist, normalized=True, Save=False, HistName=None):
    

    '''
    countlist is fully made by the definition above (histogram_counting) with items:
    [label=signal and prediction=signal (good!), label=glitch and prediction=glitch (good!), 
    label=signal and prediction=glitch (wrong!), label=glitch and prediction=signal (wrong!)]
    '''
    if normalized:
        if countlist[0] != 0 and countlist[2] != 0:
            countlist_signal = np.array([countlist[0], countlist[2]])/(countlist[0] + countlist[2])
        else:
            countlist_signal = np.array([0,0])
        if countlist[1] != 0 and countlist[3] != 0:
            countlist_noise = np.array([countlist[1], countlist[3]])/(countlist[1] + countlist[3])
        else:
            countlist_noise = np.array([0,0])
        countlist = np.array([countlist_signal[0], countlist_signal[1], countlist_noise[0], countlist_noise[1]])

    plt.figure()

    plt.grid("lavander", zorder=0)
    plt.bar(range(len(countlist)), countlist, color=['limegreen', 'orangered', 'limegreen', 'orangered'], zorder=3) #idk why zorder 3 but oke

    plt.xticks(range(len(countlist)), ['True Positive', 'False Positive', 'True Negative', 'False Negative'])
    plt.ylabel('Percentage of data')

    if Save:
        dir = "./CNN/FinalPlots"
        location = f"./CNN/FinalPlots/{HistName}.png"
        os.makedirs(dir, exist_ok=True)
        plt.savefig(location, dpi=300, bbox_inches='tight')

    # Show the plot
    #plt.show()

#########################################################################################################