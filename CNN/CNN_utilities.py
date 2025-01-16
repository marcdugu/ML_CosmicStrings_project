import os
import numpy as np
import matplotlib.pyplot as plt



"""
Pythonfile where we put some utility functions such as:

"""


def EarlyStopper(validation_loss, patience, min_delta, state):

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

#Calculate Manually L_out after applying 1d convolution
def Calc_Lout_conv1d(L_in, padding, dilation, kernel_size, stride):
    return ((L_in + (2*padding) - (dilation*(kernel_size-1)) - 1) / stride) + 1

def MakePlot(epochs, train_losses, val_losses, val_accuracies, Save=False, LearningName=None):
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
    
    plt.show()

def Normalize(dataset):
    '''
    input: dataset with 3 datasets with the same length size [3,n]
    returns: normalized dataset size [3,n]
    '''
    datasetmax = max(np.max(dataset[0]), np.max(dataset[1]), np.max(dataset[2]))
    for i in range(len(dataset)):
        dataset[i] = dataset[i]/datasetmax
    
    return dataset

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

def histogram_counting(labels, predictions):

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

def histogram_plot(countlist, normalized=True, Save=False, HistName=None):

    '''
    countlist is fully made by the definition above (histogram_counting) with items:
    [label=signal and prediction=signal (good!), label=glitch and prediction=glitch (good!), 
    label=signal and prediction=glitch (wrong!), label=glitch and prediction=signal (wrong!)]
    '''
    if normalized:
        countlist = np.array(countlist)/sum(countlist)

    plt.figure()

    plt.grid("lavander", zorder=0)
    plt.bar(range(len(countlist)), countlist, color=['lawngreen', 'limegreen', 'red', 'orangered'], zorder=3) #idk why zorder 3 but oke

    plt.xticks(range(len(countlist)), ['True positive', 'True Negative', 'False Positive', 'False Negative'])
    plt.ylabel('Percentage of data')

    if Save:
        dir = "./CNN/FinalPlots"
        location = f"./CNN/FinalPlots/{HistName}.png"
        os.makedirs(dir, exist_ok=True)
        plt.savefig(location, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()