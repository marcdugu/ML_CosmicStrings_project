import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gwpy.timeseries import TimeSeries as ts

def combined_plot(data: np.array, x=1, alpha=1):
    """
    data[n.parray]: 4x65536 array -> column 0-2 are E1, E2 and E3, column 3 is time
    x[int](optional): the number of the figure
    alpha[float](optional): the opacity of the plot

    Returns the figure of the average strains as function of time
    """
    plt.figure(x, figsize=(30, 10))
    plt.grid(color='grey')
    E1 = data[:, 0]
    E2 = data[:, 1]
    E3 = data[:, 2]
    average_data =  (E1 + E2 + E3)/3
    time = data[:, 3]
    t0 = time[0]
    time = time-time[0]
    plt.plot(time, average_data, alpha=alpha)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

def combined_plots_together(data: np.array, x=1, alpha=1):
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
    for arm in range(n_arms):
        plt.plot(time, E[arm], alpha=alpha)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

def individual_plots(data: np.array, x=1, alpha=1):
    """
    data[n.parray]: 4x65536 array -> column 0-2 are E1, E2 and E3, column 3 is time
    x[int](optional): the number of the figure
    alpha[floar](optional): the opacity of the plot

    Returns the figure of the combined strains as function of time
    """
    plt.figure(x, figsize=(30, 10))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    E1 = data[:, 0]
    E2 = data[:, 1]
    E3 = data[:, 2]
    time = data[:, 3]
    t0 = time[0]
    time = time-time[0]

    axes[0].plot(time, E1, label='Arm 1', color='red')
    axes[0].set_title('Strain data in arm 1')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('strain amplitude')
    axes[0].legend()

    axes[1].plot(time, E2, label='Arm 2', color = 'green')
    axes[1].set_title('Strain data in arm 2')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('strain amplitude')
    axes[1].legend()

    axes[2].plot(time, E1, label='Arm 3', color = 'orange')
    axes[2].set_title('Strain data in arm 3')
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('strain amplitude')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def gradient_plot(data: np.array, plotname: str, x=1, cmap='viridis'):

    """
    data[n.parray]: 4x65536 array -> column 0-2 are E1, E2 and E3, column 3 is time
    x[int](optional): the number of the figure
    cmap[string](optional): colors for the color map

    Returns a plot of the frequency (y-axis) and energy (colorbar) as function of time 
    """
    
    # splice the data
    t0 = data[:,3][0]
    t1 = data[:,3][1]
    start_time = t0 
    sample_rate = 1/(t1-t0)  

    # Create a GWpy TimeSeries object
    data = ts(data[:, 0], sample_rate=sample_rate, t0=start_time)

    # Generate a Q-transform spectrogram
    qspecgram = data.q_transform()

    #spectogram plot
    plot = qspecgram.plot(figsize=[8, 4])
    ax = plot.gca()

    ax.set_xscale('seconds')
    ax.set_yscale('log')
    ax.set_ylabel('Frequency [Hz]')
    ax.grid(True, axis='y', which='both')

    # Add a colorbar
    plot.colorbar(label='Normalized energy', cmap=cmap)

    # Save the figure
    plot.savefig(f'{plotname}.png') 
    print("figure has been saved")
    return



