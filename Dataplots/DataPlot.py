import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gwpy.timeseries import TimeSeries as ts

def overlap_plot(data: np.array, x=1, alpha=1):
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
    plt.legend()

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

    axes[0].plot(time, E1,color='mediumpurple')
    axes[0].set_title('Strain data in arm 1')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('strain amplitude')
    axes[0].set_xlim(0, time[-1])

    axes[1].plot(time, E2, color = 'mediumaquamarine')
    axes[1].set_title('Strain data in arm 2')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('strain amplitude')
    axes[1].set_xlim(0, time[-1])

    axes[2].plot(time, E3, color = 'cornflowerblue')
    axes[2].set_title('Strain data in arm 3')
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('strain amplitude')
    axes[2].set_xlim(0, time[-1])


    plt.tight_layout()
    plt.show()

def gradient_plot(data: np.array, arm:int, plotname: str, cmap='viridis', max_25='False', normalized='False'):

    """
    data[n.parray]: 4x65536 array -> column 0-2 are E1, E2 and E3, column 3 is time
    arm[int]: select an arm (can be 1, 2 or 3)
    plotname[string]: name of the plot
    x[int](optional): the number of the figure
    cmap[string](optional): colors for the color map
    max_25[bool](optional): change all energy values above 25 to 25
    normalized[bool](optional): plot normalized energy (normalize between 0 and 1)

    Returns a plot of the frequency (y-axis) and energy (colorbar) as function of time 
    """
    if arm not in (1, 2, 3):
        raise ValueError(f"arm must be 1, 2, or 3, but got {arm}")
    else:
        arm = arm-1
    
    # splice the data
    t0 = data[:,3][0]
    t1 = data[:,3][1]
    start_time = t0 
    sample_rate = 1/(t1-t0)  

    # Create a GWpy TimeSeries object
    data = ts(data[:, arm], sample_rate=sample_rate, t0=start_time)

    # Generate a Q-transform spectrogram
    qspecgram = data.q_transform()

    #Plot energies higher then 25 as 25
    if max_25 == True:
        qspecgram = np.minimum(qspecgram, 25)

    #Plot normalized energies (normalize between 0 and 1)
    if normalized == True:
        normalized_data = (qspecgram) / (qspecgram.max())
        qspecgram = normalized_data


    #spectogram plot
    plot = qspecgram.plot(figsize=[8, 4])
    ax = plot.gca()

    ax.set_xscale('seconds')
    ax.set_yscale('log')
    ax.set_ylabel('Frequency [Hz]')
    #set the upper frequency limit to 500Hz
    ax.set_ylim(top=500)
    ax.grid(True, axis='y', which='both')
    

    # Add a colorbar
    plot.colorbar(label='Energy', cmap=cmap)

    # Save the figure
    plot.savefig(f'{plotname}.png') 
    print(f"figure has been saved as {plotname}.png")

def final_dataplot(x, data:np.array, arm:int, deviation:str, normalized=False, plotname=None):

    if arm not in (1, 2, 3):
        raise ValueError(f"arm must be 1, 2, or 3, but got {arm}")
    else:
        arm = arm-1

    strain = data[:, arm]
    time = data[:, 3]
    t0 = time[0]
    time = time-t0
    
    strainmaxindex = np.argmax(strain)
    timemax = time[strainmaxindex]
    deviation = deviation

    if normalized:
        strain = strain/strainmaxindex

    plt.figure(x, figsize=(4, 6))
    plt.plot(time, strain, zorder=3)
    plt.grid(zorder=1, color='lavender')
    plt.xlim(timemax-deviation, timemax+deviation)
    plt.xlabel('time (s)')
    plt.ylabel('strain amplitude')

    plt.savefig(f'{plotname}.png', bbox_inches='tight') 
    print(f"figure has been saved as {plotname}.png")

    plt.show()
