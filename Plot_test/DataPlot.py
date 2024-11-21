import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def combined_plot(data: np.array, x=1, alpha=1):
    """
    data[n.parray]: 4x65536 array -> column 0-2 are E1, E2 and E3, column 3 is time
    x[int](optional): the number of the figure
    alpha[floar](optional): the opacity of the plot

    Returns the figure of the combined strains as function of time
    """
    plt.figure(x, figsize=(30, 10))
    plt.grid(color='lavender')
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

def gradient_plot(data: np.array, x=1, cmap='viridis'):
    """
    data[n.parray]: 4x65536 array -> column 0-2 are E1, E2 and E3, column 3 is time
    x[int](optional): the number of the figure
    cmap[string](optional): colors for the color map

    Returns a plot of the frequency (y-axis) and energy (colorbar) as function of time
    """
    plt.figure(x, figsize=(30, 10))
    E1 = data[:, 0]
    E2 = data[:, 1]
    E3 = data[:, 2]
    average_data =  (E1 + E2 + E3)/3
    time = data[:, 3]
    delta_t = time[1]-time[0]
    f, t, Zxx = signal.stft(average_data, fs=delta_t, nperseg=256)
    t = np.arange(0, (time[-1]-time[0]), (time[-1]-time[0])/len(t))
    magnitude = np.abs(Zxx)
    energy = magnitude*magnitude
    plt.pcolormesh(t, f, energy, shading='auto', cmap=cmap)
    plt.colorbar(label='Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')