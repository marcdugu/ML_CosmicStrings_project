# Project GW2: Cosmic strings
## The challenge
Cosmic strings are hypothetical one-dimensional objects that may have formed during phase transitions in the early Universe. Similar to vortex lines in liquid helium, cosmic strings are theorized to be remnants of the rapid cooling and expansion that occurred shortly after the Big Bang. Despite their theoretical significance, cosmic strings have yet to be detected, and their impact on cosmological observations remains unclear. Observations from COBE and WMAP satellites have demonstrated that cosmic strings do not make a measurable contribution to the cosmic microwave background (CMB), which limits one potential avenue for their detection. Another promising avenue for observing cosmic strings is through their potential to generate gravitational wave bursts. However, detecting these bursts is challenging due to the presence of "blip glitches" in detector data. Blip glitches are short-lived, transient artifacts that appear in gravitational wave detectors due to unknown causes, closely resembling the expected signals from cosmic strings. This resemblance makes it difficult to confidently differentiate between actual cosmic string signals and noise artifacts. As an example, in the graphic above we show a cosmic string (left) and a blip glitch (right) embedded in the Einstein Telescope detector noise, in time domain and time-frequency representation. Furthermore, without a precise model for cosmic string waveforms, it is challenging to develop reliable detection methods using current algorithms. Consequently, existing detection techniques are unable to robustly distinguish cosmic string signals from background noise, which impedes progress in validating the existence of cosmic strings.

## The project 
In response to these challenges, this project proposes the use of machine learning, specifically convolutional neural networks (CNNs), to enhance the detection of cosmic strings in gravitational wave data. CNNs are highly effective in pattern recognition and classification tasks, making them a suitable tool for differentiating between similar-looking data classes, such as cosmic string signals and blip glitches. By training CNNs on a dataset of simulated cosmic string waveforms alongside modeled blip glitches, we aim to develop analgorithm capable of distinguishing these two classes in detector data.

This approach involves generating a set of approximate cosmic string waveforms and crafting realistic blip glitch models to simulate the types of noise found in gravitational wave detectors. The CNN will be trained using this dataset to recognize the subtle differences between glitches and true cosmic string events. In particular, this project focuses on developing and testing this methodology within the context of the Einstein Telescope, a proposed third-generation gravitational wave observatory, which promises to advance our understanding of the early Universe and the potential role of cosmic strings in cosmology.

## Plan
Have a small neural network running on time series data

## Questions


## Plots
-Folding figure
-Data figure
-Loss plot/accuracy figure

