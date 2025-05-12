from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

class Preprocessing(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

def plot_signal_in_frequency(signal, filtered_signal, sample_rate):
    # Suppose `signal` is your signal data, `filtered_signal` is the filtered data
    # signal = ...
    # filtered_signal = ...

    # Compute the frequency representation of the signals
    fft_orig = rfft(signal)
    fft_filtered = rfft(filtered_signal)

    # Compute the frequencies corresponding to the FFT output elements
    freqs = rfftfreq(len(signal), 1/sample_rate)

    # Plot the original signal in frequency domain
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(freqs, np.abs(fft_orig))
    plt.title('Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Plot the filtered signal in frequency domain
    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.abs(fft_filtered))
    plt.title('Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

def make_a_filtered_plot_for_comparison(signals, filtered_signals, thisFS):
    plt.figure()
    plt.clf()
    maximum_samples = 200
    channel_index = 5
    # if maximum_samples == -1:
    #     t = np.linspace(0, signals.shape[1]/thisFS, signals.shape[1])
    #     plt.plot(t, signals[channel_index,:], label='Noisy signal')
    #     plt.plot(t, filtered_signals[channel_index][:], label='Filtered signal')
    # else: 
    t = np.linspace(0, maximum_samples/thisFS, maximum_samples)    
    plt.plot(t, signals[:maximum_samples, channel_index], label='Noisy signal')
    plt.plot(t, filtered_signals[:maximum_samples, channel_index], label='Filtered signal')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    # plt.show()
    plt.show()
    
    return