import numpy as np
from scipy.signal import iirnotch, lfilter, butter, filtfilt
from preprocessing.utils import Preprocessing
from preprocessing.utils import make_a_filtered_plot_for_comparison, plot_signal_in_frequency


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y


class GitHubFilter(Preprocessing):
    def __init__(self,
                 lowcut: float = 0.5,
                 highcut: float = 50.0, #instead of 120
                 fs: int = 1024,
                 resampleFS: int = 250):
        """
        Initialize YourClass.

        Parameters
        ----------
        lowcut : float
            Lower cutoff frequency for bandpass filter (Hz).
        highcut : float
            Upper cutoff frequency for bandpass filter (Hz).
        fs : int
            Sampling frequency for original signal (Hz).
        resampleFS : int
            Target sampling frequency when resampling (Hz).
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.resampleFS = resampleFS


        notch_1_b, notch_1_a = iirnotch(1, Q=30, fs=self.resampleFS)
        self.notch_1_b = notch_1_b
        self.notch_1_a = notch_1_a

        notch_60_b, notch_60_a = iirnotch(60, Q=30, fs=self.resampleFS)
        self.notch_60_b = notch_60_b
        self.notch_60_a = notch_60_a



    def __call__(self, signals: np.ndarray) -> np.ndarray:
        # our dataset is switched with windows and example, if you compare it to the loaded dataset from github
        n_windows, n_samples = signals.shape
        filtered = np.zeros((n_windows, n_samples))
        order=3
        for i in range(n_samples):
            # 1) band-pass via SOS zero-phase
            # x_bp = sosfiltfilt(self.bp_sos, signals[i, :])
            b, a = butter(order, [self.lowcut, self.highcut], fs=self.fs, btype="bandpass")
            x_bp = filtfilt(b, a, signals[:, i])
            # 2) 50 Hz (or 1 Hz) notch
            x_n1 = lfilter(self.notch_1_b, self.notch_1_a, x_bp)
            # 3) 60 Hz (or 60 Hz) notch
            x_n2 = lfilter(self.notch_60_b, self.notch_60_a, x_n1)
            # put back
            # filtered[i  , :] = x_n2
            filtered[:, i] = x_n2

        # Nice Code to few the filtered signals
        # make_a_filtered_plot_for_comparison(signals, filtered, self.resampleFS)
        # plot_signal_in_frequency(signals[0], filtered[0], self.resampleFS)
        return filtered

