import numpy as np
from scipy import signal
from preprocessing.utils import Preprocessing
from preprocessing.preprocessing_github import GitHubFilter

class FFTFilter(Preprocessing):
    def __init__(self, filter_order, low_freq, high_freq, fs):
        self.bp_filter = signal.butter(filter_order, (low_freq, high_freq),
                                       btype="bandpass", output="sos", fs=fs)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_filtered = signal.sosfiltfilt(self.bp_filter, x, axis=0)
        x_fft = np.abs(np.fft.fft(x_filtered, axis=0))
        x_log = np.log(np.where(x_fft > 1e-8, x_fft, 1e-8))
        win_len = x_log.shape[0]
        return x_log[int(0.5 * win_len // 250): int(30 * win_len // 250)]

class Identity(Preprocessing):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

PREPROCESSING_REGISTRY = {
    "fft_filter": FFTFilter,
    "identity": Identity,
    "github_filter": GitHubFilter
}
