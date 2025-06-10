import numpy as np
from scipy.signal import iirnotch, lfilter, butter, filtfilt, resample
from preprocessing.utils import Preprocessing
from preprocessing.utils import make_a_filtered_plot_for_comparison, plot_signal_in_frequency
from scipy.fftpack import fft
from model.neuro_constants import META_NODE_INDICES

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


class BandpassFFT(Preprocessing):
    def __init__(self,
                 lowcut: float = 0.5,
                 highcut: float = 120.0, 
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
# From https://github.com/USC-InfoLab/NeuroGNN/blob/main/data/data_utils.py#L13
def computeFFT(signals, n):
    fourier_signal = fft(signals, n=n, axis=-1)
    
    # cut away symetric negative frequencys
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]

    amp = np.abs(fourier_signal)
    # amp[amp == 0.0] = 1e-8
    amp = np.maximum(amp, 1e-8)
    return np.log(amp)

def augment_data(x, meta_node_indices):
    """
    Args:
        x: (max_seq_len, num_nodes, input_dim)
        meta_node_indices: list of indices of meta nodes
    Returns:
        x: (max_seq_len, num_nodes + len(meta_node_indices), input_dim)
    """
    # for index_list in meta_node_indices:
    #     node_series_list = x[:, index_list, :]  # Extract the series for the current node from x
    #     meta_series = node_series_list.mean(axis=1, keepdims=True)  # Take the mean of the series
    #     x = torch.cat([x, meta_series], axis=1)
    # return x
    """
    x : (max_seq_len, num_nodes, input_dim)  NumPy array
    returns : (max_seq_len, num_nodes + len(meta_node_indices), input_dim)
    """
    # for index_list in meta_node_indices:
    #     meta_series = x[:, index_list, :].mean(axis=1, keepdims=True)   # axis=1 → electrodes :contentReference[oaicite:3]{index=3}
    #     x_2 = np.concatenate([x, meta_series], axis=1) 

    extra = [x[:, idx, :].mean(axis=1, keepdims=True)
    for idx in meta_node_indices]        # no side-effects
    x_1 = np.concatenate([x] + extra, axis=1)
    
                     # append along node-axis
    # assert np.allclose(x_1, x_2)
    return x_1

class StandardScaler:
    """
    Parameters
    ----------
    mean : array-like
        Either a scalar or an array shaped (1, num_nodes, 1) holding the
        per-node mean in the same format used by the original code.
    std : array-like
        Matching per-node standard deviation.  A small epsilon is added
        internally to guard against divide-by-zero.
    """

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

        # optional safety check
        if np.any(self.std == 0):
            raise ValueError("std contains zeros; add epsilon or verify input.")

    # ------------------------------------------------------------------ #
    # Forward transform:  z = (x − μ) / σ
    # ------------------------------------------------------------------ #
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Standardise *data* clip-wise.

        Parameters
        ----------
        data : np.ndarray
            Shape (..., num_nodes, input_dim) – the leading axes can be
            batch/time; broadcasting handles them automatically.

        Returns
        -------
        np.ndarray
            Standardised array with the same shape as *data*.
        """
        return (data - self.mean) / self.std

    # ------------------------------------------------------------------ #
    # Inverse transform:  x = z⋅σ + μ
    # ------------------------------------------------------------------ #
    def inverse_transform(
        self,
        data: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Revert standardisation; optionally ignore rows given by *mask*.

        Parameters
        ----------
        data : np.ndarray
            Standardised values.
        mask : np.ndarray or None, optional
            Boolean array with shape matching the *batch* dimension
            (e.g. (batch_size,)).  Entries set to True will **not** be
            inverse-scaled – useful when some clips are padded or invalid.

        Returns
        -------
        np.ndarray
            Unscaled data (same dtype/shape as input).
        """
        original = data * self.std + self.mean
        if mask is not None:
            # broadcasting mask over all trailing dims
            original = np.where(mask[..., None, None], data, original)
        return original


class NeuroGNNFilter(Preprocessing):
    def __init__(self,
                 fs: int = 250,
                 resampleFS: int = 250,
                 time_step_size: int = 1,
                 clip_len: int = 12,
                 ):
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
        self.fs = fs
        self.resampleFS = resampleFS

        # new
        self.clip_len = clip_len
        self.time_step_size = time_step_size

        self.mean = 3.9594627590488702
        self.std = 1.5625200363244962

        self.scalar = StandardScaler(self.mean, self.std)
        self.resample = resampleFS != fs

        self.physical_clip_len = int(self.resampleFS * clip_len)
        self.physical_time_step_size = int(self.resampleFS * time_step_size)

        
    def __call__(self, signals: np.ndarray) -> np.ndarray:
        # First transpose to match original format (channels, time)
        signals = signals.T  # Now shape is (n_samples, n_windows)
        
        if  self.resample:
            signals = resample(signals, num=self.resampleFS*self.clip_len, axis=1)
        

        # Create time windows as in original
        time_steps = []
        start_time_step = 0
        while start_time_step <= signals.shape[1] - self.physical_time_step_size:
            end_time_step = start_time_step + self.physical_time_step_size
            curr_time_step = signals[:, start_time_step:end_time_step]
            FT = computeFFT(curr_time_step, self.physical_time_step_size)
            time_steps.append(FT)
            start_time_step = end_time_step
        
        # Stack and transpose back to original format
        filtered = np.stack(time_steps, axis=0)

        filtered = augment_data(filtered, META_NODE_INDICES)

        filtered = self.scalar.transform(filtered)

        # filtered_turn = filtered.T
        
        # make_a_filtered_plot_for_comparison(signals, filtered, self.resampleFS)
        # plot_signal_in_frequency(signals[0], filtered[0], self.resampleFS)
        
        return filtered



