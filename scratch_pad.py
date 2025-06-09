import pickle
from pathlib import Path

def load_pickle(pkl_path):
    """
    Load a pickle file and return the stored object.

    Parameters
    ----------
    pkl_path : str or pathlib.Path
        Path to the .pkl file.

    Returns
    -------
    Any
        The Python object that was pickled.
    """
    pkl_path = Path(pkl_path)

    if not pkl_path.is_file():
        raise FileNotFoundError(f"No pickle file found at: {pkl_path}")

    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    return obj


if __name__ == "__main__":
    # Replace with your actual file
    pickle_file = "/home/veit/Downloads/stds_seq2seq_fft_12s_szdetect_single.pkl"


    try:
        result = load_pickle(pickle_file)
        print("Loaded object type:", type(result))
        print("Content preview:\n", result)
    except Exception as e:
        print(f"Error loading pickle: {e}")