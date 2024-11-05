from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt


markers = ['S', 'K', 'REM', 'Son', 'Soff', 'A', 'MS', ]
marker_colors = {
    'S': 'r',
    'K': 'g',
    'REM': 'b',
    'Son': 'm',
    'Soff': 'c',
    'A': 'y',
    'MS': 'k',
}

def load_eeg(data_dir:Path, subject: str) -> mne.io.Raw:
    fname = f"{subject}_eeg_raw.mat"
    fpath = data_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} does not exist")
    
    mat = loadmat(fpath)
    
    eeg_data = mat["EEG"][0, 0]['data']   
    
    info = mne.create_info(ch_names=['EEG1'], sfreq=250, ch_types=['eeg'])
    raw = mne.io.RawArray(eeg_data, info)
    print(raw.info)

    return raw

def load_labels(data_dir:Path, subject: str) -> pd.DataFrame:
    fname = f"{subject}_labels.csv"
    fpath = data_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} does not exist")
    
    df = pd.read_csv(fpath)

    # for each marker M, theres M0 and M1. Change marker names to just M for both
    df['Marker'] = df['Marker'].str.replace('0', '').str.replace('1', '')

    # # keep only rows in which Marker has the '1' suffix
    # df = df[df['Marker'].str.endswith('1')]

    # # remove the '1' suffix from the Marker column
    # df['Marker'] = df['Marker'].str.replace('1', '')

    return df


def get_event_epochs(raw: mne.io.Raw, events_labels: pd.DataFrame, event:str, tmin:int=0, tmax:int=30) -> mne.Epochs:
    df = events_labels
    event_times = df[df['Marker'] == event]["Timestamp_samples"].values
    event_epochs = df[df['Marker'] == event]["Epoch"].values


    # Create an events array for MNE
    events = np.column_stack([event_times, np.zeros(len(event_times), dtype=int), event_epochs])
    try:
        epochs = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    except ValueError as e:
        print(f"Error creating epochs for event {event}: {e}")
        return None

    return epochs



def sample2min(sample: int, sfreq: int) -> float:
    return sample / sfreq / 60


def plot_eeg(raw: mne.io.Raw, subject: str, t0: float, t1: float, events_labels: pd.DataFrame=None):
    time = np.arange(raw._data.shape[1]) / raw.info['sfreq'] / 60

    f, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time, raw._data[0], alpha=.5)

    if events_labels is not None:
        for marker in markers:
            events_samples = events_labels[events_labels['Marker'] == marker]["Timestamp_samples"].values
            events_time = sample2min(events_samples, raw.info['sfreq'])
            ax.vlines(events_time, -400, 400, alpha=0.5, color=marker_colors[marker])

    ax.set(
        xlabel='Time (min)',
        ylabel='EEG Amplitude',
        ylim=[-400, 400],
        xlim=[t0, t1],
        title=f'EEG data for subject {subject} between {t0:.2f} and {t1:.2f} minutes'
        
    )
    return f, ax


