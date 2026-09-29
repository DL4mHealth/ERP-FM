import os
import numpy as np
import pandas as pd
import torch
from itertools import repeat
from scipy.signal import butter, lfilter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle


def collate_fn(data):
    features, labels = zip(*data)

    shapes = [tuple(x.shape) for x in features]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"All samples in one batch must have the same shape, got: {sorted(set(shapes))}"
        )

    batch_x = torch.stack(features, dim=0).float()
    label_id = torch.stack(labels, dim=0)

    return batch_x, label_id


def bandpass_filter_func(signal, fs, lowcut, highcut):
    # length of signal
    fft_len = signal.shape[1]
    # FFT
    fft_spectrum = np.fft.rfft(signal, n=fft_len, axis=1)
    # get frequency bins
    freqs = np.fft.rfftfreq(fft_len, d=1/fs)
    # create mask for freqs
    mask = (freqs >= lowcut) & (freqs <= highcut)
    # expand mask to match fft_spectrum dimensions
    mask = mask[:, np.newaxis]  # Adjust mask shape if necessary
    # apply mask
    fft_spectrum = fft_spectrum * mask
    # IFFT
    filtered_signal = np.fft.irfft(fft_spectrum, n=fft_len, axis=1)

    return filtered_signal


"""def normalize_batch_ts(batch):
    '''Normalize a batch of time-series data.

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (N, T, C).

    Returns:
        numpy.ndarray: A batch of processed time-series, normalized for each channel of each sample.
    '''
    # Calculate mean and std for each sample's each channel
    mean_values = batch.mean(axis=1, keepdims=True)  # Shape: (N, 1, C)
    std_values = batch.std(axis=1, keepdims=True)  # Shape: (N, 1, C)

    # Perform standard normalization
    normalized_batch = (batch - mean_values) / (std_values + 1e-8)  # Add small value to avoid division by zero

    return normalized_batch


def split_eeg_segments(data, segment_length=128, overlapping=0.5):
    '''
    Splits EEG data into overlapping segments.

    Parameters:
        data (numpy.ndarray): EEG data of shape (T, C), where T is the time dimension and C is the number of channels.
        segment_length (int): Length of each segment.
        overlapping (float): Overlap ratio between consecutive segments (0 to 1).

    Returns:
        numpy.ndarray: Segmented EEG data of shape (num_segments, segment_length, C).
    '''
    if overlapping < 0 or overlapping >= 1:
        raise ValueError("Overlapping ratio must be between 0 and 1.")
    T, C = data.shape
    step_size = int(segment_length * (1 - overlapping))  # Compute step size based on overlap
    num_segments = (T - segment_length) // step_size + 1  # Compute the number of segments
    segments = np.array([data[i:i + segment_length] for i in range(0, num_segments * step_size, step_size)])

    return segments


def load_data_by_ids(data_path, label_path, ids, args):
    '''
    Loads subjects with IDs in the ids list
    Args:
        data_path: directory of data files
        label_path: directory of label files
        ids: list of subject IDs to load
        args: arguments
    Returns:
        X: (N, T, C)
        y: (N, xxx)
    '''
    feature_list = []
    label_list = []

    # load data by subject ids
    for feature_filename, label_filename in zip(os.listdir(data_path), os.listdir(label_path)):
        # get subject ID from filename, e.g., 'AD_1.npy'
        sub_id = int(feature_filename.split('_')[-1].split('.')[0])
        # only load subject with ID in the ids list
        if sub_id in ids:
            sub_feature_path = os.path.join(data_path, feature_filename)
            sub_label_path = os.path.join(label_path, label_filename)
            subject_feature = np.load(sub_feature_path)  # (N, T, C)
            subject_label = np.load(sub_label_path)   # (N, xxx), column number depends on dataset
            if subject_feature.shape[0] != subject_label.shape[0]:
                print(f"Subject {sub_id} data and label length mismatch: " 
                      f"{subject_feature.shape[0]} vs {subject_label.shape[0]}, skipped")
                continue
            feature_list.append(subject_feature)
            label_list.append(subject_label)
    # concat and shuffle
    X = np.concatenate(feature_list, axis=0)
    y = np.concatenate(label_list, axis=0)
    X, y = shuffle(X, y, random_state=42)

    return X, y"""
