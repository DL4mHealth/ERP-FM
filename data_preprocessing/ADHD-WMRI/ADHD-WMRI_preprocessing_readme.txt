Dataset raw name: EEG raw data - Economical Assessment of Working Memory and Response Inhibition in ADHD
Dataset Link: https://figshare.com/articles/dataset/EEG_raw_data_-_Economical_Assessment_of_Working_Memory_and_Response_Inhibition_in_ADHD/12115773?utm_source=&file=22280061
Paper Link: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2020.00322/full

Short name: ADHD-WMRI

I. EEG Preprocessing Steps

1. Select common EEG channels shared across all `.cnt` files and reorder them.
2. Drop non-EEG channels (HEOG, VEOG, and any channels not marked as 'eeg').
3. Rename channels to match the standard 10–20 naming (e.g., FP1→Fp1, FP2→Fp2, FZ→Fz, CZ→Cz, PZ→Pz).
4. Set the EEG montage to 'standard_1020' (ignore missing positions).
5. Apply a 50 Hz notch filter before bandpass filtering.
6. Apply bandpass filtering with default frequency range: 0.5–40 Hz.
7. Interpolate bad channels if `do_bad_interp` is set to True.
8. Re-reference EEG signals to the average reference.
9. Perform ICA on a 1 Hz high-pass filtered copy:
   - If `mne_icalabel` is available, automatically exclude components related to eye blinks, muscle activity, heartbeats, etc.
   - Otherwise, ICA is applied without component exclusion.
10. Downsample EEG data to 200 Hz.

II. Data Segmentation Steps

1. Extract stimulus events directly from `raw.annotations` in each `.cnt` file.
2. Filter only stimulus events with event code == 2 (S 2).
3. Epoch the data around stimulus onset from -0.2 to 0.8 seconds.
4. Apply baseline correction using the interval from -0.2 to 0 seconds.
5. Reshape EEG data from (N, C, T) to (N, T, C) for model input.

III. Output Format

Each subject's processed data is saved as:

- `X.dat`: EEG data with shape (N, T, C)
- `y.dat: Corresponding labels with shape (N, 4)
- `meta.json`: Metadata containing channel names, sampling rate, and other relevant information.

Where:
- N: number of trials
- T: number of time points per trial (e.g., 200 for 1 seconds at 200 Hz)
- C: number of EEG channels

Label Structure (`y`):

Each label row contains:

[disease_id, stimulus_type, subject_id, task_id]

- `disease_id`:  0: HC, 1: ADHD
- `stimulus_type`: -1 (unknown)
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: 6 (NBackTask), 7 (GoNogoTask), 8 (CombinedTask)
