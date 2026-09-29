Dataset raw name: The Nencki-Symfonia EEG/ERP dataset
Dataset Link: https://openneuro.org/datasets/ds004621/versions/1.0.4
Paper Link: https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac015/6543635

Short name: NSERP
Three processed sub-dataset dependent on task type: SRT, Oddball, MSIT: NSERP-MSIT, NSERP-ODD, NSERP-SRT

I. EEG Preprocessing Steps

1. Select common EEG channels shared across all subjects and reorder them.
2. Drop non-standard channels not included in 'standard_1020' montage.
3. Set the EEG montage to 'standard_1020'.
4. Apply a 50 Hz notch filter before bandpass filtering.
5. Apply bandpass filtering with default frequency range: 0.5–40 Hz.
6. Interpolate bad channels if `do_bad_interp` is set to True.
7. Re-reference EEG signals to the average reference.
8. Perform ICA on a 1 Hz high-pass filtered copy:
   - If `mne_icalabel` is available, automatically exclude components related to eye blinks, muscle activity, heartbeats, etc.
   - Otherwise, ICA is applied without component exclusion.
9. Downsample EEG data to 250 Hz.

II. Data Segmentation Steps

1. Extract stimulus events from the corresponding `events.tsv` file.
2. Retain only events starting with 'S' followed by a number (e.g., S 5, S 7, S 8).
3. Map event codes to stimulus labels by task type:
    # SRT: only 5;
    # Oddball: 5,6,7 for standard,target,deviant;
    # MSIT: 5,6,7,8 for F0, FS, 00, S0 according to paper
4. Epoch the data around stimulus onset from -0.5 to 1.0 seconds.
5. Apply baseline correction using the interval from -0.5 to -0.0 seconds.
6. Reshape EEG data from (N, C, T) to (N, T, C) for model input.

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

NSERP-MSIT:
- `disease_id`:  -1 (unknown)
- `stimulus_type`: 5 (F0), 6 (FS), 7 (00), 8 (S0) -> mapped to 0,1,2,3 respectively
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: 4 (MSIT Task)

NSERP-ODD:
- `disease_id`:  -1 (unknown)
- `stimulus_type`: 5 (standard), 6 (target), 7 (novel) -> mapped to 0,1,2 respectively
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: 1 (Visual Oddball Task)

NSERP-SRT:
- `disease_id`:  -1 (unknown)
- `stimulus_type`: 5 (simple stimulus response) -> mapped to 0
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: 9 (SRT Task)
