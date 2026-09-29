Dataset raw Name: Cross-Modal Oddball Task
Dataset Link: https://openneuro.org/datasets/ds004574/versions/1.0.0
Paper Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10592174/

Short name: PD-ODD

I. EEG Preprocessing Steps

1. Select common EEG channels across all subjects and reorder them consistently.
2. Set the EEG montage to 'standard_1020'.
3. Apply a 60 Hz notch filter before bandpass filtering.
4. Apply bandpass filtering with a default range of 0.5–40 Hz.
5. Interpolate bad channels if any are marked and `do_bad_interp` is True.
6. Re-reference the EEG signals to the average reference.
7. Perform ICA on a 1 Hz high-pass filtered copy:
8. Downsample all EEG data to 200 Hz.

II. Data Segmentation Steps

1. Extract events from the `events.tsv` file.
2. Select events with value code `S 2`(go cue) as time-lock  for segmentation.
3. Match these events with behavioral data from `beh.tsv`, using the number of trials in both files.
4. Epoch EEG data from -0.5 to 1.0 seconds around the selected event onset.
5. Apply baseline correction using the interval from -0.3 to -0.2 seconds.
6. Reshape data into shape (N, T, C), where:
   - N = number of trials,
   - T = number of time points,
   - C = number of EEG channels.

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

- `disease_id`:  0: CTL, 1: PD
- `stimulus_type`: -1 (unknown)
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: always 1 (Cross-Modal Oddball Task)