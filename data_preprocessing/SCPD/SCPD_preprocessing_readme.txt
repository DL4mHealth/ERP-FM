Dataset raw name: EEG: Simon Conflict in Parkinson's
Dataset Link: https://openneuro.org/datasets/ds003509/versions/1.1.0
Paper Link: https://www.sciencedirect.com/science/article/pii/S0028393218302185

Short name: SCPD

I. Preprocess the EEG Data

Preprocessing Steps:**
1. Select common EEG channels and reorder them (excluding unreliable electrodes: FT9, FT10, TP9, TP10).
2. Set the standard 10–20 montage (`standard_1020`).
3. Apply a **60 Hz notch filter** before band-pass filtering.
4. Apply a **band-pass filter** (default: 0.5–40 Hz).
5. Interpolate bad channels if `do_bad_interp=True`.
6. Re-reference to average.
7. Perform ICA on a 1 Hz high-pass copy of the data and automatically remove eye movement/muscle components using **mne-icalabel** thresholds:
   - Eye blink ≥ 0.7
   - Muscle artifact ≥ 0.6
   - Heartbeat ≥ 0.5
   - Line noise ≥ 0.8
   - Channel noise ≥ 0.9
8. Downsample to **200 Hz** if necessary.

**Notes:**
- Subject `sub-026` is automatically skipped due to a known ICA issue.
- The pipeline supports loading `.set` files directly with fallback to epoch reading if needed.

II. Data Segmentation Steps

1. Extract **cue-lock events** from the `task-Simon_events.tsv` file using `trial_type` entries beginning with "Test Stim" and "Trn Stim".
2. Use the **stimulus onset times** as the lock point for segmenting.
3. Epoch the data from **-0.5 s to +1.0 s** around the cue event.
4. Baseline-correct the epochs using the interval **(-0.3 s to -0.2 s)**.
5. Set stimulus_type label to 0 for "train_yellow_congruent", 1 for "train_yellow_incongruent", 2 for "train_blue_congruent", 3 for "train_blue_incongruent", 4 for "test_AB", 5 for "test_AC", 6 for "test_AD", 7 for "test_BC", 8 for "test_BD", and 9 for "test_CD".

III. Output Data Structure

Each subject's processed data is saved as:

- `X.dat`: EEG data with shape (N, T, C)
- `y.dat: Corresponding labels with shape (N, 4)
- `meta.json`: Metadata containing channel names, sampling rate, and other relevant information.

Where:
- N: number of trials
- T: number of time points per trial (e.g., 250 for 1 seconds at 250 Hz)
- C: number of EEG channels

Label Structure (`y`):

Each label row contains:

[disease_id, stimulus_type, subject_id, task_id]

- `disease_id`:  0: CTL, 1: PD
- `stimulus_type`: 0: "train_yellow_congruent", 1: "train_yellow_incongruent", 2: "train_blue_congruent", 3: "train_blue_incongruent", 4: "test_AB", 5: "test_AC", 6: "test_AD", 7: "test_BC", 8: "test_BD", 9: "test_CD"
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: always 2 (Simon Conflict task)
