Dataset raw name: EEG: Reinforcement Learning in Parkinson's
Dataset Link: https://openneuro.org/datasets/ds003506/versions/1.1.0
Paper Link: https://www.sciencedirect.com/science/article/pii/S0006899319305955

Short name: RLPD

I. Preprocess the EEG Data

Preprocessing Steps:
1. Select common EEG channels and reorder them (excluding unreliable electrodes: FT9, FT10, TP9, TP10).
2. Set the standard 10-20 montage (`standard_1020`).
3. Apply a 60 Hz notch filter before band-pass filtering.
4. Apply a band-pass filter (default: 0.5-40 Hz).
5. Interpolate bad channels if `do_bad_interp=True`.
6. Re-reference to average.
7. Perform ICA on a 1 Hz high-pass copy of the data and automatically remove eye movement/muscle components using `mne-icalabel` thresholds:
   - Eye blink >= 0.7
   - Muscle artifact >= 0.6
   - Heartbeat >= 0.5
   - Line noise >= 0.8
   - Channel noise >= 0.9
8. Downsample to 200 Hz if necessary.

Notes:
- The pipeline supports loading `.set` files directly with fallback to epoch reading if needed.
- Event sample indices should be computed from `onset * raw.info["sfreq"]` after preprocessing/resampling. Do not directly use the `sample` column for epoch construction.

II. Data Segmentation Steps

This dataset contains two parts with different event structures. The preprocessing function uses a mixed-locking segmentation strategy:

1. Training trials: feedback-locked segmentation
   - Training trials contain instruction and feedback markers.
   - Use feedback onset as the lock point.
   - Keep only normal feedback events:
     - `S10` = feedback 0 / no reward
     - `S11` = feedback +1 / reward
   - Exclude invalid feedback trials:
     - `S6` = FB: "No Match"
     - `S7` = FB: "Too Slow"
   - The current trial condition is determined by the most recent instruction marker before feedback:
     - `S1` = Instr: "Choose"
     - `S2` = Instr: "Match"

2. Test trials: response-locked segmentation
   - Test trials do not contain instruction or feedback information.
   - Use button response onset as the lock point.
   - Classify test trials according to response side:
     - `S4` = left button response
     - `S5` = right button response
   - Response events that occur inside an active training trial are not used as test events.

3. Epoch window
   - Default epoch window: -0.2 s to +0.8 s around the selected lock point.
   - Default baseline correction: (-0.2 s, 0 s).
   - For training trials, the lock point is the feedback event.
   - For test trials, the lock point is the response event.

III. Output Data Structure

Each subject's processed data is saved as:

- `X.dat`: EEG data with shape (N, T, C)
- `y.dat`: corresponding labels with shape (N, 4)
- `meta.json`: metadata containing channel names, sampling rate, label format, and other relevant information

Where:
- N: number of retained trials
- T: number of time points per trial
- C: number of EEG channels

Label Structure (`y`):

Each label row contains:

[disease_id, stimulus_type, subject_id, task_id]

- `disease_id`: 0 = CTL, 1 = PD
- `stimulus_type`: mixed training/test condition label
- `subject_id`: numeric identifier from `sub-XXX`
- `task_id`: always 3 (Reinforcement Learning task)

Stimulus Type Mapping:

Training phase, feedback-locked:
- 0 = choose_reward
- 1 = choose_no_reward
- 2 = match_reward
- 3 = match_no_reward

Test phase, response-locked:
- 4 = test_left_response
- 5 = test_right_response

IV. Recommended `epoch_and_make_xy` Logic

The segmentation function should scan events sequentially:

1. When `S1` is encountered, mark the current training condition as `choose`.
2. When `S2` is encountered, mark the current training condition as `match`.
3. When `S10` or `S11` is encountered during an active training trial, create a feedback-locked epoch and assign one of labels 0-3.
4. When `S6` or `S7` is encountered, discard the current training trial and reset the trial state.
5. When `S4` or `S5` is encountered outside an active training trial, create a response-locked test epoch and assign label 4 or 5.
6. All labels are stored in the second column of `y` as integer `stimulus_type` values.

V. Notes for ERP-FM Usage

- This dataset is a disease-related reinforcement-learning EEG dataset.
- Training and test phases should not be collapsed into the same label space because their event structures and lock points are different.
- Training labels preserve the reward/condition structure that is relevant to feedback-related reward processing.
- Test labels preserve the response-side information when no instruction/feedback markers are available.
- Although the field is named `stimulus_type` for consistency across datasets, here it represents mixed event types: feedback-locked training conditions and response-locked test response sides.
