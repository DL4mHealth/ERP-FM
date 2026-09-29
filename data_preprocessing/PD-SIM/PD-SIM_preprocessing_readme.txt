Dataset raw name: Simon-conflict Task
Dataset Link: https://openneuro.org/datasets/ds004580/versions/1.0.0
Paper Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10592174/

Short name: PD-SIM

## Manually delete  "1182.9060000000	1.0000000000	591453	S  3" in sub-33_task-Simon_events.tsv
## No S 3 event in the EEG data, should be a mistake.

I. Preprocess the EEG data as following steps:
    Preprocessing steps ：
      1) choose common channels and reorder
      2) Set Montage
      3) 60 Hz Notch（before band pass）
      4) bandpass filter（default 0.5–40 Hz）
      5) interpolate bad channels（if do_bad_interp is True）
      6) re-reference to average
      7) Perform ICA on a 1 Hz high-pass filtered copy:
      8) downsample to 200 Hz


II. Data segmentation steps:
    1) Extract the events from the task-Simon_events.tsv file.
    2) Use the S1 - Visual Cue event as the lock for segmenting.
    3) Epoch the data from -500 ms to 1000 ms around the S1 - Visual Cue event.
    4) Baseline correct the epochs using the pre-stimulus interval (-300 ms to -200 ms).


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
- `task_id`: always 2 (Simon-conflict Task)
