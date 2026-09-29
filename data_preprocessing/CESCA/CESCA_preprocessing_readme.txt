Dataset raw name: Cognitive Electrophysiology in Socioeconomic Context in Adulthood: An EEG dataset
Dataset Link: https://openneuro.org/datasets/ds006018/versions/1.2.2
Paper Link:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307406
https://www.nature.com/articles/s41597-025-05209-z

Short name: CESCA
Sub-task dataset short name: CESCA-AODD, CESCA-VODD, CESCA-VS, CESCA-FLANKER


This data processing file only processes the auditory oddball task and visual oddball task of the
Cognitive Electrophysiology in Socioeconomic Context in Adulthood: An EEG dataset.

I. Preprocess the EEG data as following steps:
    Preprocessing steps ：
      1) choose common channels and reorder
      2) Set Montage
      3) 60 Hz Notch（before band pass）
      4) bandpass filter（default 0.5–40 Hz）
      5) interpolate bad channels（if do_bad_interp is True）
      6) re-reference to average
      7) ICA（在 1 Hz 高通的副本上拟合，自动剔除眼动/肌电等分量，需 mne-icalabel）
      8) downsample to 200 Hz

II. Data segmentation steps:
    1) Extract the events from the task-auditoryoddball_events.tsv or task-visualoddball_events.tsv or task-visualsearch_eeg_events.tsv or task-flanker_events.tsv file
    2) Remove invalid events at the beginning - 'S  1', 'S  2', 'S180', 'boundary', 'Buffer Overflow'
    3) Epoch the data from -200ms to 800ms around stimulation and response events.
    4) Baseline correct the epochs using the pre-stimulus interval (-200 ms to 0 ms).


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

CESCA-AODD:
- `disease_id`:  -1 (unknown)
- `stimulus_type`: 0 (standard), 1 (target)
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: 0 (auditory oddball task)

CESCA-VODD:
- `disease_id`:  -1 (unknown)
- `stimulus_type`: 0 (standard), 1 (target)
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: 1 (visual oddball task)

CESCA-VS:
- `disease_id`:  -1 (unknown)
- `stimulus_type`: 0 (left target), 1 (right target)
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: 10 (visual search task)

CESCA-FLANKER:
- `disease_id`:  -1 (unknown)
- `stimulus_type`: 0 (congruent), 1 (incongruent)
- `subject_id`: numeric identifier from 'sub-XXX'
- `task_id`: 5 (flanker task)
