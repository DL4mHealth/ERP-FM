"""
Multi-dataset loader for heterogeneous ERP datasets with different sequence lengths (T) and channel counts (C).

Main goal:
    1. Load multiple ERP datasets without concatenating X.
    2. Allow each dataset to have different T and C in X: N x T x C.
    3. Ensure each mini-batch contains samples from only one dataset.
    4. Shuffle the order of dataset-level mini-batches across an epoch.

Expected output from DataLoader after collate_fn:
    batch_x:  B x T_d x C_d
    label_id: B x 5, columns = [class_label, stimulus_type, subject_id, task_id, dataset_id]
"""


import os
from collections import OrderedDict
from typing import Dict, Iterator, List
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# Reuse your existing dataset-folder-to-loader mapping.
# This keeps all existing single-dataset loaders unchanged.
from data_provider.dataset_loader.tuep_loader import TUEPLoader
from data_provider.dataset_loader.caueeg_loader import CAUEEGLoader
from data_provider.dataset_loader.tdbrain_loader import TDBRAINLoader
from data_provider.dataset_loader.baca_rs_loader import BACARSLoader
from data_provider.dataset_loader.p_adic_loader import PADICLoader
from data_provider.dataset_loader.pd_odd_loader import PDODDLoader
from data_provider.dataset_loader.pd_sim_loader import PDSIMLoader
from data_provider.dataset_loader.pd_itt_loader import PDITTLoader
from data_provider.dataset_loader.scpd_loader import SCPDLoader
from data_provider.dataset_loader.rlpd_loader import RLPDLoader
from data_provider.dataset_loader.aopd_loader import AOPDLoader
from data_provider.dataset_loader.adhd_wmri_loader import ADHDWMRILoader
from data_provider.dataset_loader.cognision_erp_loader import CognisionERPLoader
from data_provider.dataset_loader.nserp_loader import NSERPLoader
from data_provider.dataset_loader.cesca_loader import CESCALoader
from data_provider.dataset_loader.mtbi_loader import MTBILoader
from data_provider.dataset_loader.ims_odd_loader import IMSODDLoader
from data_provider.dataset_loader.runabout_loader import RunaboutLoader
from data_provider.dataset_loader.avss_loader import AVSSLoader
from data_provider.dataset_loader.cct_mjah_loader import CCTMJAHLoader
from data_provider.dataset_loader.heartbeam_loader import HeartBEAMLoader
from data_provider.dataset_loader.avspp_loader import AVSPPLoader
from data_provider.dataset_loader.mri_aodd_loader import MRIAODDLoader
from data_provider.dataset_loader.pstcc_loader import PSTCCLoader
from data_provider.dataset_loader.vwmcc_loader import VWMCCLoader
from data_provider.dataset_loader.nafps_loader import NAFPSLoader
from data_provider.dataset_loader.sice_loader import SICELoader
from data_provider.dataset_loader.go_nogo_loader import GoNogoLoader
from data_provider.dataset_loader.epsd_loader import EPSDLoader
from data_provider.dataset_loader.aud_loader import AUDLoader
from data_provider.dataset_loader.tabg_loader import TABGLoader
from data_provider.dataset_loader.simcc_loader import SIMCCLoader
from data_provider.dataset_loader.plaf_loader import PLAFLoader
from data_provider.dataset_loader.mmpst_loader import MMPSTLoader
from data_provider.dataset_loader.erpcore_loader import ERPCORELoader
from data_provider.dataset_loader.hbn_eeg_loader import HBNEEGLoader

# data folder dict to loader mapping
data_folder_dict = {
    # should use the same name as the dataset folder
    'TUEP': TUEPLoader,
    'CAUEEG': CAUEEGLoader,
    'TDBrain': TDBRAINLoader,
    'BACA-RS': BACARSLoader,
    'P-ADIC': PADICLoader,
    'PD-ODD': PDODDLoader,
    'PD-SIM': PDSIMLoader,
    'PD-ITT': PDITTLoader,
    'SCPD': SCPDLoader,
    'RLPD': RLPDLoader,
    'AOPD': AOPDLoader,
    'Cognision-ERP': CognisionERPLoader,
    'ADHD-WMRI': ADHDWMRILoader,
    'NSERP-MSIT': NSERPLoader,
    'NSERP-ODD': NSERPLoader,
    'NSERP-SRT': NSERPLoader,
    'CESCA-AODD': CESCALoader,
    'CESCA-VODD': CESCALoader,
    'CESCA-VS': CESCALoader,
    'CESCA-FLANKER': CESCALoader,
    'mTBI-ODD': MTBILoader,
    'mTBI-DPX': MTBILoader,
    'mTBI-VWM': MTBILoader,
    'IMS-ODD': IMSODDLoader,
    'Runabout': RunaboutLoader,
    'AVSS-AODD': AVSSLoader,
    'AVSS-VODD': AVSSLoader,
    'CCT-MJAH': CCTMJAHLoader,
    'HeartBEAM': HeartBEAMLoader,
    'AVSPP': AVSPPLoader,
    'MRI-AODD': MRIAODDLoader,
    'PSTCC': PSTCCLoader,
    'VWMCC': VWMCCLoader,
    'NAFPS': NAFPSLoader,
    'SICE': SICELoader,
    'Go-Nogo': GoNogoLoader,
    'EPSD': EPSDLoader,
    'AUD-MAB': AUDLoader,
    'AUD-PS': AUDLoader,
    'TABG': TABGLoader,
    'SIMCC-TRAIN': SIMCCLoader,
    'SIMCC-TEST': SIMCCLoader,
    'PLAF-EXP1': PLAFLoader,
    'PLAF-EXP2': PLAFLoader,
    'MMPST-EXP1-TRAIN': MMPSTLoader,
    'MMPST-EXP1-TEST': MMPSTLoader,
    'MMPST-EXP2-TRAIN': MMPSTLoader,
    'MMPST-EXP2-TEST': MMPSTLoader,
    'TDBrain-ODD': TDBRAINLoader,
    'ERPCORE-N170': ERPCORELoader,
    'ERPCORE-MMN': ERPCORELoader,
    'ERPCORE-N2pc': ERPCORELoader,
    'ERPCORE-N400': ERPCORELoader,
    'ERPCORE-P3': ERPCORELoader,
    'ERPCORE-LRP': ERPCORELoader,
    'ERPCORE-ERN': ERPCORELoader,
    'HBN-EEG-SUS': HBNEEGLoader,
    'HBN-EEG-CCD': HBNEEGLoader,
    'HBN-EEG-SYS': HBNEEGLoader,
}

# Global stable dataset IDs.
# Do NOT assign dataset_id with enumerate(data_folder_list), because TRAIN/VAL/TEST
# may use different dataset lists or different orders. The same dataset name must
# always map to the same dataset_id so that model-side buffers, such as 3D
# electrode coordinates, remain consistent across splits.
global_dataset_id_by_name = {
    dataset_name: dataset_idx + 1
    for dataset_idx, dataset_name in enumerate(data_folder_dict.keys())
}

warnings.filterwarnings('ignore')


class MultiDataset(Dataset):
    """
    A wrapper for multiple ERP datasets with different sequence lengths and channels.

    IMPORTANT:
        This class does NOT concatenate X from different datasets.
        It only concatenates labels/metadata and keeps a global-index mapping.
    """

    def __init__(self, args, root_path, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = self._normalize_flag(flag)
        self.no_normalize = getattr(args, "no_normalize", False)

        data_folder_list = self._get_dataset_list(args, self.flag)
        print(f"Loading {self.flag} samples from multiple ERP datasets...")
        print(f"Datasets used: {data_folder_list}")

        self.datasets = []
        self.dataset_names: List[str] = []
        self.dataset_ids: List[int] = []
        self.dataset_id_to_name: Dict[int, str] = {}
        self.global_to_local: List[tuple] = []  # list of (dataset_idx, local_idx)
        self.indices_by_dataset: Dict[int, List[int]] = {}

        # Dataset-level metadata from meta.json.
        # These dictionaries are useful for potential channel/sampling-rate embeddings.
        # Keys are dataset_id starting from 1, matching label_id[:, 4].
        self.dataset_metadata_by_id: Dict[int, dict] = {}
        self.channel_names_by_id: Dict[int, List[str]] = {}
        self.montage_by_id: Dict[int, str] = {}
        self.sample_rate_by_id: Dict[int, int] = {}

        all_y = []
        global_index = 0
        max_seq_len = 0
        max_num_channels = 0

        for dataset_idx, dataset_name in enumerate(data_folder_list):
            if dataset_name not in data_folder_dict:
                raise ValueError(
                    f"Dataset name '{dataset_name}' is not found in data_folder_dict. "
                    f"Available datasets: {list(data_folder_dict.keys())}"
                )

            Data = data_folder_dict[dataset_name]
            dataset = Data(
                root_path=os.path.join(root_path, dataset_name),
                args=args,
                flag=self.flag,
                dataset_name=dataset_name,
            )

            if not hasattr(dataset, "X") or not hasattr(dataset, "y"):
                raise AttributeError(
                    f"The loader for {dataset_name} must provide dataset.X and dataset.y."
                )

            if dataset.y.ndim != 2 or dataset.y.shape[1] != 4:
                raise ValueError(
                    f"{dataset_name}.y should have four columns: "
                    f"[label, stimulus_type, subject_id, task_id]. Got shape {dataset.y.shape}."
                )

            # Use a global stable dataset_id based on dataset name.
            # Column format:
            # [label, stimulus_type, subject_id, task_id, dataset_id].
            # The original task_id column is preserved.
            y = np.asarray(dataset.y).copy()
            dataset_id = global_dataset_id_by_name[dataset_name]
            dataset_id_col = np.full((y.shape[0], 1), dataset_id, dtype=y.dtype)
            y = np.concatenate([y[:, :4], dataset_id_col], axis=1)
            dataset.y = y

            n_samples = len(dataset.y)
            dataset_global_indices = list(range(global_index, global_index + n_samples))
            self.indices_by_dataset[dataset_idx] = dataset_global_indices
            self.global_to_local.extend((dataset_idx, local_idx) for local_idx in range(n_samples))
            global_index += n_samples

            self.datasets.append(dataset)
            self.dataset_names.append(dataset_name)
            self.dataset_ids.append(dataset_id)
            self.dataset_id_to_name[dataset_id] = dataset_name
            all_y.append(y)

            # Save metadata for model-side channel embeddings. For memmap loaders, these come from meta.json.
            dataset_metadata = getattr(dataset, "dataset_metadata", {}) or {}
            self.dataset_metadata_by_id[dataset_id] = dataset_metadata
            self.channel_names_by_id[dataset_id] = list(getattr(dataset, "channels", dataset_metadata.get("channels", [])))
            self.montage_by_id[dataset_id] = getattr(dataset, "montage", dataset_metadata.get("montage", None))
            self.sample_rate_by_id[dataset_id] = getattr(dataset, "sample_rate", dataset_metadata.get("sample_rate", None))

            # Statistics only. Do not use them to force all datasets into the same shape.
            seq_len = int(dataset.X.shape[1])
            num_channels = int(dataset.X.shape[2])
            max_seq_len = max(max_seq_len, seq_len)
            max_num_channels = max(max_num_channels, num_channels)

            print(f"[dataset_id={dataset_id}] {dataset_name}: "
                  f"N={dataset.y.shape[0]}, T={seq_len}, C={num_channels}")

        if len(self.datasets) == 0:
            raise ValueError("No dataset was loaded. Please check your parser dataset list.")

        # y can be concatenated because all labels now have shape N x 5:
        # [label, stimulus_type, subject_id, task_id, dataset_id].
        self.y = np.concatenate(all_y, axis=0)

        # Make metadata reachable from configs/args after _get_data() builds the dataset.
        # This is a convenient path for future model modules that need channel names, montage, or sample rate.
        self.args.global_dataset_id_by_name = global_dataset_id_by_name
        self.args.dataset_id_to_name = self.dataset_id_to_name
        self.args.dataset_metadata_by_id = self.dataset_metadata_by_id
        self.args.channel_names_by_id = self.channel_names_by_id
        self.args.montage_by_id = self.montage_by_id
        self.args.sample_rate_by_id = self.sample_rate_by_id

        classify_choice = getattr(args, "classify_choice", "disease")
        if classify_choice == "disease":
            target_col = 0
        elif classify_choice == "stimulus":
            target_col = 1
        else:
            raise ValueError(f"classify_choice must be 'disease' or 'stimulus', got {classify_choice}.")

        # Compatibility fields used by supervised/linear-probe/fine-tuning code.
        if self.flag in ["TRAIN", "VAL", "TEST"] and len(self.datasets) == 1:
            # Downstream uses exactly one dataset, so enc_in and num_class are real fixed number.
            self.seq_len = max_seq_len
            self.enc_in = max_num_channels
            self.num_class = len(np.unique(self.y[:, target_col]))
        elif self.flag == "PRETRAIN":
            # Pretraining uses multiple datasets, so only max_seq_len and max_num_channels are meaningful.
            self.max_seq_len = max_seq_len
            self.max_num_channels = max_num_channels

        # Do NOT create self.X by concatenating datasets. Different T/C may exist.
        self.X = None

        """print(f"Loaded {len(self.datasets)} datasets, total samples={len(self.y)}, "
              f"classify_choice={getattr(self.args, 'classify_choice', 'disease')}, "
              f"num_class={self.num_class}, max_T={self.max_seq_len}, max_C={self.max_num_channels}.")
        print("Note: X is not concatenated because different datasets may have different T and C.\n")"""
        if self.flag in ["TRAIN", "VAL", "TEST"] and len(self.datasets) == 1:
            print(f"{self.flag} dataset: {self.dataset_names[0]}, "
                  f"N={self.y.shape[0]}, T={self.datasets[0].X.shape[1]}, C={self.datasets[0].X.shape[2]}, "
                  f"num_class={self.num_class}, classify_choice={getattr(self.args, 'classify_choice', 'disease')}.\n")
        elif self.flag == "PRETRAIN":
            print(f"{self.flag} datasets: total trials={len(self.y)}, "
                  f"max_T={self.max_seq_len}, max_C={self.max_num_channels}.")
            print("Note: X is not concatenated because different datasets may have different T and C.\n")

    @staticmethod
    def _normalize_flag(flag):
        if flag in ["TRAIN", "train"]:
            return "TRAIN"
        if flag in ["VAL", "val", "valid"]:
            return "VAL"
        if flag in ["TEST", "test"]:
            return "TEST"
        if flag in ["PRETRAIN", "pretrain"]:
            return "PRETRAIN"
        raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")

    @staticmethod
    def _split_dataset_names(dataset_string: str) -> List[str]:
        return [x.strip() for x in dataset_string.split(",") if x.strip()]

    def _get_dataset_list(self, args, flag) -> List[str]:
        """
        PRETRAIN may use a comma-separated list of datasets.

        TRAIN/VAL/TEST are downstream supervised splits and must all use the
        same single dataset specified by args.training_datasets. The actual
        train/val/test samples are selected inside each single-dataset loader
        through subject-level splits, so there is no separate testing_datasets
        argument anymore.
        """
        if flag == "PRETRAIN":
            dataset_list = self._split_dataset_names(args.pretraining_datasets)
        elif flag in ["TRAIN", "VAL", "TEST"]:
            dataset_list = self._split_dataset_names(args.training_dataset)
            if len(dataset_list) != 1:
                raise ValueError(
                    f"For {flag} dataset, exactly one training dataset is allowed. "
                    f"Got {len(dataset_list)} datasets: {dataset_list}. "
                    "Only pretraining_datasets can be a comma-separated list."
                )
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")

        if len(dataset_list) == 0:
            raise ValueError(f"No dataset was provided for flag={flag}.")
        return dataset_list

    def __getitem__(self, index):
        dataset_idx, local_idx = self.global_to_local[index]
        dataset = self.datasets[dataset_idx]

        # Important: call the underlying dataset's __getitem__ instead of directly
        # indexing dataset.X. For memmap datasets, this reads only one trial from
        # X.dat and applies on-the-fly normalization without loading the full array.
        return dataset[local_idx]

    def __len__(self):
        return len(self.global_to_local)

    def get_dataset_metadata(self, dataset_id: int) -> dict:
        """Return metadata for one dataset_id. dataset_id matches label_id[:, 4]."""
        return self.dataset_metadata_by_id[int(dataset_id)]

    def get_batch_metadata(self, label_id) -> dict:
        """
        Return metadata for a homogeneous batch.

        Because DatasetHomogeneousBatchSampler guarantees that one batch comes
        from one dataset, label_id[:, 4] should contain exactly one value.
        """
        if torch.is_tensor(label_id):
            dataset_ids = torch.unique(label_id[:, 4]).detach().cpu().numpy().tolist()
        else:
            dataset_ids = np.unique(np.asarray(label_id)[:, 4]).tolist()
        if len(dataset_ids) != 1:
            raise ValueError(f"Expected one dataset_id per batch, got {dataset_ids}.")
        return self.get_dataset_metadata(int(dataset_ids[0]))

    def summary(self) -> str:
        lines = [
            f"Dataset(flag={self.flag}), total trials: {len(self)}, total datasets: {len(self.datasets)}."
        ]
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_id = self.dataset_ids[dataset_idx]
            lines.append(
                f"  dataset_id={dataset_id}, name={self.dataset_names[dataset_idx]}, "
                f"trials={dataset.y.shape[0]}, "
                f"seq_len={dataset.X.shape[1] if dataset.X is not None else 'N/A'}, "
                f"channels={len(self.channel_names_by_id.get(dataset_id, []))}, "
                f"montage={self.montage_by_id.get(dataset_id)}, "
            )
        return "\n".join(lines)


class SubjectStimulusAveragedDataset(Dataset):
    """
    ERP trial averaging wrapper.

    When enabled, TRAIN/VAL/TEST samples are grouped by (dataset_id, subject_id,
    stimulus_type). All trials in one group are averaged into one ERP sample
    on the fly. PRETRAIN is not wrapped, so MAE pretraining remains strictly
    single-trial.
    """

    def __init__(self, base_dataset: Dataset, args, flag: str):
        self.base_dataset = base_dataset
        self.args = args
        self.flag = flag
        self.group_keys = []
        self.group_indices = []

        base_y = np.asarray(base_dataset.y)
        if base_y.ndim != 2 or base_y.shape[1] < 5:
            raise ValueError(
                "SubjectStimulusAveragedDataset expects label_id columns "
                "[label, stimulus_type, subject_id, task_id, dataset_id]. "
                f"Got shape {base_y.shape}."
            )

        classify_choice = getattr(args, "classify_choice", "disease")
        if classify_choice == "disease":
            target_col = 0
        elif classify_choice == "stimulus":
            target_col = 1
        else:
            raise ValueError(
                f"classify_choice must be 'disease' or 'stimulus', got {classify_choice}."
            )

        groups = OrderedDict()
        for index, row in enumerate(base_y):
            dataset_id = int(row[4])
            subject_id = int(row[2])
            stimulus_id = int(row[1])
            key = (dataset_id, subject_id, stimulus_id)
            groups.setdefault(key, []).append(index)

        averaged_y = []
        for key, indices in groups.items():
            indices = np.asarray(indices, dtype=np.int64)
            labels = base_y[indices]

            target_values = np.unique(labels[:, target_col].astype(np.int64))
            if target_values.size != 1:
                raise ValueError(
                    "Cannot average trials with conflicting target labels. "
                    f"flag={flag}, key=(dataset_id, subject_id, stimulus_id)={key}, "
                    f"target labels={target_values.tolist()}."
                )

            self.group_keys.append(key)
            self.group_indices.append(indices)
            averaged_y.append(labels[0].astype(np.float32, copy=True))

        self.y = np.stack(averaged_y, axis=0).astype(np.float32, copy=False)

        print(
            f"[{self.flag}] ERP trial averaging enabled: "
            f"group by (subject_id, stimulus_type); "
            f"original trials={len(base_dataset)}, averaged trials={len(self.group_indices)}."
        )

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, index):
        indices = self.group_indices[int(index)]
        x_sum = None

        for base_index in indices:
            x, _ = self.base_dataset[int(base_index)]
            x = x.float()
            if x_sum is None:
                x_sum = x.clone()
            else:
                x_sum += x

        x_avg = x_sum / float(len(indices))
        y = torch.from_numpy(self.y[int(index)].astype(np.float32, copy=False))
        return x_avg, y

    def __getattr__(self, name):
        # Delegate metadata such as seq_len, enc_in, num_class, summary(), etc.
        # to the original MultiDataset instance.
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def summary(self) -> str:
        base_summary = self.base_dataset.summary() if hasattr(self.base_dataset, "summary") else ""
        average_summary = (
            f"ERP trial averaging: enabled for {self.flag}.\n"
            f"  Group key: (subject_id, stimulus_type).\n"
            f"  Original trials: {len(self.base_dataset)}.\n"
            f"  Averaged trials: {len(self)}."
        )
        return base_summary + "\n" + average_summary if base_summary else average_summary


class DatasetHomogeneousBatchSampler(Sampler[List[int]]):
    """
    Yield dataset-homogeneous mini-batches with memmap-friendly access.

    Main guarantees:
        1. Every mini-batch contains samples from exactly one dataset.
        2. Samples inside each mini-batch are contiguous in that dataset's
           global-index range whenever possible.
        3. The first epoch keeps fully sequential reading to warm up memmap /
           OS page cache.
        4. From the second epoch onward, contiguous mini-batches are grouped
           into local blocks and the block order is shuffled.
        5. Inside each selected mini-batch, neighboring samples are split into
           small local groups and only the group order is shuffled.

    Compared with sample-level random shuffling, block-level and local-group
    shuffling preserve much better disk locality while still introducing
    randomness both across batches and inside each batch.
    """

    def __init__(
        self,
        dataset: MultiDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        sampling_strategy: str = "proportional",
        sequential_first_epoch: bool = True,
        batch_block_size: int = 8,
        batch_inner_group_size: int = 16,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if sampling_strategy not in ["proportional", "balanced"]:
            raise ValueError("sampling_strategy must be 'proportional' or 'balanced'.")
        if batch_block_size <= 0:
            raise ValueError("batch_block_size must be positive.")
        if batch_inner_group_size <= 0:
            raise ValueError("batch_inner_group_size must be positive.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.sampling_strategy = sampling_strategy
        self.sequential_first_epoch = sequential_first_epoch
        self.batch_block_size = int(batch_block_size)
        self.batch_inner_group_size = int(batch_inner_group_size)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _num_batches_from_n(self, n: int) -> int:
        if self.drop_last:
            return n // self.batch_size
        return int(np.ceil(n / self.batch_size))

    def _make_contiguous_batches(self, indices: np.ndarray) -> List[List[int]]:
        """
        Build contiguous batches from ordered global indices.

        Since MultiDataset allocates one continuous global-index range for each
        underlying dataset, this leads to locally sequential memmap reads.
        """
        n = len(indices)

        if self.drop_last:
            usable_n = (n // self.batch_size) * self.batch_size
        else:
            usable_n = n

        batches: List[List[int]] = []
        for start in range(0, usable_n, self.batch_size):
            batch = indices[start:start + self.batch_size].tolist()
            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                batches.append(batch)

        return batches

    def _make_batch_blocks(self, batches: List[List[int]]) -> List[List[List[int]]]:
        """
        Group neighboring contiguous mini-batches into local blocks.

        The order inside each block is intentionally preserved. This allows the
        OS and storage device to benefit from read-ahead after a block is chosen.
        """
        return [
            batches[start:start + self.batch_block_size]
            for start in range(0, len(batches), self.batch_block_size)
        ]

    @staticmethod
    def _flatten_blocks(blocks: List[List[List[int]]]) -> List[List[int]]:
        return [batch for block in blocks for batch in block]

    def _block_shuffle_batches(
        self,
        batches: List[List[int]],
        rng: np.random.Generator,
    ) -> List[List[int]]:
        """
        Shuffle contiguous mini-batch blocks without shuffling samples inside a
        batch or mini-batches inside a selected block.
        """
        blocks = self._make_batch_blocks(batches)
        rng.shuffle(blocks)
        return self._flatten_blocks(blocks)

    def _shuffle_groups_inside_batch(
        self,
        batch: List[int],
        rng: np.random.Generator,
    ) -> List[int]:
        """
        Shuffle small contiguous groups inside one mini-batch.

        Example for batch_inner_group_size=4:
            [0,1,2,3, 4,5,6,7, 8,9,10,11]
        may become:
            [8,9,10,11, 0,1,2,3, 4,5,6,7]

        Samples remain sequential inside each local group, so memmap locality is
        largely preserved while the token/trial order inside a batch changes.
        """
        groups = [
            batch[start:start + self.batch_inner_group_size]
            for start in range(0, len(batch), self.batch_inner_group_size)
        ]
        rng.shuffle(groups)
        return [index for group in groups for index in group]

    def _shuffle_groups_inside_batches(
        self,
        batches: List[List[int]],
        rng: np.random.Generator,
    ) -> List[List[int]]:
        return [self._shuffle_groups_inside_batch(batch, rng) for batch in batches]

    def _make_dataset_batches(
        self,
        indices: np.ndarray,
        rng: np.random.Generator,
        use_sequential_epoch: bool,
    ) -> List[List[int]]:
        """
        Build one dataset's batches.

        Epoch 0:
            keep fully sequential mini-batch order.

        Later epochs:
            keep each mini-batch contiguous, but shuffle local blocks of
            mini-batches to introduce randomness while limiting random I/O.
        """
        indices = np.asarray(indices, dtype=np.int64)
        batches = self._make_contiguous_batches(indices)

        if self.shuffle and not use_sequential_epoch:
            batches = self._block_shuffle_batches(batches, rng)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        use_sequential_epoch = self.sequential_first_epoch and self.epoch == 0

        if self.sampling_strategy == "proportional":
            all_blocks: List[List[List[int]]] = []

            for dataset_idx, indices in self.dataset.indices_by_dataset.items():
                indices = np.asarray(indices, dtype=np.int64)
                dataset_batches = self._make_contiguous_batches(indices)
                dataset_blocks = self._make_batch_blocks(dataset_batches)
                all_blocks.extend(dataset_blocks)

            # First epoch is fully sequential. From epoch 1 onward, shuffle
            # dataset-local blocks globally; batches inside one block remain
            # adjacent to preserve memmap locality.
            if self.shuffle and not use_sequential_epoch:
                rng.shuffle(all_blocks)

            all_batches = self._flatten_blocks(all_blocks)

        else:  # balanced
            dataset_batches_dict: Dict[int, List[List[int]]] = {}
            batch_counts: List[int] = []

            for dataset_idx, indices in self.dataset.indices_by_dataset.items():
                indices = np.asarray(indices, dtype=np.int64)
                dataset_batches = self._make_contiguous_batches(indices)

                # Preserve the previous balanced fallback for datasets smaller
                # than one full batch.
                if len(dataset_batches) == 0 and len(indices) > 0:
                    replace = len(indices) < self.batch_size
                    batch = rng.choice(
                        indices,
                        size=self.batch_size,
                        replace=replace,
                    ).tolist()
                    dataset_batches = [batch]

                dataset_batches_dict[dataset_idx] = dataset_batches
                batch_counts.append(len(dataset_batches))

            target_batches_per_dataset = max(batch_counts) if len(batch_counts) > 0 else 0
            all_blocks: List[List[List[int]]] = []

            for dataset_idx, dataset_batches in dataset_batches_dict.items():
                if len(dataset_batches) == 0:
                    continue

                # Repeat whole contiguous batches to balance datasets. This is
                # still more memmap-friendly than random sample-level resampling.
                if len(dataset_batches) < target_batches_per_dataset:
                    repeat_times = int(np.ceil(target_batches_per_dataset / len(dataset_batches)))
                    dataset_batches = (dataset_batches * repeat_times)[:target_batches_per_dataset]
                else:
                    dataset_batches = dataset_batches[:target_batches_per_dataset]

                all_blocks.extend(self._make_batch_blocks(dataset_batches))

            if self.shuffle and not use_sequential_epoch:
                rng.shuffle(all_blocks)

            all_batches = self._flatten_blocks(all_blocks)

        # Keep epoch 0 fully sequential for memmap warm-up. From epoch 1
        # onward, add limited intra-batch randomness by shuffling contiguous
        # local groups rather than individual samples.
        if self.shuffle and not use_sequential_epoch:
            all_batches = self._shuffle_groups_inside_batches(all_batches, rng)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        if self.sampling_strategy == "proportional":
            total = 0
            for indices in self.dataset.indices_by_dataset.values():
                total += self._num_batches_from_n(len(indices))
            return total

        batch_counts = []
        for indices in self.dataset.indices_by_dataset.values():
            n = len(indices)
            if self.drop_last:
                batch_counts.append(max(1, n // self.batch_size))
            else:
                batch_counts.append(max(1, int(np.ceil(n / self.batch_size))))

        return max(batch_counts) * len(self.dataset.indices_by_dataset)

