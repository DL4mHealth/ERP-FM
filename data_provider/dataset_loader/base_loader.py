"""
Memory-mapped ERP dataset loader.

Expected dataset folder structure:
    <root_path>/X.dat
    <root_path>/y.dat
    <root_path>/meta.json

X.dat:
    memory-mapped array with shape (N, T, C)

y.dat:
    memory-mapped array with shape (N, 4), columns:
        [disease_id, stimulus_type, subject_id, task_id]

The loader keeps X.dat as np.memmap and only reads one trial at a time in
__getitem__. This avoids loading the full N x T x C array into RAM.
"""

import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_single_trial(x: np.ndarray) -> np.ndarray:
    """
    Normalize one ERP trial channel-wise along time.

    This matches normalize_batch_ts behavior without loading the full dataset:
        mean/std are computed over the time dimension for each channel.
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + 1e-8)


class BaseLoader(Dataset):
    """
    Generic memory-mapped ERP loader for disease classification.

    This class returns labels as [disease_id, stimulus_type, subject_id].
    The multi-dataset wrapper will later convert them into [disease_id, stimulus_type, subject_id, dataset_id].
    """

    def __init__(self, args, root_path, flag=None, dataset_name: str = "ERP"):
        self.args = args
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.flag = self._normalize_flag(flag)
        self.no_normalize = getattr(args, "no_normalize", False)

        self.x_path = os.path.join(root_path, "X.dat")
        self.y_path = os.path.join(root_path, "y.dat")
        self.meta_path = os.path.join(root_path, "meta.json")

        for path in [self.x_path, self.y_path, self.meta_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{dataset_name}: missing required file: {path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.N = int(self.meta["N"])
        self.T = int(self.meta["T"])
        self.C = int(self.meta["C"])
        self.sample_rate = self.meta.get("SAMPLE_RATE", None)
        self.tmin = self.meta.get("TMIN", None)
        self.tmax = self.meta.get("TMAX", None)
        self.baseline = self.meta.get("BASELINE", None)
        self.montage = self.meta.get("MONTAGE", None)
        self.channels = list(self.meta.get("CHANNELS", []))

        if len(self.channels) > 0 and len(self.channels) != self.C:
            raise ValueError(
                f"{dataset_name}: meta.json C={self.C}, but len(CHANNELS)={len(self.channels)}."
            )

        # Keep X as memmap. This does not load the full array into memory.
        self.X = np.memmap(self.x_path, dtype=np.float32, mode="r", shape=(self.N, self.T, self.C))

        # y is small. We still open it through memmap and then materialize it for fast splitting/indexing.
        self.y_full_memmap = np.memmap(self.y_path, dtype=np.float32, mode="r", shape=(self.N, 4))
        self.y_full = np.asarray(self.y_full_memmap).astype(np.float32, copy=True)

        self.all_ids, self.train_ids, self.val_ids, self.test_ids = self._split_subject_ids(self.y_full, args)
        ids = self._ids_for_flag(self.flag)

        subject_mask = np.isin(self.y_full[:, 2], np.asarray(ids, dtype=np.float32))
        self.indices = np.nonzero(subject_mask)[0].astype(np.int64)

        # The label exposed to the multi-dataset wrapper is [disease_id, stimulus_type, subject_id, task_id].
        self.y = self.y_full[self.indices][:, [0, 1, 2, 3]].astype(np.float32, copy=True)
        self.max_seq_len = self.T
        self.num_channels = self.C
        self.enc_in = self.C
        self.num_class = len(np.unique(self.y[:, 0]))

        self.dataset_metadata = {
            "dataset_name": self.dataset_name,
            "N": self.N,
            "T": self.T,
            "C": self.C,
            "sample_rate": self.sample_rate,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "baseline": self.baseline,
            "montage": self.montage,
            "channels": self.channels,
            "labels": self.meta.get("LABELS", {}),
        }

        print(
            f"[{self.dataset_name}] flag={self.flag}, subjects={len(ids)}, "
            f"trials={len(self.indices)}, montage={self.montage}, C={self.C}"
        )

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

    def _ids_for_flag(self, flag: str) -> List[int]:
        if flag == "TRAIN":
            print(f"[{self.dataset_name}] train ids: {self.train_ids}")
            return self.train_ids
        if flag == "VAL":
            print(f"[{self.dataset_name}] val ids: {self.val_ids}")
            return self.val_ids
        if flag == "TEST":
            print(f"[{self.dataset_name}] test ids: {self.test_ids}")
            return self.test_ids
        if flag == "PRETRAIN":
            print(f"[{self.dataset_name}] all ids: {self.all_ids}")
            return self.all_ids
        raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")

    def _split_subject_ids(self, y_full: np.ndarray, args) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Dataset-specific subject split logic.
        Must return (all_ids, train_ids, val_ids, test_ids).
        y_full columns are [disease_id, stimulus_type, subject_id, task_id].
        Splitting is stratified by disease_id at the subject level.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        raw_index = int(self.indices[index])

        # Read only one trial from the memory-mapped X.dat.
        x = np.asarray(self.X[raw_index], dtype=np.float32)
        if not self.no_normalize:
            x = _normalize_single_trial(x)
        else:
            # Make a small contiguous array for torch.from_numpy.
            x = np.array(x, dtype=np.float32, copy=True)

        y = self.y[index]
        return torch.from_numpy(x), torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self):
        return len(self.indices)
