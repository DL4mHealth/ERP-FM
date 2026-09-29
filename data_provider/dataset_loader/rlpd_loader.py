
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data_provider.dataset_loader.base_loader import BaseLoader


class RLPDLoader(BaseLoader):
    """RLPD loader for X.dat/y.dat/meta.json memmap format."""

    def __init__(self, args, root_path, flag=None, dataset_name="RLPD"):
        super().__init__(args=args, root_path=root_path, flag=flag, dataset_name=dataset_name)

    def _split_subject_ids(self, y_full: np.ndarray, args) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Build subject-independent train/val/test subject lists from y.dat.

        y_full columns are [disease_id, stimulus_type, subject_id, task_id].
        Splitting is stratified by disease_id at the subject level.
        """
        a, b = args.ratio_a, args.ratio_b
        label_col = 0
        subject_col = 2

        subject_ids = np.unique(y_full[:, subject_col]).astype(int)

        subject_to_label: Dict[int, int] = {}
        for subject_id in subject_ids:
            subject_mask = y_full[:, subject_col] == subject_id
            labels = np.unique(y_full[subject_mask, label_col]).astype(int)
            subject_to_label[int(subject_id)] = int(labels[0])

        label_to_subjects: Dict[int, List[int]] = {}
        for subject_id, disease_id in subject_to_label.items():
            label_to_subjects.setdefault(disease_id, []).append(subject_id)

        if args.cross_val in ["fixed", "mccv"]:
            seed = 42 if args.cross_val == "fixed" else int(args.seed)
            rng = random.Random(seed)

            train_ids, val_ids, test_ids = [], [], []
            for disease_id in sorted(label_to_subjects.keys()):
                ids = label_to_subjects[disease_id]
                rng.shuffle(ids)

                n = len(ids)
                train_ids.extend(ids[: int(a * n)])
                val_ids.extend(ids[int(a * n): int(b * n)])
                test_ids.extend(ids[int(b * n):])

            return sorted(subject_ids.tolist()), sorted(train_ids), sorted(val_ids), sorted(test_ids)

        if args.cross_val == "loso":
            all_ids = sorted(subject_ids.tolist())
            test_ids = [all_ids[(int(args.seed) - 41) % len(all_ids)]]
            train_ids = [subject_id for subject_id in all_ids if subject_id not in test_ids]
            val_ids = train_ids
            return all_ids, sorted(train_ids), sorted(val_ids), sorted(test_ids)

        raise ValueError("Invalid cross_val. Please use fixed, mccv, or loso.")
