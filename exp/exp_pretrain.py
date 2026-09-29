from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    CONFUSION_MATRIX_KEY, calculate_subject_level_metrics,
    row_normalized_confusion_matrix,
)
from utils import eval_protocols

import os
import time
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Exp_Pretrain(Exp_Basic):
    """
    Basic masked reconstruction pretraining + downstream linear probing.

    Pretraining:
        PRETRAIN split may contain multiple datasets.
        The model reconstructs canonical c_out-channel temporal patches from
        a masked input. No early stopping or SWA is used.

    Linear probing:
        TRAIN/VAL/TEST are all from one downstream dataset specified by
        args.training_dataset. Encoder representations are extracted and a
        scikit-learn logistic regression classifier is fitted on TRAIN and
        evaluated on VAL/TEST.
    """

    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        if self.args.is_training == 1:
            pretrain_data, _ = self._get_data(flag="PRETRAIN")
            pretrain_channel_names = dict(getattr(self.args, "channel_names_by_id", {}))
            pretrain_montage = dict(getattr(self.args, "montage_by_id", {}))
            pretrain_dataset_id_to_name = dict(getattr(self.args, "dataset_id_to_name", {}))
        else:
            # linear probe or test-only mode, no need to load pretraining data or metadata
            pretrain_data = None
            pretrain_channel_names = {}
            pretrain_montage = {}
            pretrain_dataset_id_to_name = {}

        train_data, _ = self._get_data(flag="TRAIN")
        downstream_channel_names = dict(getattr(self.args, "channel_names_by_id", {}))
        downstream_montage = dict(getattr(self.args, "montage_by_id", {}))
        downstream_dataset_id_to_name = dict(getattr(self.args, "dataset_id_to_name", {}))

        self.args.channel_names_by_id = {**pretrain_channel_names, **downstream_channel_names}
        self.args.montage_by_id = {**pretrain_montage, **downstream_montage}
        self.args.dataset_id_to_name = {**pretrain_dataset_id_to_name, **downstream_dataset_id_to_name}

        # Important: seq_len and enc_in always come from the unique downstream
        # training dataset, even during pretraining. PRETRAIN may contain multiple
        # heterogeneous datasets, so its max_T/max_C should not define fixed-shape
        # downstream modules such as classifiers.
        self.args.seq_len = train_data.seq_len
        self.args.enc_in = train_data.enc_in
        self.args.max_seq_len = train_data.seq_len
        self.args.max_num_channels = train_data.enc_in

        target_col = self._target_col(self.args)
        target_labels = train_data.y[:, target_col].astype(np.int64)
        unique_labels = np.unique(target_labels)
        expected_labels = np.arange(len(unique_labels))
        if not np.array_equal(unique_labels, expected_labels):
            raise ValueError(
                f"Labels for classify_choice='{getattr(self.args, 'classify_choice', 'disease')}' "
                f"must be contiguous integers from 0 to num_class - 1. "
                f"Got unique labels {unique_labels.tolist()}."
            )
        self.args.num_class = len(unique_labels)

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    @staticmethod
    def _checkpoint_dir(args, setting):
        return os.path.join(
            "./checkpoints",
            args.method,
            args.task_name,
            args.model,
            args.model_id,
            setting,
        )

    @staticmethod
    def _strip_prefixes(state_dict):
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            if k == "n_averaged":
                continue
            while k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v
        return cleaned

    def _load_pretrain_init_checkpoint(self):
        init_path = getattr(self.args, "pretrain_init_path", "")
        # empty or whitespace-only path means no initialization checkpoint, start from random weights
        if init_path is None or str(init_path).strip() == "":
            return

        ckpt_path = str(init_path)
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "checkpoint.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No MAE pretraining initialization checkpoint found at {ckpt_path}")

        print(f"\n\nLoading MAE pretraining initialization checkpoint from {ckpt_path}")
        print("Second-stage MAE pretraining initialization: loading only ERPEmbedding + Encoder.")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        checkpoint = self._strip_prefixes(checkpoint)

        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_state = model_to_load.state_dict()

        # For second-stage pretraining, reuse only the representation backbone from
        # the previous MAE checkpoint. The MAE-specific decoder, reconstruction head,
        # mask token, and any downstream classifier are intentionally reinitialized.
        load_prefixes = ("enc_embedding.", "encoder.")
        filtered = OrderedDict()
        skipped_by_component = []
        skipped_by_shape = []
        for k, v in checkpoint.items():
            if not k.startswith(load_prefixes):
                skipped_by_component.append(k)
                continue
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped_by_shape.append(k)

        missing, unexpected = model_to_load.load_state_dict(filtered, strict=False)
        missing_backbone = [k for k in missing if k.startswith(load_prefixes)]
        missing_reinitialized = [k for k in missing if not k.startswith(load_prefixes)]

        print(f"Loaded {len(filtered)} ERPEmbedding/Encoder parameters into MAE pretraining model.")
        if missing_backbone:
            print(f"Missing ERPEmbedding/Encoder keys: {missing_backbone}")
        if missing_reinitialized:
            print(
                f"Reinitialized {len(missing_reinitialized)} non-backbone keys "
                "(decoder, mask token, reconstruction head, classifier, etc.)."
            )
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        if skipped_by_shape:
            print(f"Skipped {len(skipped_by_shape)} ERPEmbedding/Encoder keys due to name or shape mismatch.")
        if skipped_by_component:
            print(
                f"Skipped {len(skipped_by_component)} non-backbone checkpoint keys "
                "because second-stage pretraining only loads ERPEmbedding + Encoder."
            )

    @staticmethod
    def _result_dir(args):
        return os.path.join(
            "./results",
            args.method,
            args.task_name,
            args.model,
            args.model_id,
        )

    @staticmethod
    def _target_col(args):
        classify_choice = getattr(args, "classify_choice", "disease")
        if classify_choice == "disease":
            return 0
        if classify_choice == "stimulus":
            return 1
        raise ValueError(f"Invalid classify_choice={classify_choice}. Expected 'disease' or 'stimulus'.")

    @staticmethod
    def _safe_classification_metrics(trues, predictions, probs, num_class):
        metrics = {
            "Accuracy": accuracy_score(trues, predictions),
            CONFUSION_MATRIX_KEY: row_normalized_confusion_matrix(
                trues, predictions, num_class
            ),
        }
        unique_labels = np.unique(trues)
        if len(unique_labels) < 2:
            metrics.update({"Precision": -1, "Recall": -1, "F1": -1, "AUROC": -1, "AUPRC": -1})
            return metrics

        metrics["Precision"] = precision_score(trues, predictions, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(trues, predictions, average="macro", zero_division=0)
        metrics["F1"] = f1_score(trues, predictions, average="macro", zero_division=0)

        trues_onehot = torch.nn.functional.one_hot(
            torch.as_tensor(trues, dtype=torch.long),
            num_classes=num_class,
        ).float().numpy()
        try:
            if num_class == 2:
                metrics["AUROC"] = roc_auc_score(trues, probs[:, 1])
            else:
                metrics["AUROC"] = roc_auc_score(trues_onehot, probs, multi_class="ovr", average="macro")
        except ValueError:
            metrics["AUROC"] = -1
        try:
            metrics["AUPRC"] = average_precision_score(trues_onehot, probs, average="macro")
        except ValueError:
            metrics["AUPRC"] = -1
        return metrics

    @staticmethod
    def _format_metrics(prefix, metrics):
        return (
            f"{prefix} --- "
            f"Accuracy: {metrics['Accuracy']:.5f}, "
            f"Precision: {metrics['Precision']:.5f}, "
            f"Recall: {metrics['Recall']:.5f}, "
            f"F1: {metrics['F1']:.5f}, "
            f"AUROC: {metrics['AUROC']:.5f}, "
            f"AUPRC: {metrics['AUPRC']:.5f}"
        )

    def _resolve_batch_mask_strategy(self):
        """Resolve one MAE masking strategy for the current global batch."""
        strategy = getattr(self.args, "mask_strategy", "random").lower()
        if strategy == "mixed":
            return random.choice(["random", "temporal", "spatial"])
        return strategy

    def _validate_mask_ratio(self, mask_ratio, name="mask_ratio"):
        mask_ratio = float(mask_ratio)
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {mask_ratio}.")
        return mask_ratio

    def _resolve_mask_ratio_bounds(self):
        """Resolve --mask_ratio as either a fixed value or a linear schedule.

        Supported command-line forms:
            --mask_ratio 0.5       -> fixed 0.5, internally (0.5, 0.5)
            --mask_ratio 0.5 0.5   -> fixed 0.5
            --mask_ratio 0.4 0.8   -> linearly increase from 0.4 to 0.8
        """
        mask_ratio = getattr(self.args, "mask_ratio", [0.5, 0.5])
        if isinstance(mask_ratio, (float, int)):
            values = [float(mask_ratio)]
        else:
            values = [float(x) for x in mask_ratio]

        if len(values) == 1:
            start = end = values[0]
        elif len(values) == 2:
            start, end = values
        else:
            raise ValueError(
                f"--mask_ratio expects one value or two values, got {values}. "
                "Examples: --mask_ratio 0.5 or --mask_ratio 0.4 0.8."
            )

        start = self._validate_mask_ratio(start, "mask_ratio[0]")
        end = self._validate_mask_ratio(end, "mask_ratio[1]")
        return start, end

    def _resolve_epoch_mask_ratio(self, epoch):
        """Resolve the MAE mask ratio for one epoch by linear interpolation."""
        start, end = self._resolve_mask_ratio_bounds()
        if self.args.train_epochs <= 1 or start == end:
            return start

        progress = float(epoch) / float(self.args.train_epochs - 1)
        return start + (end - start) * progress

    def _pretrain_loss(self, reconstruction, target, mask):
        """Compute the selected MAE reconstruction loss on masked patches only.

        Supported losses:
            mse:       L2 mean squared error
            mae:       L1 mean absolute error
            smooth_l1: Huber-style Smooth L1 loss controlled by args.huber_beta
        """
        mae_loss = getattr(self.args, "mae_loss", "mse").lower()

        if mae_loss == "mse":
            elementwise_loss = (reconstruction - target) ** 2
        elif mae_loss == "mae":
            elementwise_loss = torch.abs(reconstruction - target)
        elif mae_loss == "smooth_l1":
            huber_beta = float(getattr(self.args, "huber_beta", 1.0))
            if huber_beta <= 0:
                raise ValueError(f"huber_beta must be > 0 for smooth_l1 loss, got {huber_beta}.")
            elementwise_loss = F.smooth_l1_loss(
                reconstruction,
                target,
                reduction="none",
                beta=huber_beta,
            )
        else:
            raise ValueError(
                f"Unsupported mae_loss={mae_loss}. Expected one of: mse, mae, smooth_l1."
            )

        if mask is not None and mask.any():
            elementwise_loss = elementwise_loss[
                mask.unsqueeze(-1).expand_as(elementwise_loss)
            ]
        return elementwise_loss.mean()

    @staticmethod
    def _unwrap_model(model):
        # Handles plain model and DataParallel.
        if hasattr(model, "module"):
            model = model.module
        if isinstance(model, nn.DataParallel):
            model = model.module
        return model

    def encode(self, loader):
        labels, subject_ids, dataset_ids, reprs = [], [], [], []
        eval_model = self.model
        base_model = self._unwrap_model(eval_model)
        was_training = eval_model.training
        eval_model.eval()
        target_col = self._target_col(self.args)

        with torch.no_grad():
            for batch_x, label_id in loader:
                batch_x = batch_x.float().to(self.device)
                rep = base_model.encode(batch_x, label_id=label_id)
                reprs.append(rep.detach().cpu().float().numpy())
                labels.append(label_id[:, target_col].detach().cpu().numpy())
                subject_ids.append(label_id[:, 2].detach().cpu().numpy())
                dataset_ids.append(label_id[:, 4].detach().cpu().numpy())

        eval_model.train(was_training)
        return (
            np.concatenate(reprs, axis=0),
            np.concatenate(labels, axis=0).astype(np.int64),
            np.concatenate(subject_ids, axis=0).astype(np.int64),
            np.concatenate(dataset_ids, axis=0).astype(np.int64),
        )

    def linear_probe(self, train_loader, eval_loader):
        train_repr, train_labels, _, _ = self.encode(train_loader)
        eval_repr, eval_labels, eval_subject_ids, eval_dataset_ids = self.encode(eval_loader)

        clf = eval_protocols.fit_lr(train_repr, train_labels)
        probs = clf.predict_proba(eval_repr)
        predictions = probs.argmax(axis=1)

        sample_metrics = self._safe_classification_metrics(
            trues=eval_labels,
            predictions=predictions,
            probs=probs,
            num_class=self.args.num_class,
        )

        subject_metrics = None
        if self.args.use_subject_vote:
            vote_ids = eval_dataset_ids * 1_000_000_000 + eval_subject_ids
            subject_metrics = calculate_subject_level_metrics(
                predictions,
                eval_labels,
                vote_ids,
                self.args.num_class,
            )
        return sample_metrics, subject_metrics

    def _print_probe_results(self, sample_val, subject_val, sample_test, subject_test):
        print("Linear probing results:")
        print(self._format_metrics("Validation", sample_val))
        print(self._format_metrics("Test", sample_test))
        if self.args.use_subject_vote:
            print("Subject-level results after majority voting:")
            print(self._format_metrics("Validation", subject_val))
            print(self._format_metrics("Test", subject_test))

    def train(self, setting):
        pretrain_data, pretrain_loader = self._get_data(flag="PRETRAIN")
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        print("Pretraining data summary:\n" + pretrain_data.summary())
        print("\nDownstream training data summary:\n" + train_data.summary())
        print("\nDownstream validation data summary:\n" + vali_data.summary())
        print("\nDownstream test data summary:\n" + test_data.summary())

        path = self._checkpoint_dir(self.args, setting)
        os.makedirs(path, exist_ok=True)

        # load pretraining initialization checkpoint if specified, before any training steps or linear probing
        self._load_pretrain_init_checkpoint()

        optimizer = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.train_epochs)
        train_steps = len(pretrain_loader)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters: {total_params}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Patch length: {self.args.patch_len}")
        mask_ratio_start, mask_ratio_end = self._resolve_mask_ratio_bounds()
        if mask_ratio_start == mask_ratio_end:
            print(f"Mask ratio: {mask_ratio_start}")
        else:
            print(
                f"Mask ratio schedule: linear from {mask_ratio_start} "
                f"to {mask_ratio_end} across {self.args.train_epochs} epochs"
            )
        print(f"Mask strategy: {getattr(self.args, 'mask_strategy', 'random')}")
        print(f"MAE reconstruction loss: {getattr(self.args, 'mae_loss', 'mse')}")
        if getattr(self.args, "mae_loss", "mse").lower() == "smooth_l1":
            print(f"Smooth L1 beta: {getattr(self.args, 'huber_beta', 1.0)}")
        print("ERP-FM pretraining uses MAE masked reconstruction and has no early stopping.")

        for epoch in range(self.args.train_epochs):
            if hasattr(pretrain_loader, "batch_sampler") and hasattr(pretrain_loader.batch_sampler, "set_epoch"):
                pretrain_loader.batch_sampler.set_epoch(epoch)

            self.model.train()
            current_mask_ratio = self._resolve_epoch_mask_ratio(epoch)
            epoch_start = time.time()
            losses = []
            mask_strategy_counts = {"random": 0, "temporal": 0, "spatial": 0}
            print(f"\nEpoch {epoch + 1}/{self.args.train_epochs}: mask_ratio={current_mask_ratio:.4f}")

            for i, (batch_x, label_id) in enumerate(pretrain_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_mask_strategy = self._resolve_batch_mask_strategy()
                mask_strategy_counts[batch_mask_strategy] += 1
                reconstruction, target, mask = self.model(
                    batch_x,
                    label_id=label_id,
                    mask_strategy=batch_mask_strategy,
                    mask_ratio=current_mask_ratio,
                )
                loss = self._pretrain_loss(reconstruction, target, mask)
                losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - epoch_start
                    speed = elapsed / (i + 1)
                    remaining_iters = (self.args.train_epochs - epoch - 1) * train_steps + (train_steps - i - 1)
                    print(
                        f"\tEpoch {epoch + 1}, iter {i + 1}/{train_steps}, "
                        f"reconstruction loss: {loss.item():.7f}, speed: {speed:.4f}s/iter, "
                        f"ETA: {speed * remaining_iters:.1f}s"
                    )

            scheduler.step()
            avg_loss = float(np.average(losses))
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch: {epoch + 1}, cost time: {time.time() - epoch_start:.2f}s, "
                f"Steps: {train_steps}, Mask Ratio: {current_mask_ratio:.4f}, "
                f"Reconstruction Loss: {avg_loss:.6f}, LR: {current_lr:.5e}"
            )
            print(f"Mask strategy counts: {mask_strategy_counts}")

            print("Linear probing on downstream dataset...")
            sample_val, subject_val = self.linear_probe(train_loader, vali_loader)
            sample_test, subject_test = self.linear_probe(train_loader, test_loader)
            self._print_probe_results(sample_val, subject_val, sample_test, subject_test)

            torch.save(self.model.state_dict(), os.path.join(path, "checkpoint.pth"))
            print(f"Saved pretraining checkpoint to {os.path.join(path, 'checkpoint.pth')}")

        return self.model

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        if test:
            path = self._checkpoint_dir(self.args, setting)
            model_path = os.path.join(path, "checkpoint.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model found at {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pretraining model from {model_path}")

        total_params = sum(p.numel() for p in self.model.parameters())
        sample_val, subject_val = self.linear_probe(train_loader, vali_loader)
        sample_test, subject_test = self.linear_probe(train_loader, test_loader)
        self._print_probe_results(sample_val, subject_val, sample_test, subject_test)

        folder_path = self._result_dir(self.args)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "results.txt")
        with open(file_path, "a") as f:
            f.write("Model Setting: " + setting + "\n")
            f.write(self._format_metrics("Validation", sample_val) + "\n")
            f.write(self._format_metrics("Test", sample_test) + "\n\n")

        return sample_val, subject_val, sample_test, subject_test, total_params
