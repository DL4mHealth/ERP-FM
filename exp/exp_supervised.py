from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    CONFUSION_MATRIX_KEY, EarlyStopping, calculate_subject_level_metrics,
    row_normalized_confusion_matrix,
)

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Exp_Supervised(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.swa = bool(getattr(args, "swa", False))
        self.swa_model = optim.swa_utils.AveragedModel(self.model) if self.swa else None
        if self.swa and hasattr(self.swa_model, "to"):
            self.swa_model.to(self.device)

    def _build_model(self):
        train_data, _ = self._get_data(flag="TRAIN")
        self.args.seq_len = train_data.seq_len
        self.args.enc_in = train_data.enc_in
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

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

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
        raise ValueError(
            f"Invalid classify_choice={classify_choice}. "
            "Expected 'disease' or 'stimulus'."
        )

    @staticmethod
    def _validate_labels(labels, num_class, classify_choice):
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        labels_np = labels_np.astype(np.int64)
        if labels_np.size == 0:
            raise ValueError(f"Empty labels for classify_choice='{classify_choice}'.")
        min_label = int(labels_np.min())
        max_label = int(labels_np.max())
        if min_label < 0 or max_label >= num_class:
            raise ValueError(
                f"Labels for classify_choice='{classify_choice}' must be in [0, {num_class - 1}]. "
                f"Got min={min_label}, max={max_label}."
            )

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
    def _format_metrics(prefix, loss, metrics):
        return (
            f"{prefix} --- Loss: {loss:.5f}, "
            f"Accuracy: {metrics['Accuracy']:.5f}, "
            f"Precision: {metrics['Precision']:.5f}, "
            f"Recall: {metrics['Recall']:.5f}, "
            f"F1: {metrics['F1']:.5f}, "
            f"AUROC: {metrics['AUROC']:.5f}, "
            f"AUPRC: {metrics['AUPRC']:.5f}"
        )

    @staticmethod
    def _format_subject_metrics(prefix, metrics):
        return (
            f"{prefix}, "
            f"Accuracy: {metrics['Accuracy']:.5f}, "
            f"Precision: {metrics['Precision']:.5f}, "
            f"Recall: {metrics['Recall']:.5f}, "
            f"F1: {metrics['F1']:.5f}, "
            f"AUROC: {metrics['AUROC']:.5f}, "
            f"AUPRC: {metrics['AUPRC']:.5f}"
        )

    def _format_eval_results(
        self,
        vali_loss,
        sample_val_metrics,
        subject_val_metrics,
        test_loss,
        sample_test_metrics,
        subject_test_metrics,
    ):
        lines = [
            "Sample-level results:",
            self._format_metrics("Validation results", vali_loss, sample_val_metrics),
            self._format_metrics("Test results", test_loss, sample_test_metrics),
        ]

        if self.args.use_subject_vote:
            lines.extend([
                "Subject-level results after majority voting:",
                self._format_subject_metrics("Validation results", subject_val_metrics),
                self._format_subject_metrics("Test results", subject_test_metrics),
            ])

        return "\n".join(lines)

    def _print_data_summary(self, train_data, vali_data, test_data):
        if hasattr(train_data, "summary"):
            print("Training data summary:\n" + train_data.summary())
            print("\nValidation data summary:\n" + vali_data.summary())
            print("\nTest data summary:\n" + test_data.summary())
            return

        for name, data in [("Training", train_data), ("Validation", vali_data), ("Test", test_data)]:
            print(f"{name} data shape: {data.X.shape}")
            print(f"{name} label shape: {data.y.shape}")

    def vali(self, vali_data, vali_loader, criterion):
        target_col = self._target_col(self.args)
        total_loss, preds, trues, ids, dataset_ids = [], [], [], [], []

        eval_model = self.swa_model if self.swa else self.model
        eval_model.eval()
        with torch.no_grad():
            for batch_x, label_id in vali_loader:
                batch_x = batch_x.float().to(self.device)
                label = label_id[:, target_col].long().to(self.device)

                outputs = eval_model(batch_x, label_id=label_id)
                loss = criterion(outputs, label)

                total_loss.append(loss.item())
                preds.append(outputs.detach().cpu())
                trues.append(label.detach().cpu())
                ids.append(label_id[:, 2].detach().cpu())
                # label_id columns: [disease_id, stimulus_type, subject_id, task_id, dataset_id].
                if label_id.shape[1] > 4:
                    dataset_ids.append(label_id[:, 4].detach().cpu())

        total_loss = float(np.average(total_loss))
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0).numpy().astype(np.int64)
        ids = torch.cat(ids, dim=0).numpy().astype(np.int64)
        self._validate_labels(trues, self.args.num_class, getattr(self.args, "classify_choice", "disease"))

        probs = torch.softmax(preds, dim=1).numpy()
        predictions = np.argmax(probs, axis=1)

        sample_metrics = self._safe_classification_metrics(
            trues=trues,
            predictions=predictions,
            probs=probs,
            num_class=self.args.num_class,
        )

        subject_metrics = None
        if self.args.use_subject_vote:
            # Avoid accidentally merging subjects with the same original subject_id
            # from different datasets, while still computing one global subject-level
            # metric instead of averaging per-dataset metrics.
            if dataset_ids:
                dataset_ids = torch.cat(dataset_ids, dim=0).numpy().astype(np.int64)
                vote_ids = dataset_ids * 1_000_000_000 + ids
            else:
                vote_ids = ids
            subject_metrics = calculate_subject_level_metrics(
                predictions,
                trues,
                vote_ids,
                self.args.num_class,
            )

        self.model.train()
        return total_loss, sample_metrics, subject_metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        self._print_data_summary(train_data, vali_data, test_data)

        path = self._checkpoint_dir(self.args, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=1e-5)
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim,
            T_max=self.args.train_epochs,
        )
        criterion = self._select_criterion()

        total_params = sum(p.numel() for p in self.model.parameters())
        target_col = self._target_col(self.args)
        print(f"\nTotal parameters: {total_params}")
        print(f"Classification target: {getattr(self.args, 'classify_choice', 'disease')} "
              f"(label_id column {target_col})")
        if self.swa:
            print("SWA enabled: validation, testing, and checkpointing use the averaged model.")

        for epoch in range(self.args.train_epochs):
            if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            self.model.train()
            epoch_start = time.time()
            train_loss = []

            print(f"\nEpoch {epoch + 1}/{self.args.train_epochs}:")
            for i, (batch_x, label_id) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label_id[:, target_col].long().to(self.device)
                self._validate_labels(label, self.args.num_class, getattr(self.args, "classify_choice", "disease"))

                outputs = self.model(batch_x, label_id=label_id)
                loss = criterion(outputs, label)
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - epoch_start
                    speed = elapsed / (i + 1)
                    remaining_iters = (self.args.train_epochs - epoch - 1) * train_steps + (train_steps - i - 1)
                    print(
                        f"\tEpoch {epoch + 1}, iter {i + 1}/{train_steps}, "
                        f"loss: {loss.item():.7f}, speed: {speed:.4f}s/iter, "
                        f"ETA: {speed * remaining_iters:.1f}s"
                    )

            if self.swa:
                self.swa_model.update_parameters(self.model)

            train_loss = float(np.average(train_loss))
            vali_loss, sample_val_metrics, subject_val_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, sample_test_metrics, subject_test_metrics = self.vali(test_data, test_loader, criterion)
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch: {epoch + 1}, cost time: {time.time() - epoch_start:.2f}s, "
                f"Steps: {train_steps}, Train Loss: {train_loss:.5f}, LR: {current_lr:.5e}\n"
                + self._format_eval_results(
                    vali_loss,
                    sample_val_metrics,
                    subject_val_metrics,
                    test_loss,
                    sample_test_metrics,
                    subject_test_metrics,
                )
                + "\n"
            )

            early_stopping(
                -sample_val_metrics["F1"],
                self.swa_model if self.swa else self.model,
                path,
                tie_breaker_loss=vali_loss,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step()

        best_model_path = os.path.join(path, "checkpoint.pth")
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.swa_model if self.swa else self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        if test:
            path = self._checkpoint_dir(self.args, setting)
            model_path = os.path.join(path, "checkpoint.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model found at {model_path}")
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")

        total_params = sum(p.numel() for p in self.model.parameters())
        criterion = self._select_criterion()
        vali_loss, sample_val_metrics, subject_val_metrics = self.vali(vali_data, vali_loader, criterion)
        test_loss, sample_test_metrics, subject_test_metrics = self.vali(test_data, test_loader, criterion)

        result_text = self._format_eval_results(
            vali_loss,
            sample_val_metrics,
            subject_val_metrics,
            test_loss,
            sample_test_metrics,
            subject_test_metrics,
        )
        print(result_text + "\n")

        folder_path = self._result_dir(self.args)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "results.txt")
        with open(file_path, "a") as f:
            f.write("Model Setting: " + setting + "\n")
            f.write(result_text + "\n\n")

        return (
            sample_val_metrics,
            subject_val_metrics,
            sample_test_metrics,
            subject_test_metrics,
            total_params,
        )
