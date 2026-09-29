from collections import OrderedDict
import os
import torch
import torch.nn as nn

from exp.exp_supervised import Exp_Supervised


class Exp_Finetune(Exp_Supervised):
    """
    Downstream fine-tuning.

    This class is intentionally close to Exp_Supervised. The only difference is
    that it first loads a checkpoint from args.checkpoints_path, usually produced
    by exp_pretrain.py. The pretraining reconstruction head and downstream
    classifier are task-specific and are not loaded from the pretraining checkpoint.
    """

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

    def _load_pretrained_checkpoint(self):
        ckpt_path = self.args.checkpoints_path
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "checkpoint.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No pretraining checkpoint found at {ckpt_path}")

        print(f"Loading pretraining checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        checkpoint = self._strip_prefixes(checkpoint)

        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_state = model_to_load.state_dict()

        filtered = OrderedDict()
        skipped = []
        for k, v in checkpoint.items():
            # Pretraining-only modules and downstream classifier are intentionally skipped.
            if (
                k.startswith("reconstruction_head.")
                or k.startswith("mask_token")
                or k.startswith("mae_decoder.")
                or k.startswith("classifier.")
            ):
                skipped.append(k)
                continue
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)

        missing, unexpected = model_to_load.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} matched parameters from pretraining checkpoint.")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        if skipped:
            print(f"Skipped {len(skipped)} checkpoint keys due to task mismatch or shape mismatch.")

    def train(self, setting):
        self._load_pretrained_checkpoint()
        return super().train(setting)
