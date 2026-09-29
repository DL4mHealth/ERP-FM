import torch

from exp.exp_finetune import Exp_Finetune


class Exp_Probe(Exp_Finetune):
    """
    GPU-based linear probing on top of a frozen ERP-FM encoder.

    This experiment intentionally reuses the fine-tuning pipeline, including:
        - downstream TRAIN / VAL / TEST loaders,
        - pretraining-checkpoint loading,
        - GPU optimization,
        - early stopping,
        - optional SWA,
        - sample-level and subject-level evaluation.

    The only difference from Exp_Finetune is that all pretrained backbone
    parameters are frozen. Only the downstream linear classifier is trainable.

    Unlike exp_pretrain.py's lightweight linear-probe evaluation, this class
    does not use pooled encoder features plus sklearn logistic regression.
    It uses the same flattened encoder representation and nn.Linear classifier
    as downstream fine-tuning, and trains that classifier on GPU.
    """

    def _freeze_backbone(self):
        model_to_freeze = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        for name, param in model_to_freeze.named_parameters():
            param.requires_grad = name.startswith("classifier.")

        trainable = [name for name, param in model_to_freeze.named_parameters() if param.requires_grad]
        frozen = [name for name, param in model_to_freeze.named_parameters() if not param.requires_grad]

        if not trainable:
            raise RuntimeError(
                "Probe mode did not find any trainable classifier parameters. "
                "Check that the model creates self.classifier for task_name='probe'."
            )

        print("Probe mode: frozen pretrained backbone; training classifier only.")
        print(f"Trainable parameters: {trainable}")
        print(f"Frozen parameter tensors: {len(frozen)}")

    def _load_pretrained_checkpoint(self):
        super()._load_pretrained_checkpoint()
        self._freeze_backbone()

    def _select_optimizer(self):
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("Probe mode found no trainable parameters.")
        return torch.optim.AdamW(trainable_parameters, lr=self.args.learning_rate)
