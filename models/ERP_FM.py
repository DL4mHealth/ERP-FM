import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ERP_FM_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import ERPEmbedding
from einops import rearrange


def compute_patch_num(seq_len, patch_len, stride):
    """Compute patch number after the same right padding used by ERPEmbedding."""
    if seq_len < patch_len:
        pad_right = patch_len - seq_len
    else:
        remainder = (seq_len - patch_len) % stride
        pad_right = 0 if remainder == 0 else stride - remainder
    padded_len = seq_len + pad_right
    return (padded_len - patch_len) // stride + 1


class Model(nn.Module):
    """
    ERP-FM.

    Supported task_name:
        supervised: fully supervised training from scratch.
        finetune:   supervised fine-tuning after loading a pre-trained checkpoint.
        probe:      GPU-based linear probing with a frozen encoder and trainable classifier.
        pretrain:   MAE-style masked raw-patch reconstruction pretraining.

    Pretraining uses an asymmetric MAE design:
        1. ERPEmbedding converts EEG into full electrode-patch tokens.
        2. The selected masking strategy keeps only visible tokens for the encoder.
        3. Encoded visible tokens are restored with learnable mask tokens.
        4. A lightweight Transformer MAE decoder reconstructs raw EEG patches.

    Fine-tuning / supervised classification only use ERPEmbedding + Encoder + classifier.
    The MAE decoder and reconstruction head are discarded when loading checkpoints.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        mask_ratio = getattr(configs, "mask_ratio", 0.5)
        if isinstance(mask_ratio, (list, tuple)):
            self.mask_ratio = float(mask_ratio[0])
        else:
            self.mask_ratio = float(mask_ratio)
        self.mask_strategy = getattr(configs, "mask_strategy", "random").lower()
        if self.mask_strategy not in ["random", "temporal", "spatial", "mixed"]:
            raise ValueError(
                f"Unsupported mask_strategy={self.mask_strategy}. "
                "Expected one of: random, temporal, spatial, mixed."
            )

        self.use_augmentation = bool(getattr(configs, "use_augmentation", False))
        augmentations = getattr(configs, "augmentations", "none")
        if isinstance(augmentations, str):
            augmentations = [aug.strip() for aug in augmentations.split(",") if aug.strip()]
        if not augmentations:
            augmentations = ["none"]

        # ERPEmbedding returns electrode-patch tokens:
        # (B, T, C) -> (B, C*N, D).
        self.enc_embedding = ERPEmbedding(
            configs.d_model,
            configs.patch_len,
            self.stride,
            configs.channel_names_by_id,
            configs.montage_by_id,
            augmentations,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.RMSNorm(configs.d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        if self.task_name == "pretrain":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, configs.d_model))
            nn.init.normal_(self.mask_token, std=0.02)

            self.mae_decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for _ in range(configs.d_layers)
                ],
                norm_layer=nn.RMSNorm(configs.d_model),
            )
            self.reconstruction_head = nn.Linear(configs.d_model, self.patch_len)

        if self.task_name in ["supervised", "finetune", "probe"]:
            patch_num = compute_patch_num(self.seq_len, self.patch_len, self.stride)
            input_dim = configs.d_model * self.enc_in * patch_num
            self.classifier = nn.Linear(input_dim, configs.num_class)

    def train(self, mode: bool = True):
        """
        Keep the frozen backbone deterministic in GPU probe mode.

        Exp_Supervised.train() calls model.train() at the beginning of every
        epoch and after validation. For a strict linear probe, the frozen
        ERPEmbedding and Encoder must remain in eval mode so their dropout
        behavior does not change the extracted representations. Only the
        downstream classifier follows the requested training mode.
        """
        super().train(mode)
        if self.task_name == "probe":
            self.enc_embedding.eval()
            self.encoder.eval()
            self.classifier.train(mode)
        return self

    def _pad_to_stride(self, x):
        """x: (B, C, T)"""
        L = x.size(-1)
        if L < self.patch_len:
            pad_right = self.patch_len - L
        else:
            remainder = (L - self.patch_len) % self.stride
            pad_right = 0 if remainder == 0 else self.stride - remainder
        return F.pad(x, (0, pad_right), mode="replicate")

    def _patchify_original_target(self, x):
        """
        Build raw reconstruction targets in the same token order as ERPEmbedding.

        Args:
            x: raw EEG, shape (B, T, C)

        Returns:
            target: raw patches, shape (B, C*N, patch_len)
        """
        x = x.permute(0, 2, 1).contiguous()  # (B, C, T)
        x = self._pad_to_stride(x)
        x = x.unfold(-1, self.patch_len, self.stride)  # (B, C, N, patch_len)
        return rearrange(x, "b c n p -> b (c n) p")

    def _token_pos_embedding(self, x_enc, dataset_id):
        """
        Build full token positional encoding matching ERPEmbedding's token order.

        It combines temporal patch PE and 3D electrode PE:
            pos: (B, C*N, D)
        """
        dataset_id_int = int(dataset_id[0].item())
        B, T, C = x_enc.shape
        patch_num = compute_patch_num(T, self.patch_len, self.stride)
        device = x_enc.device
        dtype = x_enc.dtype

        # Temporal patch PE: (1, N, D). A single dummy sequence is enough
        # because PositionalEmbedding is independent of the batch/channel count.
        dummy = torch.zeros(1, patch_num, self.d_model, device=device, dtype=dtype)
        temporal_pe = self.enc_embedding.position_embedding(dummy).to(device=device, dtype=dtype)
        temporal_pe = temporal_pe.view(1, 1, patch_num, self.d_model)

        # Channel 3D PE: (C, D)
        coords = getattr(self.enc_embedding, self.enc_embedding.coord_buffer_names[dataset_id_int]).to(
            device=device,
            dtype=dtype,
        )
        channel_pe = self.enc_embedding.electrode_embedding(coords).view(1, C, 1, self.d_model)

        pos = temporal_pe + channel_pe  # (1, C, N, D)
        pos = pos.expand(B, -1, -1, -1)
        return rearrange(pos, "b c n d -> b (c n) d")

    @staticmethod
    def _random_masking(x, mask_ratio):
        """
        Per-sample random masking using the original MAE visible-token order.

        Args:
            x: (B, L, D)
            mask_ratio: float

        Returns:
            x_keep:      visible tokens in random shuffle order, (B, len_keep, D)
            mask:        (B, L), False is visible and True is masked
            ids_restore: indices for restoring [visible, mask] tokens to original order
        """
        B, L, D = x.shape
        len_keep = max(1, int(L * (1 - mask_ratio)))

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_keep = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()

        return x_keep, mask, ids_restore

    @staticmethod
    def _mask_from_boolean(x, mask):
        """
        Gather visible tokens in random MAE order from a precomputed mask.

        Args:
            x:    full tokens, (B, L, D)
            mask: boolean mask in original token order, (B, L).
                  False is visible and True is masked.

        Returns:
            x_keep:      visible tokens in random shuffle order, (B, len_keep, D)
            mask:        unchanged boolean mask in original token order, (B, L)
            ids_restore: indices for restoring [visible, mask] tokens to original order
        """
        B, L, D = x.shape
        visible_counts = (~mask).sum(dim=1)
        len_keep = int(visible_counts[0].item())
        if not torch.all(visible_counts == len_keep):
            raise ValueError("All samples in one batch must keep the same number of visible tokens.")
        if len_keep <= 0:
            raise ValueError("Masking strategy removed all tokens. At least one visible token is required.")

        # Visible tokens come first, masked tokens come second. Random noise
        # shuffles both partitions, preserving the original MAE random order.
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(mask.to(dtype=noise.dtype) * 2.0 + noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_keep = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )
        return x_keep, mask, ids_restore

    @staticmethod
    def _structured_masking(x, mask_ratio, num_channels, num_patches, strategy):
        """
        Structured MAE masking for flattened channel-major tokens.

        Token layout before flattening:
            (B, C, P, D)

        temporal:
            randomly mask selected temporal patch indices across all channels.

        spatial:
            randomly mask selected channels across all temporal patches.
        """
        B, L, _ = x.shape
        if L != num_channels * num_patches:
            raise ValueError(
                f"Token length mismatch: L={L}, C={num_channels}, P={num_patches}."
            )

        if strategy == "temporal":
            axis_len = num_patches
        elif strategy == "spatial":
            axis_len = num_channels
        else:
            raise ValueError(f"Unsupported structured mask strategy: {strategy}.")

        # Structured masking is impossible when the selected axis has only one
        # element because MAE must retain at least one visible token. Fall back
        # to ordinary random masking for this rare edge case.
        if axis_len <= 1:
            return Model._random_masking(x, mask_ratio)

        if mask_ratio <= 0:
            num_mask_axis = 0
        else:
            num_mask_axis = max(1, int(axis_len * mask_ratio))
            num_mask_axis = min(axis_len - 1, num_mask_axis)

        axis_noise = torch.rand(B, axis_len, device=x.device)
        masked_axis_ids = torch.argsort(axis_noise, dim=1)[:, :num_mask_axis]
        axis_mask = torch.zeros(B, axis_len, device=x.device, dtype=torch.bool)
        if num_mask_axis > 0:
            axis_mask.scatter_(dim=1, index=masked_axis_ids, value=True)

        if strategy == "temporal":
            # (B, P) -> (B, C, P) -> (B, C*P)
            mask = axis_mask.unsqueeze(1).expand(-1, num_channels, -1).reshape(B, L)
        else:
            # (B, C) -> (B, C, P) -> (B, C*P)
            mask = axis_mask.unsqueeze(-1).expand(-1, -1, num_patches).reshape(B, L)

        return Model._mask_from_boolean(x, mask)

    def _mask_tokens(self, x, mask_ratio, num_channels, num_patches, mask_strategy=None):
        """Apply the requested MAE masking strategy."""
        strategy = (mask_strategy or self.mask_strategy).lower()
        if strategy == "mixed":
            # Direct model calls remain supported. Exp_Pretrain resolves mixed
            # before DataParallel forward so all GPU replicas use one strategy.
            strategy_id = int(torch.randint(0, 3, (1,), device=x.device).item())
            strategy = ["random", "temporal", "spatial"][strategy_id]

        if strategy == "random":
            return self._random_masking(x, mask_ratio)
        if strategy in ["temporal", "spatial"]:
            return self._structured_masking(
                x,
                mask_ratio,
                num_channels=num_channels,
                num_patches=num_patches,
                strategy=strategy,
            )
        raise ValueError(
            f"Unsupported mask_strategy={strategy}. Expected one of: random, temporal, spatial, mixed."
        )

    @staticmethod
    def _pool_tokens(enc_out):
        """Mean-max pooling over token dimension for fixed-size sample representations."""
        mean_out = enc_out.mean(dim=1)
        max_out = enc_out.max(dim=1).values
        return torch.cat([mean_out, max_out], dim=-1)

    def _encode_tokens(self, x_enc, dataset_id, token_mask=None, apply_augmentation=False):
        # token_mask is kept only for backward compatibility. The MAE pretraining
        # path uses visible-only encoding through _encode_visible_tokens().
        tokens = self.enc_embedding(
            x_enc,
            dataset_id=dataset_id,
            apply_augmentation=apply_augmentation,
        )  # (B, C*N, D)
        enc_out, _ = self.encoder(tokens, attn_mask=None)
        return enc_out

    def _encode_visible_tokens(self, x_enc, dataset_id, mask_ratio, mask_strategy=None):
        # MAE reconstruction uses the clean input. The selected masking strategy
        # below is the only corruption used by this branch.
        tokens = self.enc_embedding(
            x_enc,
            dataset_id=dataset_id,
            apply_augmentation=False,
        )  # (B, C*N, D)
        _, T, C = x_enc.shape
        patch_num = compute_patch_num(T, self.patch_len, self.stride)
        visible_tokens, mask, ids_restore = self._mask_tokens(
            tokens,
            mask_ratio,
            num_channels=C,
            num_patches=patch_num,
            mask_strategy=mask_strategy,
        )
        enc_visible, _ = self.encoder(visible_tokens, attn_mask=None)
        return enc_visible, mask, ids_restore, tokens.shape[1]

    def _mae_decode(self, enc_visible, ids_restore, full_pos):
        """
        Restore full token order and reconstruct raw patches.

        Args:
            enc_visible: (B, L_keep, D), encoded visible tokens in random shuffle order
            ids_restore: (B, L_full), indices for restoring the original token order
            full_pos: (B, L_full, D)

        Returns:
            pred: (B, L_full, patch_len)
        """
        B, L_full, D = full_pos.shape
        num_mask = L_full - enc_visible.shape[1]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        x_ = torch.cat([enc_visible, mask_tokens], dim=1)
        x_full = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )

        x_full = x_full + full_pos
        x_full, _ = self.mae_decoder(x_full, attn_mask=None)
        return self.reconstruction_head(x_full)

    def encode(self, x_enc, label_id=None, dataset_id=None):
        """
        Return fixed-dimensional sample representations for pretraining stage linear probing.
        This uses only ERPEmbedding + Encoder.
        """
        if dataset_id is None:
            dataset_id = label_id[:, 4]
        enc_out = self._encode_tokens(
            x_enc,
            dataset_id=dataset_id,
            apply_augmentation=False,
        )
        # return enc_out.reshape(enc_out.shape[0], -1)  # (B, C*N*D)
        return self._pool_tokens(enc_out)

    def supervised(self, x_enc, dataset_id):
        # Augmentation is disabled by default. It is applied only when
        # --use_augmentation is explicitly set and the corresponding downstream
        # branch is in training mode. Pretraining never uses augmentation.
        if self.task_name == "probe":
            apply_augmentation = self.use_augmentation and self.classifier.training
        else:
            apply_augmentation = self.use_augmentation and self.training

        enc_out = self._encode_tokens(
            x_enc,
            dataset_id=dataset_id,
            apply_augmentation=apply_augmentation,
        )
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        return self.classifier(output)

    def pretrain(self, x_enc, dataset_id, mask_strategy=None, mask_ratio=None):
        """
        MAE-style masked raw-patch reconstruction.

        Returns:
            reconstruction: (B, C*N, patch_len)
            target:         (B, C*N, patch_len)
            mask:           (B, C*N), True means the patch contributes to loss
        """
        target = self._patchify_original_target(x_enc)
        full_pos = self._token_pos_embedding(x_enc, dataset_id=dataset_id)
        effective_mask_ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        enc_visible, mask, ids_restore, _ = self._encode_visible_tokens(
            x_enc,
            dataset_id=dataset_id,
            mask_ratio=effective_mask_ratio,
            mask_strategy=mask_strategy,
        )
        reconstruction = self._mae_decode(enc_visible, ids_restore, full_pos)
        return reconstruction, target, mask

    def forward(self, x_enc, label_id=None, mask_strategy=None, mask_ratio=None):
        """
        label_id columns:
            [disease_id, stimulus_type, subject_id, task_id, dataset_id]
        """
        dataset_id = label_id[:, 4]
        if self.task_name in ["supervised", "finetune", "probe"]:
            return self.supervised(x_enc, dataset_id=dataset_id)
        if self.task_name == "pretrain":
            return self.pretrain(
                x_enc,
                dataset_id=dataset_id,
                mask_strategy=mask_strategy,
                mask_ratio=mask_ratio,
            )
        raise ValueError(f"Unsupported task_name: {self.task_name}")
