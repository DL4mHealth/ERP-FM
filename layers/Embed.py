import copy
import math
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
import mne
from torch.nn.utils import weight_norm

from layers.Augmentation import get_augmentation
from data_provider.uea import bandpass_filter_func


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        sin_part = torch.sin(position * div_term)
        cos_part = torch.cos(position * div_term)

        pe[:, 0::2] = sin_part[:, :pe[:, 0::2].shape[1]]
        pe[:, 1::2] = cos_part[:, :pe[:, 1::2].shape[1]]

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):  # (batch_size, seq_len, enc_in)
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)  # c_in is seq_length here
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_length)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)  # (batch_size, enc_in, d_model)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class ShallowNetEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout):
        super().__init__()

        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, d_model, (1, 25), (1, 1)),
            nn.Conv2d(d_model, d_model, (c_in, 1), (1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, 4), (1, 2)),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(d_model, d_model, (1, 1), stride=(1, 1)),
        )

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1).unsqueeze(1)  # Shape becomes (B, 1, C, T)
        x = self.shallow_net(x)
        x = self.projection(x)
        # Rearrange the output to match the Transformer input format (B, patch_num, d_model)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class EEGDeformerConv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with a DataParallel-safe max-norm weight constraint.

    Do not modify ``self.weight`` in-place inside ``forward``. Under
    ``torch.nn.DataParallel``, broadcast parameters can be views tracked by
    autograd, and in-place renorm can trigger:

        RuntimeError: Output ... of BroadcastBackward0 is a view and its base
        or another view of its base has been modified inplace.

    Instead, build a normalized temporary weight tensor and pass it to
    ``F.conv2d``. Gradients still flow back to ``self.weight`` while avoiding
    the in-place parameter update.
    """

    def __init__(self, *args, max_norm=1.0, do_weight_norm=True, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.do_weight_norm:
            weight = torch.renorm(
                self.weight,
                p=2,
                dim=0,
                maxnorm=self.max_norm,
            )
        else:
            weight = self.weight

        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class EEGDeformerEmbedding(nn.Module):
    """
    Shallow convolutional feature encoder used by EEG-Deformer.

    Input:
        x: [B, T, C]

    Output:
        tokens: [B, D, F]
            D = d_model convolutional feature tokens / filters
            F = temporal feature dimension after temporal/spatial convolution and pooling
    """

    def __init__(self, num_channels, num_time, temporal_kernel, d_model, dropout=0.1):
        super().__init__()
        temporal_kernel = int(temporal_kernel)
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1

        self.num_channels = int(num_channels)
        self.num_time = int(num_time)
        self.temporal_kernel = temporal_kernel
        self.num_kernel = int(d_model)
        self.feature_dim = max(1, self.num_time // 2)

        self.cnn_encoder = nn.Sequential(
            EEGDeformerConv2dWithConstraint(
                1,
                self.num_kernel,
                kernel_size=(1, self.temporal_kernel),
                padding=(0, self.temporal_kernel // 2),
                max_norm=2.0,
                bias=True,
            ),
            EEGDeformerConv2dWithConstraint(
                self.num_kernel,
                self.num_kernel,
                kernel_size=(self.num_channels, 1),
                padding=0,
                max_norm=2.0,
                bias=True,
            ),
            nn.BatchNorm2d(self.num_kernel),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_kernel, self.feature_dim) * 0.02)

    def forward(self, x):
        # [B, T, C] -> [B, 1, C, T]
        x = x.permute(0, 2, 1).unsqueeze(1).contiguous()
        x = self.cnn_encoder(x)  # [B, K, 1, F]
        x = x.squeeze(2)         # [B, K, F]

        # For rare odd-length inputs, MaxPool2d may produce floor(T/2). Keep the
        # positional embedding aligned with the actual runtime feature length.
        feature_len = x.shape[-1]
        if feature_len <= self.pos_embedding.shape[-1]:
            pos = self.pos_embedding[:, :, :feature_len]
        else:
            pos = F.interpolate(
                self.pos_embedding,
                size=feature_len,
                mode="linear",
                align_corners=False,
            )
        return x + pos


class CrossChannelTokenEmbedding(nn.Module):  # (batch_size, 1, enc_in, seq_len)
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )

    def forward(self, x):
        x = self.tokenConv(x)
        return x  # (batch_size, d_model, 1, patch_num)


class UpDimensionChannelEmbedding(nn.Module):  # B x C x T
    def __init__(self, c_in, t_in, u_dim, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.u_dim = u_dim
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=u_dim,
            kernel_size=3,
            padding=padding,
            bias=False,
        )
        self.fc = nn.Linear(t_in, d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)  # B x u_dim x T
        x = self.fc(x)  # B x u_dim x d_model
        return x


class MedformerEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        patch_len_list,
        stride_list,
        dropout,
        augmentation=["none"],
    ):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list
        self.enc_in = enc_in
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]

        linear_layers = [
            CrossChannelTokenEmbedding(
                c_in=enc_in,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        self.value_embeddings = nn.ModuleList(linear_layers)
        self.position_embedding_t = PositionalEmbedding(d_model=d_model)
        self.position_embedding_c = PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation])
        self.routers = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for _ in self.patch_len_list])
        self.learnable_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in self.patch_len_list])

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)

        x_list = []
        for padding, value_embedding, router in zip(self.paddings, self.value_embeddings, self.routers):
            x_copy = x.clone()
            # per granularity augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new = self.augmentation[aug_idx](x_copy)
            # add positional embedding to tag each channel
            x_new = x_new + self.position_embedding_c(x_new)
            # temporal dimension
            x_new = padding(x_new).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
            x_new = value_embedding(x_new)  # (batch_size, d_model, 1, patch_num)
            x_new = x_new.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
            # append router
            router = router.expand(x_new.size(0), 1, x_new.size(-1))  # (B,1,D)
            x_new = torch.cat([x_new, router], dim=1)  # (batch_size, patch_num+1, d_model)
            x_list.append(x_new)

        x = [
            x + cxt + self.position_embedding_t(x)
            for x, cxt in zip(x_list, self.learnable_embeddings)
        ]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
        return x


class MultiResolutionData(nn.Module):
    def __init__(self, enc_in, resolution_list, stride_list):
        super().__init__()
        self.paddings = nn.ModuleList([nn.ReplicationPad1d((0, stride)) for stride in stride_list])

        self.multi_res = nn.ModuleList([
            nn.Conv1d(
                in_channels=enc_in,
                out_channels=enc_in,
                kernel_size=res,
                stride=res,
                padding=0,
                padding_mode='circular')
            for res in resolution_list
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_list = []
        for l in range(len(self.multi_res)):
            out = self.paddings[l](x)
            out = self.multi_res[l](out)
            x_list.append(out)
        return x_list


class FrequencyEmbedding(nn.Module):
    """
    Frequency-domain embedding used by MedGNN.

    The original implementation used ``nn.Linear(...).to(torch.cfloat)``, which
    creates complex-valued trainable parameters. Complex parameters work in some
    single-GPU settings, but they are fragile under ``torch.nn.DataParallel`` on
    several PyTorch/CUDA stacks. To keep MedGNN multi-GPU compatible, this layer
    keeps all trainable parameters real-valued and only uses complex tensors as
    temporary FFT activations inside ``forward``.

    Input tensors in ``x_list``:  [B, C, L_r]
    Output tensors:              [B, C, d_model]
    """

    def __init__(self, d_model, res_len, augmentation=["none"]):
        super().__init__()
        self.d_model = int(d_model)
        self.freq_in_dims = [int(res / 2) + 1 for res in res_len]
        self.freq_out_dim = int(self.d_model / 2) + 1

        self.real_embeddings = nn.ModuleList([
            nn.Linear(freq_in_dim, self.freq_out_dim)
            for freq_in_dim in self.freq_in_dims
        ])
        self.imag_embeddings = nn.ModuleList([
            nn.Linear(freq_in_dim, self.freq_out_dim)
            for freq_in_dim in self.freq_in_dims
        ])

        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

    def forward(self, x_list):
        x_out = []
        for l, x in enumerate(x_list):
            x_freq = torch.fft.rfft(x, dim=-1)

            real = self.real_embeddings[l](x_freq.real)
            imag = self.imag_embeddings[l](x_freq.imag)
            out_freq = torch.complex(real, imag)

            out = torch.fft.irfft(out_freq, dim=-1, n=self.d_model)

            aug_idx = random.randint(0, len(self.augmentation) - 1)
            out = self.augmentation[aug_idx](out)
            x_out.append(out)

        return x_out


def get_eeg_coords_from_montage(channel_names, montage_name="standard_1005"):
    montage = mne.channels.make_standard_montage(montage_name)
    pos_dict = montage.get_positions()["ch_pos"]
    name_map = {name.lower(): name for name in pos_dict.keys()}

    coords = np.zeros((len(channel_names), 3), dtype=np.float32)
    for i, ch_name in enumerate(channel_names):
        coords[i] = pos_dict[name_map[ch_name.lower()]]

    return coords


class Electrode3DEmbedding(nn.Module):
    """
    Variant B:
        Normalized xyz coordinates
        -> scaled sinusoidal encoding
        -> MLP projection
        -> RMSNorm

    This is an improved version of the original fixed sinusoidal embedding.

    Input:
        coords: [C, 3]

    Output:
        embedding: [C, D]
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = None,
        coord_scale: float = math.pi,
        eps: float = 1e-8,
    ):
        super().__init__()

        hidden_dim = hidden_dim or d_model

        self.d_model = d_model
        self.coord_scale = coord_scale
        self.eps = eps

        self.d_x = d_model // 3
        self.d_y = d_model // 3
        self.d_z = d_model - self.d_x - self.d_y

        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.RMSNorm(d_model),
        )

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates while preserving relative geometry.
        """
        scale = coords.abs().amax().clamp_min(self.eps)
        coords = coords / scale
        return coords * self.coord_scale

    @staticmethod
    def _encode_axis(pos: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Args:
            pos: [C]
            dim: embedding dimension assigned to one coordinate axis

        Returns:
            embedding: [C, dim]
        """
        C = pos.shape[0]
        device = pos.device
        dtype = pos.dtype

        pos = pos.unsqueeze(1)  # [C, 1]

        # Support odd embedding dimensions.
        num_freqs = (dim + 1) // 2

        j = torch.arange(num_freqs,device=device,dtype=dtype,)
        div_term = torch.exp(-math.log(10000.0) * (2 * j) / max(dim, 1))
        angle = pos * div_term  # [C, num_freqs]
        emb = torch.zeros(C, dim, device=device, dtype=dtype)

        num_sin = emb[:, 0::2].shape[1]
        num_cos = emb[:, 1::2].shape[1]

        emb[:, 0::2] = torch.sin(angle[:, :num_sin])
        emb[:, 1::2] = torch.cos(angle[:, :num_cos])

        return emb

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [C, 3]

        Returns:
            embedding: [C, D]
        """
        coords = self._normalize_coords(coords)

        pe_x = self._encode_axis(coords[:, 0], self.d_x)
        pe_y = self._encode_axis(coords[:, 1], self.d_y)
        pe_z = self._encode_axis(coords[:, 2], self.d_z)

        emb = torch.cat([pe_x, pe_y, pe_z], dim=-1)

        return self.proj(emb)


class ERPEmbedding(nn.Module):
    def __init__(
        self,
        d_model,
        patch_len,
        stride,
        channel_names_by_id,
        montage_by_id,
        augmentation=("none",),
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride

        # Uni-variate patch embedding.
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        nn.init.xavier_uniform_(self.value_embedding.weight)

        # Temporal patch position embedding.
        self.position_embedding = PositionalEmbedding(d_model)

        # Channel 3D coordinate embedding.
        self.electrode_embedding = Electrode3DEmbedding(d_model)

        # dataset_id -> buffer name.
        self.coord_buffer_names = {}
        for dataset_id, channel_names in channel_names_by_id.items():
            dataset_id = int(dataset_id)
            montage_name = montage_by_id.get(dataset_id, "standard_1005")

            coords = get_eeg_coords_from_montage(
                channel_names=channel_names,
                montage_name=montage_name,
            )

            buffer_name = f"electrode_coords_dataset_{dataset_id}"
            self.register_buffer(
                buffer_name,
                torch.tensor(coords, dtype=torch.float32),
                # Electrode coordinates are deterministic runtime metadata for
                # the currently loaded dataset, not learned model parameters.
                # They are rebuilt from --training_dataset during downstream
                # linear probing and therefore are not stored in checkpoints.
                persistent=False,
            )
            self.coord_buffer_names[dataset_id] = buffer_name

        self.augmentation = nn.ModuleList(
            [get_augmentation(aug, patch_len) for aug in augmentation]
        )

    @staticmethod
    def _apply_aug_module(aug_module, x):
        """Apply an augmentation even when ERPEmbedding is in eval mode.

        Probe training keeps the frozen backbone in eval mode, but input
        augmentation should still be active when explicitly requested. Some
        augmentation modules check their own ``training`` flag, so temporarily
        switch only the selected augmentation module to train mode and restore
        its previous state immediately after use.
        """
        was_training = aug_module.training
        aug_module.train(True)
        x = aug_module(x)
        aug_module.train(was_training)
        return x

    def _pad_to_stride(self, x):
        """
        x: (B, C, L)
        """
        L = x.size(-1)
        if L < self.patch_len:
            pad_right = self.patch_len - L
        else:
            remainder = (L - self.patch_len) % self.stride
            pad_right = 0 if remainder == 0 else self.stride - remainder

        return F.pad(x, (0, pad_right), mode="replicate")

    def forward(self, x, dataset_id, apply_augmentation=False):
        """
        x: (B, L, C)
        dataset_id: (B,), all values are assumed to be the same in one batch
        apply_augmentation: augmentation is applied only when this flag is
            explicitly set to True by the caller.

        return: (B, C * N, d_model)
        """
        # Because each batch comes from one dataset.
        dataset_id = int(dataset_id[0].item())

        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1).contiguous()

        if apply_augmentation and len(self.augmentation) > 0:
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x = self._apply_aug_module(self.augmentation[aug_idx], x)

        x = self._pad_to_stride(x)

        # (B, C, L) -> (B, C, N, patch_len)
        x = x.unfold(-1, self.patch_len, self.stride)
        B, C, N, _ = x.shape

        # Patch value embedding.
        # (B, C, N, patch_len) -> (B*C, N, patch_len) -> (B*C, N, D)
        x = rearrange(x, "b c n l -> (b c) n l")
        x = self.value_embedding(x)

        # Temporal positional embedding.
        x = x + self.position_embedding(x)

        # Restore channel dimension.
        # (B*C, N, D) -> (B, C, N, D)
        x = rearrange(x, "(b c) n d -> b c n d", b=B, c=C)

        # Add 3D electrode embedding.
        coords = getattr(self, self.coord_buffer_names[dataset_id]).to(
            device=x.device,
            dtype=x.dtype,
        )  # (C, 3)

        channel_pe = self.electrode_embedding(coords)  # (C, D)

        # (C, D) -> (1, C, 1, D)
        x = x + channel_pe.unsqueeze(0).unsqueeze(2)

        # (B, C, N, D) -> (B, C*N, D)
        x = rearrange(x, "b c n d -> b (c n) d")

        return x
