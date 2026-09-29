import torch.nn as nn
from layers.Activation_Family import swiglu


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, 2 * d_ff, 1)  # 2*d_ff for SwiGLU split
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.conv1(y.transpose(-1, 1))
        y = swiglu(y)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Lightweight MAE decoder block.

    It is a self-attention Transformer block over the restored full token sequence
    consisting of encoded visible tokens and learnable mask tokens. This follows
    the MAE-style decoder used by ST-EEGFormer more closely than a single linear
    reconstruction head.
    """

    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, 2 * d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.conv1(y.transpose(-1, 1))
        y = swiglu(y)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
