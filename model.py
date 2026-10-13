import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(B, T, self.heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)

        x = attn @ V
        x = x.transpose(1, 2).contiguous().view(B, T, D)

        return self.out(x)


# Feed Forward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# Transformer Model
class MotionTransformer(nn.Module):
    def __init__(self, input_dim=66, d_model=128, num_layers=3, heads=4, num_classes=6 ,seq_len =50):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional Encoding
        # self.pos_enc = PositionalEncoding(d_model)
        self.pos_enc = PositionalEncoding(d_model,seq_len=seq_len)

        # Encoder stack
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        x = self.input_proj(x)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        out = self.classifier(x)

        return out