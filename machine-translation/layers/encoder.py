import math
import torch.nn as nn
from .attention import make_attn_fn
from .ffn import FeedForwardNetwork
from .pos_encoding import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_fn):
        """
        @param attn_fn: Already initialized attention function
        """
        super(EncoderLayer, self).__init__()
        # TODO: add support for max_cluster count
        self.self_attn = make_attn_fn(attn_fn, d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, attn_fn):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, attn_fn) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, src, mask):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        return x


