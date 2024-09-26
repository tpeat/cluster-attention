import math
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import FeedForwardNetwork
from .pos_encoding import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        output = self.fc_out(x)
        return output