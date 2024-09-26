import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose to get dimensions batch_size * num_heads * seq_len * d_k
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, v)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.d_k)
        
        # Final linear layer
        output = self.fc_out(output)
        
        return output
