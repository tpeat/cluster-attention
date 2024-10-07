import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, **kwargs):
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


# Define Spherical K-Means Clustering
class SphericalKMeans(nn.Module):
    def __init__(self, num_clusters, d_model, device='cuda'):
        super(SphericalKMeans, self).__init__()
        self.num_clusters = num_clusters
        self.d_model = d_model
        self.device = device
        self.centroids = nn.Parameter(torch.randn(num_clusters, d_model).to(device))
        self.centroids.data = F.normalize(self.centroids.data, p=2, dim=1)

    def forward(self, vectors):
        vectors_norm = F.normalize(vectors, p=2, dim=1)
        similarity = torch.matmul(vectors_norm, self.centroids.t())  # (N, num_clusters)
        assignments = torch.argmax(similarity, dim=1)
        return assignments

# Define Bipartite Matching Attention
class BipartiteMatchingAttention(nn.Module):
    def __init__(self, d_model, nhead, num_clusters, dropout=0.1, device='cuda'):
        super(BipartiteMatchingAttention, self).__init__()
        self.num_clusters = num_clusters
        self.d_model = d_model
        self.nhead = nhead
        self.device = device

        self.query_cluster = SphericalKMeans(num_clusters, d_model, device)
        self.key_cluster = SphericalKMeans(num_clusters, d_model, device)

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None):
        L, N, E = query.size()
        S = key.size(0)

        Q = self.linear_q(query)  # (L, N, E)
        K = self.linear_k(key)    # (S, N, E)
        V = self.linear_v(value)  # (S, N, E)

        Q_reshaped = Q.permute(1, 0, 2).contiguous().view(N*L, E)  # (N*L, E)
        K_reshaped = K.permute(1, 0, 2).contiguous().view(N*S, E)  # (N*S, E)

        Q_clusters = self.query_cluster(Q_reshaped)  # (N*L,)
        K_clusters = self.key_cluster(K_reshaped)    # (N*S,)

        Q_clusters = Q_clusters.view(N, L)  # (N, L)
        K_clusters = K_clusters.view(N, S)  # (N, S)

        cluster_mask = torch.zeros(N, L, S).to(self.device)  # (N, L, S)
        for n in range(N):
            for l in range(L):
                cluster_id = Q_clusters[n, l]
                cluster_mask[n, l, :] = (K_clusters[n] == cluster_id).float()

        cluster_mask = cluster_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)  # (N, nhead, L, S)

        Q = Q.permute(1, 0, 2).contiguous().view(N, L, self.nhead, E // self.nhead).transpose(1, 2)  # (N, nhead, L, E')
        K = K.permute(1, 0, 2).contiguous().view(N, S, self.nhead, E // self.nhead).transpose(1, 2)  # (N, nhead, S, E')
        V = V.permute(1, 0, 2).contiguous().view(N, S, self.nhead, E // self.nhead).transpose(1, 2)  # (N, nhead, S, E')

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(E // self.nhead)  # (N, nhead, L, S)
        scores = scores.masked_fill(cluster_mask == 0, float('-inf'))  # Mask out irrelevant keys
        attn_weights = F.softmax(scores, dim=-1)  # (N, nhead, L, S)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)  # (N, nhead, L, E')

        attn_output = attn_output.transpose(1, 2).contiguous().view(N, L, E)  # (N, L, E)
        attn_output = attn_output.permute(1, 0, 2).contiguous()  # (L, N, E)

        out = self.out_proj(attn_output)  # (L, N, E)
        out = self.dropout(out)
        out = self.layer_norm(query + out)  # Residual connection and layer norm

        return out


def make_attn_fn(name, d_model, num_heads, **kwargs):
    if name == "baseline":
        return MultiHeadAttention(d_model, num_heads)
    elif name == "bipartite":
        return BipartiteMatchingAttention(d_model, num_heads, kwargs['num_clusters'])
    else:
        print("Invalid Attention name")