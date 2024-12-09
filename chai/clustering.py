import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from model import load_pretrained_model

def create_clusters(model, tokenizer, num_clusters):
    text = "Translate English to French: Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Forward pass to get outputs with attention weights
    outputs = model(**inputs, decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]]).to(model.device))

    encoder_attentions = outputs.encoder_attentions  # Attention weights from the encoder
    attention_weights = torch.stack(encoder_attentions)
    flattened_matrices = []

    for layer in range(12):  # num_layers = 12 for T5-base
        for head in range(12):  # num_heads = 12 for T5-base
            attention_matrix = attention_weights[layer, 0, head] 
            flattened_matrices.append(attention_matrix.flatten())  # Flatten
    flattened_matrices = torch.stack(flattened_matrices)

    # Perform K-Means clustering
    scaler = StandardScaler()
    scaled_matrices = scaler.fit_transform(flattened_matrices.detach().cpu().numpy())
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    head_groups = kmeans.fit_predict(scaled_matrices)

    # Display results
    print("Cluster assignments for attention heads:", head_groups)
    return head_groups


def assign_weights_to_clusters(model, head_groups, num_clusters):
    cluster_q_weights = {i: [] for i in range(num_clusters)}
    cluster_k_weights = {i: [] for i in range(num_clusters)}
    cluster_v_weights = {i: [] for i in range(num_clusters)}

    for layer_idx, layer in enumerate(model.encoder.block):
        attn = layer.layer[0].SelfAttention
        n_heads = attn.n_heads
        head_dim = attn.q.weight.shape[0] // n_heads

        q_weight = attn.q.weight.view(n_heads, head_dim, -1)
        k_weight = attn.k.weight.view(n_heads, head_dim, -1)
        v_weight = attn.v.weight.view(n_heads, head_dim, -1)

        for head_idx in range(n_heads):
            cluster_id = head_groups[layer_idx * n_heads + head_idx]  # Flatten index to get cluster ID
            cluster_q_weights[cluster_id].append(q_weight[head_idx].detach().cpu())
            cluster_k_weights[cluster_id].append(k_weight[head_idx].detach().cpu())
            cluster_v_weights[cluster_id].append(v_weight[head_idx].detach().cpu())

    avg_q_weights = {cluster: torch.mean(torch.stack(weights), dim=0) 
                     for cluster, weights in cluster_q_weights.items()}
    avg_k_weights = {cluster: torch.mean(torch.stack(weights), dim=0) 
                     for cluster, weights in cluster_k_weights.items()}
    avg_v_weights = {cluster: torch.mean(torch.stack(weights), dim=0) 
                     for cluster, weights in cluster_v_weights.items()}

    for layer_idx, layer in enumerate(model.encoder.block):
        attn = layer.layer[0].SelfAttention
        n_heads = attn.n_heads
        head_dim = attn.q.weight.shape[0] // n_heads

        q_weight = attn.q.weight.view(n_heads, head_dim, -1)
        k_weight = attn.k.weight.view(n_heads, head_dim, -1)
        v_weight = attn.v.weight.view(n_heads, head_dim, -1)
        new_q_weight = q_weight.clone()
        new_k_weight = k_weight.clone()
        new_v_weight = v_weight.clone()

        for head_idx in range(n_heads):
            cluster_id = head_groups[layer_idx * n_heads + head_idx]  # Flatten index to get cluster ID
            new_q_weight[head_idx] = avg_q_weights[cluster_id].to(q_weight.device)
            new_k_weight[head_idx] = avg_k_weights[cluster_id].to(k_weight.device)
            new_v_weight[head_idx] = avg_v_weights[cluster_id].to(v_weight.device)

        attn.q.weight.data = new_q_weight.reshape(-1, q_weight.shape[-1])
        attn.k.weight.data = new_k_weight.reshape(-1, k_weight.shape[-1])
        attn.v.weight.data = new_v_weight.reshape(-1, v_weight.shape[-1])

    print("Model weights have been reassigned based on cluster averages.")
