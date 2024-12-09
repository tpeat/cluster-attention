import matplotlib.pyplot as plt
import seaborn as sns
import torch

def get_weights(model, tokenizer):
    text = "Translate English to French: Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Forward pass to get outputs with attention weights
    outputs = model(**inputs, decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]]).to(model.device))

    encoder_attentions = outputs.encoder_attentions
    attention_weights = torch.stack(encoder_attentions)
    return attention_weights

def plot_heads(model, tokenizer, head_groups, plot_name="plots/test.png"):
    text = "Translate English to French: Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Forward pass to get outputs with attention weights
    outputs = model(**inputs, decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]]).to(model.device))

    encoder_attentions = outputs.encoder_attentions  # Attention weights from the encoder
    attention_weights = torch.stack(encoder_attentions)
    layer_idx = 0
    attention_weights = get_weights(model, tokenizer)
    attention_matrices = attention_weights[layer_idx].squeeze(0)

    # Visualize attention matrices for all heads in the layer
    num_heads = attention_matrices.shape[0]
    print("num_heads:", num_heads)
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))

    for i, head_idx in enumerate(range(num_heads)):
        ax = axes[head_idx]
        cluster = head_groups[i]
        heatmap_data = attention_matrices[head_idx].detach().cpu().numpy()
        sns.heatmap(heatmap_data, cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Head {head_idx}, cluster {cluster}")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens")

    print("saving plot")
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()
    
    
def plot_head(model, tokenizer,head_idx, layer_idx, plot_name="test"):

    attention_weights = get_weights(model, tokenizer)
    attention_weights = attention_weights[layer_idx].squeeze(0)
    head_attention = attention_weights[head_idx].detach().cpu().numpy()

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(head_attention, cmap="Blues", cbar=True, annot=False)
    plt.title(f"Attention Head {head_idx}")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.tight_layout()
    
    # Save the plot
    print(f"Saving plot for layer {layer_idx}, head {head_idx} at {plot_name}")
    plt.savefig(plot_name)
    plt.close()

