import matplotlib.pyplot as plt
import seaborn as sns



def plot_heads(attention_weights, head_groups, plot_name="plots/test.png"):
    layer_idx = 0
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
    
    
def plot_head(attention_weights):
    pass
