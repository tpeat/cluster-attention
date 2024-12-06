from model import load_pretrained_model
from clustering import create_clusters, assign_weights_to_clusters
from plotting import plot_heads

def cluster_head_weights():
    model, tokenizer = load_pretrained_model()
    num_clusters = 80
    head_groups = create_clusters(model, tokenizer, num_clusters)

    assign_weights_to_clusters(model, head_groups, num_clusters)

    # Save the updated model
    modified_checkpoint_path = "modified_checkpoint_per_cluster_80"
    model.save_pretrained(modified_checkpoint_path)
    print(f"Modified checkpoint saved to: {modified_checkpoint_path}")

cluster_head_weights()
