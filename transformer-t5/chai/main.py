from model import load_pretrained_model
from clustering import create_clusters, assign_weights_to_clusters
from plot_results import plot_heads, plot_head

def cluster_head_weights():
    model, tokenizer = load_pretrained_model()
    num_clusters = 100
    head_groups = create_clusters(model, tokenizer, num_clusters)
    assign_weights_to_clusters(model, head_groups, num_clusters)
    #Save the updated model
    modified_checkpoint_path = "modified_checkpoint_per_cluster_80"
    model.save_pretrained(modified_checkpoint_path)
    print(f"Modified checkpoint saved to: {modified_checkpoint_path}")

cluster_head_weights()


#plot a specific attention head
#model, tokenizer = load_pretrained_model()
#plot_head(model, tokenizer, 8, 11, plot_name="plots/cluster-2-head-2.png")

