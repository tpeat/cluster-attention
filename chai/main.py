from model import load_pretrained_model
from clustering import create_clusters, assign_weights_to_clusters
from plot_results import plot_heads, plot_head


num_clusters = 100 #change to num clusters wanted
modified_checkpoint_path = "modified_checkpoint_per_cluster_80" #path for saving model weights 
local_checkpoint_path="pretrained_checkpoint" #path to local pretrained model checkpoint
def cluster_head_weights(num_clusters=100, model_path="clustered_heads", pretrained_model_path="pretrained_checkpoint"):
    model, tokenizer = load_pretrained_model(pretrained_model_path)
    head_groups = create_clusters(model, tokenizer, num_clusters)
    assign_weights_to_clusters(model, head_groups, num_clusters)
    #Save the updated model
    model.save_pretrained(model_path)
    print(f"Modified checkpoint saved to: {modified_checkpoint_path}")

cluster_head_weights(num_clusters, modified_checkpoint_path,local_checkpoint_path)

#plot a specific attention head
#model, tokenizer = load_pretrained_model()
#plot_head(model, tokenizer, 8, 11, plot_name="plots/cluster-2-head-2.png")
