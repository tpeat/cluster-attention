import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5ForConditionalGeneration, T5Tokenizer, AlbertModel
from datasets import load_dataset
from tqdm import tqdm
import os

def get_c4_samples(num_samples=1024, max_length=512):
    """Load and preprocess C4 dataset samples."""
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    samples = []
    
    for sample in tqdm(dataset, desc="Loading C4 samples", total=num_samples):
        if len(samples) >= num_samples:
            break
        if len(sample['text'].split()) >= 50:  # Minimum word count
            samples.append(sample['text'])
    
    return samples

def process_batch(model, tokenizer, texts, device='cuda', max_length=512):
    """Process a batch of texts and get encoder attention patterns."""
    inputs = tokenizer(texts, 
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=max_length)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # For T5, we need decoder_input_ids
    decoder_input_ids = torch.zeros((len(texts), 1), dtype=torch.long, device=device)
    
    if isinstance(model, AlbertModel):
        print("albert")
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True
            )
        attention_maps = outputs.attentions
    else: # assum t5
        with torch.no_grad():
            outputs = model(
                **inputs,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True
            )
        attention_maps = outputs.encoder_attentions
    
    # Get encoder self-attention patterns
    # Convert attention patterns to numpy for correlation calculation
    attention_patterns = [layer_attn.cpu().numpy() for layer_attn in attention_maps]
    
    return attention_patterns

def compute_head_correlations(attention_patterns):
    """
    Compute correlations between attention heads.
    attention_patterns: [batch_size, num_heads, seq_len, seq_len]
    """
    batch_size, num_heads, seq_len, _ = attention_patterns.shape
    correlations = np.zeros((num_heads, num_heads))
    
    # For each pair of heads
    for i in range(num_heads):
        for j in range(num_heads):
            # Get attention patterns for both heads across all samples
            head_i_patterns = attention_patterns[:, i, :, :]  # [batch_size, seq_len, seq_len]
            head_j_patterns = attention_patterns[:, j, :, :]
            
            # Flatten patterns for correlation calculation
            head_i_flat = head_i_patterns.reshape(batch_size, -1)  # [batch_size, seq_len*seq_len]
            head_j_flat = head_j_patterns.reshape(batch_size, -1)
            
            # Compute correlation for each sample and average
            sample_correlations = []
            for sample_idx in range(batch_size):
                corr = np.corrcoef(head_i_flat[sample_idx], head_j_flat[sample_idx])[0, 1]
                if not np.isnan(corr):
                    sample_correlations.append(corr)
            
            correlations[i, j] = np.mean(sample_correlations)
    
    return correlations

def plot_token_attention_patterns(attention_patterns, tokens, layer_idx, output_dir):
    """
    Plot token-position attention patterns for a layer.
    attention_patterns: [num_heads, seq_len, seq_len]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Average across attention scores dimension to get activation per token
    # This gives us [num_heads, seq_len]
    activations = attention_patterns.mean(1)
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        activations,
        cmap='magma_r',
        vmin=0,
        vmax=1,
        xticklabels=tokens
    )
    plt.title(f'Layer {layer_idx} Token Attention Patterns\nRows: Heads, Columns: Token Position')
    plt.xlabel('Tokens')
    plt.ylabel('Head')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/layer_{layer_idx}_token_attention.png')
    plt.close()

def plot_correlation_heatmaps(correlations, output_dir, title):
    """Plot correlation heatmaps for T5 encoder layers."""
    os.makedirs(output_dir, exist_ok=True)
    
    # For T5-base, we'll plot all 12 encoder layers in a 3x4 grid
    num_layers = len(correlations)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        sns.heatmap(
            correlations[layer_idx],
            ax=axes[layer_idx],
            cmap='magma_r',
            vmin=0,
            vmax=1,
            xticklabels=range(0, correlations[layer_idx].shape[0], 2),
            yticklabels=range(0, correlations[layer_idx].shape[0], 2)
        )
        
        axes[layer_idx].set_title(f'Layer {layer_idx}')
        axes[layer_idx].set_xlabel('Head ID')
        axes[layer_idx].set_ylabel('Head ID')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlations_{title.lower().replace(" ", "_")}.png')
    plt.close()

def evaluate_on_c4(model,
                  tokenizer,
                  num_samples=1024,
                  batch_size=4,
                  device='cuda'):
    """
    Evaluate T5 encoder attention patterns on C4 dataset and generate correlation plots.
    """
    
    print("Loading C4 samples...")
    samples = get_c4_samples(num_samples)

    os.makedirs("correlation_plots", exist_ok=True)
    os.makedirs("token_attention_plots", exist_ok=True)
    
    # Process samples in batches
    all_layer_correlations = []
    batched_samples = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batched_samples, desc="Processing batches")):
        attention_patterns = process_batch(model, tokenizer, batch, device)

        if batch_idx == 0:
            all_layer_correlations = [[] for _ in range(len(attention_patterns))]
            
            # For the first batch, also generate token attention visualizations
            # Get tokens for the first sample in batch
            inputs = tokenizer(batch[0], 
                             return_tensors="pt",
                             padding=True,
                             truncation=True,
                             max_length=512)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Plot token attention patterns for each layer
            for layer_idx, layer_attn in enumerate(attention_patterns):
                # Take first sample from batch
                sample_attention = layer_attn[0]  # [num_heads, seq_len, seq_len]
                plot_token_attention_patterns(
                    sample_attention,
                    tokens,
                    layer_idx,
                    "token_attention_plots"
                )
        
        # Initialize storage for layer correlations on first batch
        if batch_idx == 0:
            all_layer_correlations = [[] for _ in range(len(attention_patterns))]
        
        # Compute correlations for each layer
        for layer_idx, layer_attn in enumerate(attention_patterns):
            layer_correlations = compute_head_correlations(layer_attn)
            all_layer_correlations[layer_idx].append(layer_correlations)
            
        # Plot first batch correlations
        if batch_idx == 0:
            first_batch_correlations = [compute_head_correlations(layer_attn) 
                                      for layer_attn in attention_patterns]
            plot_correlation_heatmaps(
                first_batch_correlations,
                "t5_plots_single_batch",
                "Single Batch Correlation (T5-base Encoder)"
            )
    
    # Average correlations across all batches
    avg_correlations = []
    for layer_correlations in all_layer_correlations:
        avg_correlations.append(np.mean(layer_correlations, axis=0))
    
    plot_correlation_heatmaps(
        avg_correlations,
        "t5_plots_average",
        "Average Correlation across C4 Samples (T5-base Encoder)"
    )

def prepare_t5(checkpoint='google/t5-base'):
    print(f"Loading model {checkpoint}...")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, output_attentions=True)
    model = model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    return tokenizer, model

def prepare_albert(checkpoint, depth, attention_heads=None, apply_tome=False, margin=None):
    """
    @param depth, int number of hidden layers
    """
    print(f"Loading model {checkpoint}...")
    from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
    tokenizer = AlbertTokenizer.from_pretrained(checkpoint)
    if attention_heads is not None:
        alm2_config = AlbertConfig.from_pretrained(checkpoint, num_hidden_layers=depth, num_attention_heads=attention_heads)
    else:
        # original number of attention heads
        alm2_config = AlbertConfig.from_pretrained(checkpoint, num_hidden_layers=depth)
    model = AlbertModel.from_pretrained(checkpoint, config=alm2_config, attn_implementation='eager')
    if apply_tome:
        apply_patch_to_albert(model, trace_source=False, prop_attn=True, margin=margin, alpha=1.0, use_attn=False)
    return tokenizer, model

if __name__ == "__main__":
    # tokenizer, model = prepare_t5()
    tokenizer, model = prepare_albert('albert-xlarge-v2', depth=48)
    print(model)
    evaluate_on_c4(model, tokenizer)
