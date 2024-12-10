import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5ForConditionalGeneration, T5Tokenizer, AlbertModel
from datasets import load_dataset
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_c4_samples(num_samples=1024, max_length=512):
    # load c4 dataset samples with minimum word count
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    samples = []
    
    for sample in tqdm(dataset, desc="Loading C4 samples", total=num_samples):
        if len(samples) >= num_samples:
            break
        if len(sample['text'].split()) >= 50:
            samples.append(sample['text'])
    
    return samples

def process_batch(model, tokenizer, texts, device='cuda', max_length=512):
    # get encoder attention patterns for a batch of texts
    inputs = tokenizer(texts, 
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=max_length)
    
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    decoder_input_ids = torch.zeros((len(texts), 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        if isinstance(model, AlbertModel):
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True
            )
            attention_maps = outputs.attentions
        else:
            outputs = model(
                **inputs,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True
            )
            attention_maps = outputs.encoder_attentions
    
    return [layer_attn.detach().cpu().numpy() for layer_attn in attention_maps]

def compute_head_correlations(attention_patterns):
    # compute correlations between attention heads
    batch_size, num_heads, seq_len, _ = attention_patterns.shape
    correlations = np.zeros((num_heads, num_heads))
    
    for i in range(num_heads):
        for j in range(num_heads):
            head_i_patterns = attention_patterns[:, i, :, :]
            head_j_patterns = attention_patterns[:, j, :, :]
            head_i_flat = head_i_patterns.reshape(batch_size, -1)
            head_j_flat = head_j_patterns.reshape(batch_size, -1)
            
            sample_correlations = []
            for sample_idx in range(batch_size):
                corr = np.corrcoef(head_i_flat[sample_idx], head_j_flat[sample_idx])[0, 1]
                if not np.isnan(corr):
                    sample_correlations.append(corr)
            
            correlations[i, j] = np.mean(sample_correlations)
    
    return correlations

def plot_token_attention_patterns(attention_patterns, tokens, layer_idx, output_dir):
    # plot attention patterns for each token
    os.makedirs(output_dir, exist_ok=True)
    activations = attention_patterns.mean(1)
    
    formatted_tokens = []
    for token in tokens:
        if token.startswith('▁'):
            formatted_tokens.append(token)
        else:
            formatted_tokens.append(f"<{token}>")
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        activations,
        cmap='magma_r',
        vmin=0,
        vmax=1,
        xticklabels=formatted_tokens
    )
    plt.title(f'Layer {layer_idx} Token Attention Patterns\nRows: Heads, Columns: Token Position')
    plt.xlabel('Tokens')
    plt.ylabel('Head')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_{layer_idx}_token_attention.png')
    plt.close()

def plot_correlation_heatmaps(correlations, output_dir, title):
    # plot correlation heatmaps for encoder layers
    os.makedirs(output_dir, exist_ok=True)
    num_layers = len(correlations)
    nrows = int(np.ceil(np.sqrt(num_layers)))
    ncols = int(np.ceil(num_layers / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
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
    
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlations_{title.lower().replace(" ", "_")}.png')
    plt.close()

def evaluate_on_c4(model, tokenizer, num_samples=1024, batch_size=4, device='cuda'):
    # evaluate attention patterns on c4 dataset
    samples = get_c4_samples(num_samples)
    os.makedirs("correlation_plots", exist_ok=True)
    os.makedirs("token_attention_plots", exist_ok=True)
    
    batched_samples = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    all_layer_correlations = []
    
    for batch_idx, batch in enumerate(tqdm(batched_samples, desc="Processing batches")):
        attention_patterns = process_batch(model, tokenizer, batch, device)

        if batch_idx == 0:
            all_layer_correlations = [[] for _ in range(len(attention_patterns))]
            inputs = tokenizer(batch[0], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            for layer_idx, layer_attn in enumerate(attention_patterns):
                sample_attention = layer_attn[0]
                plot_token_attention_patterns(sample_attention, tokens, layer_idx, "token_attention_plots")
            
            first_batch_correlations = [compute_head_correlations(layer_attn) for layer_attn in attention_patterns]
            plot_correlation_heatmaps(first_batch_correlations, "t5_plots_single_batch", "Single Batch Correlation (T5-base Encoder)")
        
        for layer_idx, layer_attn in enumerate(attention_patterns):
            layer_correlations = compute_head_correlations(layer_attn)
            all_layer_correlations[layer_idx].append(layer_correlations)
    
    avg_correlations = [np.mean(layer_correlations, axis=0) for layer_correlations in all_layer_correlations]
    plot_correlation_heatmaps(avg_correlations, "t5_plots_average", "Average Correlation across C4 Samples (T5-base Encoder)")

def prepare_t5(checkpoint='t5-large'):
    # load t5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, output_attentions=True)
    model = model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    return tokenizer, model

def prepare_albert(checkpoint, depth, attention_heads=None, apply_tome=False, margin=None):
    # load albert model and tokenizer
    from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
    tokenizer = AlbertTokenizer.from_pretrained(checkpoint)
    
    if attention_heads is not None:
        config = AlbertConfig.from_pretrained(checkpoint, num_hidden_layers=depth, num_attention_heads=attention_heads)
    else:
        config = AlbertConfig.from_pretrained(checkpoint, num_hidden_layers=depth)
        
    model = AlbertModel.from_pretrained(checkpoint, config=config, attn_implementation='eager')
    if apply_tome:
        apply_patch_to_albert(model, trace_source=False, prop_attn=True, margin=margin, alpha=1.0, use_attn=False)
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = prepare_albert('albert-xlarge-v2', depth=48)
    evaluate_on_c4(model, tokenizer)