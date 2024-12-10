import copy
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from datasets import load_dataset
from transformers.models.albert.modeling_albert import AlbertModel
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
from prepare_t5 import create_tokenizer_model

def compute_correlations(hidden_states):
    # compute correlations between hidden states
    corrs = []
    for hs in hidden_states:
        T = hs.squeeze(0).clone().detach().requires_grad_(False)
        T = torch.nn.functional.normalize(T, dim=1)
        T2 = torch.matmul(T, T.transpose(0,1))
        corrs.append(T2.flatten().cpu())
    return corrs

def get_random_input(dataset, tokenizer):
    # get random input with minimum length of 300 tokens
    l = len(dataset['train'])
    while True:
        it = torch.randint(l,(1,)).item()
        text = dataset['train'][it]['text']
        ei = tokenizer(text, return_tensors='pt', truncation=True)
        if ei['input_ids'].shape[1] > 300:
            break
    return ei

def plot_histograms_save(tokenizer, model):
    # plot and save correlation histograms for each layer
    wikitext = load_dataset("wikitext", 'wikitext-103-v1')
    ei = get_random_input(wikitext, tokenizer)
    
    if isinstance(model, AlbertModel):
        of = model(**ei, output_hidden_states=True)
        correls = compute_correlations(of['hidden_states'])
    else:
        encoder_outputs = model.encoder(
            input_ids=ei['input_ids'],
            attention_mask=ei.get('attention_mask', None),
            output_hidden_states=True,
            return_dict=True
        )
        correls = compute_correlations(encoder_outputs.hidden_states)

    os.makedirs('histograms', exist_ok=True)
    max_density = 0
    for data in correls:
        counts, bin_edges = np.histogram(data, bins=100, density=True)
        max_density = max(max_density, max(counts))

    for i, data in enumerate(correls):
        IQR = np.percentile(data, 75) - np.percentile(data, 25)
        n = len(data)
        bin_width = 2 * IQR / n ** (1/3)
        bins = int((max(data) - min(data)) / bin_width)

        plt.figure()
        plt.hist(data, bins=bins, density=True, histtype='step', color='#3658bf', linewidth=1.5)
        plt.title(f'Layer {i}', fontsize=16)
        plt.xlim(-.3, 1.05)
        plt.ylim(0, max_density)
        plt.savefig(f'histograms/histogram_layer_{i}.pdf')
        plt.close()

def prepare_albert_model(checkpoint, depth, attention_heads=1):
    # prepare albert model with specified config
    tokenizer = AlbertTokenizer.from_pretrained(checkpoint)
    config = AlbertConfig.from_pretrained(checkpoint, num_hidden_layers=depth, num_attention_heads=attention_heads)
    model = AlbertModel.from_pretrained(checkpoint, config=config)
    return tokenizer, model

if __name__ == "__main__":
    num_repeats = 4
    t5_checkpoint = 'google/t5-large'
    albert_checkpoint = "albert-xlarge-v2"
    al_depth = 96

    tokenizer, model = prepare_albert_model(albert_checkpoint, depth)

    plot_histograms_save(tokenizer, model)

