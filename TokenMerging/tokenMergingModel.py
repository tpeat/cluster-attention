import torch
from tokenMergingBlock import TokenMergerT5Block

def make_model(model):
    config = model.config
    layers = torch.nn.ModuleList()
    layers.append(model.encoder.block[0])
    for i in range(config.num_layers - 1):
        layers.append(TokenMergerT5Block(config))
    model.encoder.block = layers
    
    model.generation_config.max_new_tokens = 128
    return model