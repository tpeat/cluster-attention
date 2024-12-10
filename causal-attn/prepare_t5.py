import copy
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

def adjust_attention_weights_qkv(pretrained_weights, original_num_heads, new_num_heads):
    # adjust qkv attention weights for new head count
    d_kv = pretrained_weights.shape[0] // original_num_heads
    d_model = pretrained_weights.shape[1]
    pretrained_weights = pretrained_weights.view(original_num_heads, d_kv, d_model)
    
    if new_num_heads < original_num_heads:
        factor = original_num_heads // new_num_heads
        assert original_num_heads % new_num_heads == 0
        pretrained_weights = pretrained_weights.view(new_num_heads, factor, d_kv, d_model).mean(dim=1)
    elif new_num_heads > original_num_heads:
        pretrained_weights = pretrained_weights.repeat(new_num_heads // original_num_heads, 1, 1)
    
    return pretrained_weights.view(new_num_heads * d_kv, d_model)

def adjust_attention_weights_o(pretrained_weights, original_num_heads, new_num_heads):
    # adjust output attention weights for new head count
    d_kv = pretrained_weights.shape[1] // original_num_heads
    d_model = pretrained_weights.shape[0]
    pretrained_weights = pretrained_weights.view(d_model, original_num_heads, d_kv)
    
    if new_num_heads < original_num_heads:
        factor = original_num_heads // new_num_heads
        assert original_num_heads % new_num_heads == 0
        pretrained_weights = pretrained_weights.view(d_model, new_num_heads, factor, d_kv).mean(dim=2)
    elif new_num_heads > original_num_heads:
        pretrained_weights = pretrained_weights.repeat(1, new_num_heads // original_num_heads, 1)
    
    return pretrained_weights.view(d_model, new_num_heads * d_kv)

def adjust_relative_attention_bias_weight(pretrained_weight, original_num_heads, new_num_heads):
    # adjust relative attention bias for new head count
    num_buckets = pretrained_weight.shape[0]
    
    if new_num_heads < original_num_heads:
        factor = original_num_heads // new_num_heads
        assert original_num_heads % new_num_heads == 0
        pretrained_weight = pretrained_weight.view(num_buckets, new_num_heads, factor).mean(dim=2)
    elif new_num_heads > original_num_heads:
        factor = new_num_heads // original_num_heads
        pretrained_weight = pretrained_weight.unsqueeze(2).repeat(1, 1, factor).view(num_buckets, new_num_heads)
    
    return pretrained_weight

def copy_weights_into_new_model(original_model, new_model, num_repeats, original_num_heads, new_num_heads):
    # copy and adjust weights from original to new model
    original_state_dict = original_model.state_dict()
    new_state_dict = new_model.state_dict()
    
    shared_keys = ['shared.weight', 'encoder.embed_tokens.weight', 
                   'decoder.embed_tokens.weight', 'lm_head.weight']
    for key in shared_keys:
        new_state_dict[key] = original_state_dict[key]
    
    original_num_layers = original_model.config.num_layers
    for i in range(new_model.config.num_layers):
        original_layer_idx = i % original_num_layers if i < original_num_layers else original_num_layers - 1
        
        for param_name in original_state_dict:
            for block in ['encoder.block', 'decoder.block']:
                if param_name.startswith(f'{block}.{original_layer_idx}.'):
                    new_param_name = param_name.replace(f'{block}.{original_layer_idx}.', 
                                                      f'{block}.{i}.')
                    if new_param_name in new_state_dict:
                        if ('SelfAttention' in param_name or 'EncDecAttention' in param_name) and 'weight' in param_name:
                            if 'o.weight' in param_name:
                                adjusted_weight = adjust_attention_weights_o(original_state_dict[param_name], 
                                                                          original_num_heads, new_num_heads)
                            elif 'relative_attention_bias.weight' in param_name:
                                adjusted_weight = adjust_relative_attention_bias_weight(original_state_dict[param_name],
                                                                                     original_num_heads, new_num_heads)
                            else:
                                adjusted_weight = adjust_attention_weights_qkv(original_state_dict[param_name],
                                                                            original_num_heads, new_num_heads)
                            new_state_dict[new_param_name] = adjusted_weight
                        else:
                            new_state_dict[new_param_name] = original_state_dict[param_name]
    
    final_norms = ['encoder.final_layer_norm.weight', 'encoder.final_layer_norm.bias',
                   'decoder.final_layer_norm.weight', 'decoder.final_layer_norm.bias']
    for key in final_norms:
        if key in original_state_dict and key in new_state_dict:
            new_state_dict[key] = original_state_dict[key]
    
    new_model.load_state_dict(new_state_dict)
    return new_model

def create_tokenizer_model(checkpoint, num_repeats, new_num_heads=1):
    # create new model with adjusted depth and head count
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    pretrained_model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    new_config = copy.deepcopy(pretrained_model.config)
    new_config.num_layers = pretrained_model.config.num_layers * num_repeats
    new_config.num_heads = new_num_heads
    new_model = T5ForConditionalGeneration(new_config)

    new_model = copy_weights_into_new_model(pretrained_model, new_model, num_repeats,
                                          pretrained_model.config.num_heads, new_num_heads)
    return tokenizer, new_model