# methods to apply pitome to albert
import torch
import torch.nn as nn
from typing import Tuple, Optional
import math
from transformers.models.albert.modeling_albert import (
    AlbertLayer,
    AlbertTransformer,
    AlbertAttention,
    AlbertSdpaAttention,
    apply_chunking_to_forward,
)
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import BaseModelOutputWithPooling
from PiToMe.algo.pitome.merge import merge_source, pitome_text, merge_mean, merge_wavg, merge_attention_mask

class PiToMeAlbertAttention(AlbertAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Process the input through query, key, and value projections
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query = self.transpose_for_scores(mixed_query_layer)
        key = self.transpose_for_scores(mixed_key_layer)
        value = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        # Output processing
        attention_output = self.dense(context_layer)
        attention_output = self.output_dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        
        # print(attention_output, key.sum(1), attention_probs)
        return (attention_output, key.sum(1), attention_probs)

class PiToMeAlbertLayer(AlbertLayer):
    def init_margin(self, margin):
        self.margin = margin
   
    def calculate_block_flop(self, shape):
        flops = 0
        _, N, C = shape
        mhsa_flops = 4*N*C*C + 2*N*N*C
        flops += mhsa_flops
        ffn_flops = 8*N*C*C
        flops += ffn_flops
        return flops

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        
        # Get the ratio for the current layer
        ratio = self._info["ratio"][0] if isinstance(self._info["ratio"], list) else self._info["ratio"]
        x = attention_output[0]
        key = attention_output[1]
        attn = attention_output[2]
    
        if ratio < 1.0:
            print(ratio)
            # Use pitome_text from merge module
            merge = pitome_text(
                ratio=ratio,
                metric=key,
                margin=self.margin,
                class_token=self._info["class_token"],
            )

            # Use merge_wavg from merge module
            x, self._info["size"] = merge_wavg(merge, x, None)
            
            # Track source if enabled
            if self._info["trace_source"]:
                self._info["source"] = merge_source(merge, self._info["source"])
            
            B, T, _ = x.shape
            # Adjust attention_mask dimensions if necessary
            if attention_mask.dim() >= 3:
                attention_mask = attention_mask.squeeze(-2).squeeze(-2)
            attention_mask = torch.where(attention_mask >= 0, 1, 0)
            # Use merge_attention_mask from merge module
            attention_mask = merge_attention_mask(merge, attention_mask=attention_mask[..., None]).view(B, T)
        else:
            # Adjust attention_mask to be of shape [batch_size, seq_len]
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        # Calculate FLOPS
        flops = self.calculate_block_flop(x.shape)

        # Use ALBERT's original feed-forward implementation with chunking
        ffn_output = apply_chunking_to_forward(
            self.ff_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            x
        )
        hidden_states = self.full_layer_layer_norm(ffn_output + x)

        # Return format matching original ALBERT with PiToMe additions
        outputs = (hidden_states, flops) + (key, attn) if output_attentions else (hidden_states, flops)
        return outputs
    
def make_pitome_albert_class(transformer_class):
    class PiToMeAlbertTransformer(transformer_class, ModuleUtilsMixin):
        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            print("output hidden states?", output_hidden_states)
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            batch_size, seq_length = input_shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            if token_type_ids is None:
                if hasattr(self.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            # Convert attention mask to extended attention mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            embedding_output = self.embeddings(
                input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )

            # Initialize lists to collect outputs
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            total_flops = 0

            # Pass through the encoder
            encoder_outputs = self.encoder(
                embedding_output,
                extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # Unpack encoder outputs
            sequence_output = encoder_outputs[0]
            total_flops = sum(output[1] for output in encoder_outputs if isinstance(output, tuple) and len(output) > 1)

            # Get pooled output if pooler exists
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

            if not return_dict:
                outputs = (sequence_output, pooled_output)
                # Add hidden states if they were collected
                if output_hidden_states:
                    outputs = outputs + (all_hidden_states,)
                # Add attentions if they were collected
                if output_attentions:
                    outputs = outputs + (all_self_attentions,)
                # Add FLOPS counter
                outputs = outputs + (total_flops,)
                return outputs

            return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                attentions=all_self_attentions if output_attentions else None,
            )

    return PiToMeAlbertTransformer


class PiToMeAlbertSelfAttention(AlbertSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Fall back to regular attention if using non-absolute position embeddings or need output_attentions
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            return super().forward(hidden_states, attention_mask, head_mask, output_attentions)

        batch_size, seq_len, _ = hidden_states.size()
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)

        projected_context_layer = self.dense(attention_output)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        
        # Calculate attention probabilities for the output
        attention_probs = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_probs = attention_probs + attention_mask
        attention_probs = nn.functional.softmax(attention_probs, dim=-1)
        
        return layernormed_context_layer, key.sum(1), attention_probs

def apply_patch_to_albert(
    model: AlbertTransformer, 
    trace_source: bool = False, 
    prop_attn: bool = True, 
    margin=None, 
    alpha=1.0, 
    use_attn=False
):
    PiToMeAlbertTransformer = make_pitome_albert_class(model.__class__)
    model.__class__ = PiToMeAlbertTransformer
    
    len_layers = model.config.num_hidden_layers
    shared_layers = model.config.num_hidden_groups
    layers_per_group = len_layers // shared_layers
    
    # Create ratio list with the same length as actual layers
    model.ratio = []
    for g in range(shared_layers):
        for l in range(layers_per_group):
            layer_idx = g * layers_per_group + l
            if layer_idx >= len_layers - 3:
                model.ratio.append(0.5)  # Last 3 layers use 0.5 ratio
            else:
                model.ratio.append(1.0)
    
    model._info = {
        "ratio": model.ratio,
        "margin": [],
        "size": None,
        "source": None,
        "use_attn": use_attn,
        "trace_source": trace_source, 
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
        "alpha": alpha,
    }
    
    # Calculate margins
    if margin is None:
        margins = [0.9 - 0.25 * (i / len_layers) for i in range(len_layers)]
    else:
        margins = [margin for _ in range(len_layers)]

    current_layer = 0
    # Patch each layer
    for group in model.encoder.albert_layer_groups:
        for layer in group.albert_layers:
            print(layer)
            # Update layer class and initialize margin
            layer.__class__ = PiToMeAlbertLayer
            layer.init_margin(margins[current_layer])
            layer._info = model._info
            
            # Update attention class
            if hasattr(layer, 'attention'):
                if isinstance(layer.attention, AlbertSdpaAttention):
                    print("found spda")
                    # Keep SDPA attention if it's being used
                    layer.attention.__class = PiToMeAlbertSelfAttention
                layer.attention.__class__ = PiToMeAlbertAttention
            
            current_layer += 1
    
    return model