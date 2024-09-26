import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self, tokenizer, src_vocab_size, tgt_vocab_size, d_model, num_layers,
        num_heads, d_ff, max_seq_length
    ):
        super(Transformer, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = Encoder(
            src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length
        )
        
    def make_src_mask(self, src):
        src_mask = (src != self.tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        return src_mask  # Shape: (batch_size, 1, 1, src_len)
        
    def make_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != self.tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)
        return tgt_mask  # Shape: (batch_size, 1, tgt_len, tgt_len)
        
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output
