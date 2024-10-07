import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self, tokenizer, src_vocab_size, tgt_vocab_size, d_model, num_layers,
        num_heads, d_ff, max_seq_length, self_attn_fn, x_attn_fn
    ):
        super(Transformer, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = Encoder(
            src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, self_attn_fn
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, self_attn_fn, x_attn_fn
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

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        enc_output = self.encoder(src, src_mask)
        return enc_output, src_mask

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output
        
    def forward(self, src, tgt):
        enc_output, src_mask = self.encode(src)
        tgt_mask = self.make_tgt_mask(tgt)
        output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return output

    def greedy_decode(self, src, max_len, start_symbol, end_symbol):
        """
        Greedy decoding for inference.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (1, src_len)
            max_len (int): Maximum length of the target sequence
            start_symbol (int): Token ID for the start symbol
            end_symbol (int): Token ID for the end symbol
            device (torch.device): Device to perform computations on

        Returns:
            List[int]: Generated target token IDs
        """
        self.eval()
        # move everything to same device as src
        device = src.device
        with torch.no_grad():
            enc_output, src_mask = self.encode(src.to(device))
            
            # Initialize target sequence with the start symbol
            tgt_tokens = torch.tensor([[start_symbol]], dtype=torch.long).to(device)
            
            for _ in range(max_len):
                tgt_mask = self.make_tgt_mask(tgt_tokens)
                dec_output = self.decode(tgt_tokens, enc_output, src_mask, tgt_mask)
                
                # Get the last token's logits
                next_token_logits = dec_output[:, -1, :]  # Shape: (1, vocab_size)
                
                # Select the token with the highest probability
                _, next_token = torch.max(next_token_logits, dim=-1)
                next_token = next_token.item()
                
                # Append the next token to the target sequence
                tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]], device=device)], dim=1)
                
                # If the end symbol is generated, stop decoding
                if next_token == end_symbol:
                    break
            
            return tgt_tokens.squeeze(0).tolist()
