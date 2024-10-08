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

    def greedy_decode(self, src, max_len, end_symbol):
        """
        Greedy decoding method without using a start symbol.
        
        Args:
            src (torch.Tensor): Source token IDs tensor of shape [1, src_len].
            max_len (int): Maximum length of the generated translation.
            end_symbol (int): The EOS token ID.
            
        Returns:
            List[int]: Generated target token IDs.
        """
        device = src.device
        self.eval()
        
        with torch.no_grad():
            enc_output, src_mask = self.encode(src)
            
            # Initialize target sequence with the <pad> token
            tgt_tokens = torch.full((1, 1), self.tokenizer.pad_token_id, dtype=torch.long).to(device)  # Shape: [1, 1]
            
            for _ in range(max_len):
                tgt_mask = self.make_tgt_mask(tgt_tokens)
                dec_output = self.decode(tgt_tokens, enc_output, src_mask, tgt_mask)
                
                # Get the logits for the last token
                next_token_logits = dec_output[:, -1, :]  # Shape: [1, vocab_size]
                
                # Predict the next token (greedy decoding)
                next_token_id = next_token_logits.argmax(dim=-1).item()
                
                # Append the predicted token to the target sequence
                tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token_id]], device=device)], dim=1)
                
                # If the <eos> token is generated, stop decoding
                if next_token_id == end_symbol:
                    break
            
            # Convert tensor to list and remove initial <pad> token and <eos> token
            translation_tokens = tgt_tokens.squeeze(0).tolist()
            if translation_tokens[-1] == end_symbol:
                translation_tokens = translation_tokens[:-1]
            if translation_tokens[0] == self.tokenizer.pad_token_id:
                translation_tokens = translation_tokens[1:]
                
            return translation_tokens

    def beam_search_decode(
        self,
        src: torch.Tensor,
        max_len: int,
        end_symbol: int,
        beam_width: int = 5,
        length_penalty: float = 1.0
    ):
        """
        Beam search decoding method.

        Args:
            src (torch.Tensor): Source token IDs tensor of shape [1, src_len].
            max_len (int): Maximum length of the generated translation.
            end_symbol (int): The EOS token ID.
            beam_width (int): Number of beams to keep.
            length_penalty (float): Penalty to apply based on the length of the sequence.

        Returns:
            List[int]: The generated target token IDs.
        """
        device = src.device
        self.eval()

        with torch.no_grad():
            enc_output, src_mask = self.encode(src)

        # Each beam is a tuple: (score, tokens)
        beams: List[Tuple[float, List[int]]] = [(0.0, [])]

        for _ in range(max_len):
            new_beams = []
            # Expand each beam
            for score, tokens in beams:
                if len(tokens) > 0 and tokens[-1] == end_symbol:
                    # If the last token is end_symbol, keep the beam as is
                    new_beams.append((score, tokens))
                    continue

                # Prepare the target tensor
                tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(device) if tokens else torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long).to(device)

                # Generate target mask
                tgt_mask = self.make_tgt_mask(tgt_tensor)

                # Decode
                dec_output = self.decode(tgt_tensor, enc_output, src_mask, tgt_mask)

                # Get log probabilities (for numerical stability)
                log_probs = torch.log_softmax(dec_output[:, -1, :], dim=-1)  # Shape: [1, vocab_size]
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)  # Shape: [1, beam_width]

                top_log_probs = top_log_probs.squeeze(0).detach().cpu().numpy()
                top_indices = top_indices.squeeze(0).detach().cpu().numpy()

                for log_prob, token_id in zip(top_log_probs, top_indices):
                    new_score = score + log_prob
                    new_tokens = tokens + [token_id]
                    new_beams.append((new_score, new_tokens))

            # Select top beams
            # Optionally apply length penalty
            beams = sorted(new_beams, key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)[:beam_width]

            # If all beams have ended, stop
            if all(tokens[-1] == end_symbol for _, tokens in beams):
                break

        # Select the best beam
        best_score, best_tokens = beams[0]

        # Remove end_symbol if present
        if best_tokens and best_tokens[-1] == end_symbol:
            best_tokens = best_tokens[:-1]

        # If the first token is <cls>, remove it
        if best_tokens and best_tokens[0] == self.tokenizer.cls_token_id:
            best_tokens = best_tokens[1:]

        return best_tokens
