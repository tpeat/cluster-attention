from enum import Enum, auto

import torch

from model import autoregressive_mask


class DecodingStrategy(Enum):
    """
    Enum class for different decoding strategies.
    """
    TOP_K = auto()
    TOP_P = auto()
    GREEDY = auto()
    RANDOM = auto()
    BEAM_SEARCH = auto()


class SequenceGenerator:
    def __init__(self, model, pad_token,sos_token, eos_token, max_len=50):
        """
        Initializes the sequence generator with a model and parameters for decoding.
        Args:
            model (torch.nn.Module): The trained transformer for generating predictions.
            sos_token (int): The index of the start symbol in the vocabulary.
            eos_token (int): The index of the end symbol in the vocabulary.
            pad_token (int): The index of the padding symbol in the vocabulary.
            max_len (int): The maximum length of the output sequence to generate.
        """
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_len = max_len

    def get_log_prob_from_model(self, memory, src_mask, out_tokens, tgt_mask):
        
        decoder_out = self.model.decode(memory, src_mask, out_tokens, tgt_mask)
        log_prob = self.model.generator(decoder_out)
        return log_prob

    def generate(self, src, src_mask, strategy=DecodingStrategy.GREEDY, k=None, p=None):
        """
        Performs batched autoregressive generation on the model's output using different sampling techniques.
        Args:
            src (torch.Tensor): The encoded source sequence tensor. Shape: [batch_size, seq_len, feature_dim]
            src_mask (torch.Tensor): The mask tensor for the source sequence. Shape: [batch_size, 1, seq_len]
            strategy (DecodingStrategy): The decoding strategy to use. Defaults to DecodingStrategy.GREEDY.
        Returns:
            List[List[int]]: A batch of decoded sequences of tokens.
        """
        with torch.no_grad():
            batch_size = src.size(0)
            out_tokens = torch.full((batch_size, 1), self.sos_token, dtype=torch.long).to(src.device)

            memory = self.model.encode(src, src_mask)
            tgt_mask = autoregressive_mask(out_tokens.size(1)).to(src.device)
            # encoded_src = self.model.encode(src, src_mask)
            # ag_mask = autoregressive_mask(self.max_len).type_as(src.data)
            for i in range(self.max_len -1):  # -1 to account for the SOS token
                prob = None
                # tgt_mask = ag_mask[:,:i+1, :i+1]
                # decoder_input = out_tokens[:, :i+1]
                # out = self.model.decode(encoded_src, src_mask, decoder_input, tgt_mask)
                # log_probs = self.model.generator.forward(out)
                # prob = torch.exp(log_probs)
                # prob = prob[:, -1, :]  # Shape: [batch_size, vocab_size]
                log_prob = self.get_log_prob_from_model(memory, src_mask, out_tokens, tgt_mask)
            # log_prob = self.model(src, src_mask, out_tokens, tgt_mask)
                prob = torch.exp(log_prob[:, -1, :])
                # YOUR CODE ENDS HERE
                # These are the different decoding strategies to generate the next token
                # Will be implemented in the following methods
                if strategy == DecodingStrategy.GREEDY:
                    next_word, next_word_log_prob = self.sample_greedy(prob)
                elif strategy == DecodingStrategy.RANDOM:
                    next_word, next_word_log_prob = self.sample_random(prob)
                elif strategy == DecodingStrategy.TOP_K:
                    next_word, next_word_log_prob = self.sample_top_k(prob, k=k)
                elif strategy == DecodingStrategy.TOP_P:
                    next_word, next_word_log_prob = self.sample_top_p(prob, p=p)
                else:
                    raise ValueError(f"Invalid decoding strategy: {strategy}")
                # TODO: Implement the functionality to append the next_word to the out_tokens tensor
                # YOUR CODE STARTS HERE
                out_tokens = torch.cat([out_tokens, next_word.unsqueeze(1)], dim=1)
                # append the next_word to the ys tensor
                # break the loop if all sequences have generated the EOS token (important for efficiency)
                # out_tokens[:, i+1] = next_word
                if torch.all(next_word == self.eos_token):
                    break
                # YOUR CODE ENDS HERE

            # Remove sequences after the end symbol for each batch item
            '''decoded_sequences = []
            # TODO: Implement the functionality to remove tokens after the EOS token
            # YOUR CODE STARTS HERE
            # for each sequence in the batch, remove the padding tokens and append the sequence to the decoded_sequences
            # list (remember to convert to list of ints)
            for sequence in out_tokens:
                new_seq = []
                for token in sequence:
                    if token == self.eos_token:
                        new_seq.append(token.item())
                        break
                    elif token == self.pad_token:
                        continue
                    else:
                        new_seq.append(token.item())
                decoded_sequences.append(new_seq)
            '''     
                
            # YOUR CODE ENDS HERE
            return out_tokens

    def beam_search(self, src, src_mask, beam_width=3):
        """
          Perform beam search decoding for a single input sequence.
          Args:
              src (torch.Tensor): The encoded source sequence tensor. Shape: [1, seq_len, feature_dim]
              src_mask (torch.Tensor): The mask tensor for the source sequence. Shape: [1, 1, seq_len]
              beam_width (int): The number of sequences to keep at each step in the beam.
          Returns:
              List[int]: The best sequence of token IDs based on beam search.
      """
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search is implemented for a single sequence only."
        with torch.no_grad():
            # Starting with the initial token.
            ys = torch.full((1, 1), self.sos_token, dtype=torch.long).to(src.device)
            beam_candidates = [(ys, 0)]  # list of tuples (sequence tensor, log probability)
            encoded_src = self.model.encode(src, src_mask)
            ag_mask = autoregressive_mask(self.max_len).type_as(src.data)
            for i in range(self.max_len - 1):  # -1 for the sos token
                all_candidates = []
                tgt_mask = ag_mask[:,:i+1,:i+1]
                for ys, log_prob in beam_candidates:
                    # TODO: Implement the functionality to get the log probabilities of the next token using the model's decode method
                    # YOUR CODE STARTS HERE
                    """
                    Steps:
                    1. Get the log probabilities of the next token using the model's decode method.
                    2. Get the top beam_width tokens (by probability values) and their log probabilities.
                    3. Create new candidate sequences by appending each of the top tokens to the current sequence.
                    4. Add the new candidate sequences to the list of all candidates.
                    HINT: The idea will be similar to generate, but you will have to keep track of multiple sequences.
                    """
                    
                    out = self.model.decode(encoded_src, src_mask, ys, tgt_mask)
                    log_prob_cand = self.model.generator.forward(out)
                    log_prob_cand = log_prob_cand[:,-1,:]
                    top_probs, top_indices = torch.topk(log_prob_cand, beam_width)
                    for j in range(beam_width):
                        next_seq = torch.cat((ys, top_indices[0,j].unsqueeze(-1).unsqueeze(-1)), dim=1) # Append the new token
                        cum_log_prob = log_prob + top_probs[0,j].item()
                        all_candidates.append((next_seq, cum_log_prob))
                    # YOUR CODE ENDS HERE

                # TODO: Implement the functionality to sort all candidates by log probability and select the best beam_width ones
                # YOUR CODE STARTS HERE - Sort all candidates by log probability, select the best beam_width ones
                # Sort all candidates by log probability, select the best beam_width ones
                sorted_cands = sorted(all_candidates, key= lambda x: x[1], reverse=True)
                
                beam_candidates = sorted_cands[:beam_width]
                # YOUR CODE ENDS HERE

                # Check if the end token is generated and stop early
                if all((c[0][0, -1] == self.eos_token) for c in beam_candidates):
                    break

            # Choose the sequence with the highest log probability
            best_sequence, _ = max(beam_candidates, key=lambda x: x[1])
            result = best_sequence[0].tolist()
            return result

    @staticmethod
    def sample_greedy(prob):
        """
        Perform greedy decoding to get the next token index based on the probability distribution.
        Steps -
        1. Get the index of the token with the highest probability.
        2. Retrieve the log probability of the chosen token
        Args:
            prob (torch.Tensor): The probability distribution over the target vocabulary of shape
            [batch_size, vocab_size].

        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS:
        - The functions torch.gather may be useful.
        """
        next_word, log_probability_of_next_word = None, None
        # TODO: Implement Greedy Sampling
        # YOUR CODE STARTS HERE
        next_word = torch.argmax(prob, dim=-1)
        log_probs = torch.log(prob)
        log_probability_of_next_word = torch.gather(log_probs,dim=-1, index=next_word.unsqueeze(-1)).squeeze(-1)

        # YOUR CODE ENDS HERE
        return next_word, log_probability_of_next_word

    @staticmethod
    def sample_random(prob):
        """
        Perform random sampling to get the next token index based on the probability distribution.
        Steps -
        1. Sample from the probability distribution over the target vocabulary.
        2. Retrieve the log probability of the chosen token.
        3. Map sampled indices back to the global vocabulary indices.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS:
        - The functions torch.multinomial and torch.gather may be useful.
        """
        next_word, log_probability_of_next_word = None, None
        # TODO: Implement Random Sampling
        # YOUR CODE STARTS HERE
        
        next_word = torch.multinomial(prob, 1)
        log_prob = torch.log(prob)
        log_probability_of_next_word = torch.gather(log_prob, dim=-1, index=next_word).squeeze(-1)
        

        # YOUR CODE ENDS HERE
        return next_word.squeeze(-1), log_probability_of_next_word

    @staticmethod
    def sample_top_k(prob, k=5):
        """
        Perform top-k sampling to get the next token index based on the probability distribution.
        Steps -
        1. Filter the top k tokens from the distribution.
        2. Normalize the probabilities to sum to 1.
        3. Randomly sample from this modified distribution of top-k tokens to determine the next token.
        4. Retrieve the log probability and index of the chosen token in the global vocabulary.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
            k (int): The number of top elements to sample from.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size].
        HINTS -
        - The function torch.topk may be useful.
        """
        next_word, log_probability_of_next_word = None, None
        # TODO: Implement Top-k Sampling
        # YOUR CODE STARTS HERE
        
        top_k_samples, top_k_sample_idx = torch.topk(prob, k, dim=-1)
        top_k_samples = top_k_samples / top_k_samples.sum(dim=1, keepdim=True) 

        rand_samples = torch.multinomial(top_k_samples,1)
        next_word = torch.gather(top_k_sample_idx, dim=-1, index=rand_samples)
        log_probs = torch.log(top_k_samples)
        log_probability_of_next_word = torch.gather(log_probs, dim=-1, index=rand_samples)
        
        # YOUR CODE ENDS HERE
        return next_word.squeeze(-1), log_probability_of_next_word.squeeze(-1)

    @staticmethod
    def sample_top_p(prob, p=0.9):
        """
        Perform top-p sampling to get the next token index based on the probability distribution.
        Steps -
        1. Retrieve the smallest subset of the distribution that sums just greater than p
        (since = isn't always possible).
        2. Normalize the probabilities to sum to 1.
        3. Randomly sample from this modified distribution to determine the next token.
        4. Retrieve the log probability and index of the chosen token in the global vocabulary.
        Args:
            prob (torch.Tensor): The probability distribution of the batch over the target vocabulary.
            p (float): The cumulative probability threshold for top-p sampling.
        Returns:
            torch.Tensor: The index of the next token of shape [batch_size].
            torch.Tensor: The log probability of the chosen token of shape [batch_size]
        HINTS:
        - The function torch.cumsum may be useful.
        """
        next_word, log_probability_of_next_word = None, None
        # TODO: Implement Top-p Sampling
        # YOUR CODE STARTS HERE
        sorted_probs, sorted_indices = torch.sort(prob, descending=True)
        cum_sum = torch.cumsum(sorted_probs, dim=-1)
        # find where cum_sum > p  
        cum_sum_mask = (cum_sum > p)
        #shift by one to account for including value that set cum sum > p
        cum_sum_mask[:, 1:] = cum_sum_mask[:, :-1].clone()
        cum_sum_mask[:,0] = 0
        #set low prob samples to 0
        sorted_probs = sorted_probs.masked_fill(cum_sum_mask, 0.0)
        #normalize the new distribution
        top_p_samples = sorted_probs / sorted_probs.sum(dim=1, keepdim=True) 
        
        #take samples from new distribution
        rand_samples = torch.multinomial(top_p_samples,1)
        next_word = torch.gather(sorted_indices, dim=-1, index=rand_samples)
        log_probs = torch.log(top_p_samples)
        log_probability_of_next_word = torch.gather(log_probs, dim=-1, index=rand_samples)
        
        # YOUR CODE ENDS HERE
        return next_word.squeeze(-1), log_probability_of_next_word.squeeze(-1)


    def tristan_beam_search(self, src, src_mask, beam_width=3):
        """
          Perform beam search decoding for a single input sequence.
          Args:
              src (torch.Tensor): The encoded source sequence tensor. Shape: [1, seq_len, feature_dim]
              src_mask (torch.Tensor): The mask tensor for the source sequence. Shape: [1, 1, seq_len]
              beam_width (int): The number of sequences to keep at each step in the beam.
          Returns:
              List[int]: The best sequence of token IDs based on beam search.
        """
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search is implemented for a single sequence only."

        # Starting with the initial token.
        ys = torch.full((1, 1), self.sos_token, dtype=torch.long).to(src.device)
        beam_candidates = [(ys, 0)]  # list of tuples (sequence tensor, log probability)

        for _ in range(self.max_len - 1):  # -1 for the sos token
            all_candidates = []
            for ys, log_prob in beam_candidates:
                # TODO: Implement the functionality to get the log probabilities of the next token using the model's decode method
                # YOUR CODE STARTS HERE
                """
                  Steps:
                  1. Get the log probabilities of the next token using the model's decode method.
                  2. Get the top beam_width tokens (by probability values) and their log probabilities.
                  3. Create new candidate sequences by appending each of the top tokens to the current sequence.
                  4. Add the new candidate sequences to the list of all candidates.
                  HINT: The idea will be similar to generate, but you will have to keep track of multiple sequences.
                """
                l_prob = self.get_log_prob_from_model(src, src_mask, ys)
                next_token_log_prob = l_prob[:, -1, :]
                top_log_prob, top_indices = torch.topk(next_token_log_prob, beam_width, dim=-1)

                for i in range(beam_width):
                    next_word = top_indices[0, i].unsqueeze(0)
                    next_log_prob = top_log_prob[0, i].item()
                    
                    # append new token
                    new_sequence = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                    # update log prob and store for later
                    new_cumulative_log_prob = log_prob + next_log_prob
                    all_candidates.append((new_sequence, new_cumulative_log_prob))

                # YOUR CODE ENDS HERE

            # TODO: Implement the functionality to sort all candidates by log probability and select the best beam_width ones
            # YOUR CODE STARTS HERE - Sort all candidates by log probability, select the best beam_width ones
            # Sort all candidates by log probability, select the best beam_width ones
            beam_candidates = sorted(all_candidates, 
                               key=lambda x: x[1], 
                               reverse=True)[:beam_width]

            # YOUR CODE ENDS HERE

            # Check if the end token is generated and stop early
            if all((c[0][0, -1] == self.eos_token) for c in beam_candidates):
                break

        # Choose the sequence with the highest log probability
        best_sequence, _ = max(beam_candidates, key=lambda x: x[1])
        result = best_sequence[0].tolist()
        return result