from torch.utils.data import Dataset
import torch
from model import autoregressive_mask
from .vocab import tokenize_sentence

def make_tgt_mask(tgt, pad):
        tgt_mask = None
        padding_mask = (tgt != pad).to(tgt.device)
        tgt_size = tgt.shape[1]
        autoreg_mask = autoregressive_mask((tgt_size)).to(tgt.device)
        padding_mask = padding_mask.unsqueeze(1)
        padding_mask = padding_mask.expand(-1, tgt_size, -1) 
        tgt_mask = padding_mask & autoreg_mask
        return tgt_mask

class TranslationDataset(Dataset):
    def __init__(self, dataset, device, en_token_to_id, fr_token_to_id, max_length=128):
        self.dataset = dataset
        self.max_length = max_length
        self.device = device
        self.en_token_to_id = en_token_to_id
        self.fr_token_to_id = fr_token_to_id
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Extract source and target text
        source_text = self.dataset[idx]["translation"]["en"]
        target_text = self.dataset[idx]["translation"]["fr"]
        
        # Tokenize source and target text
        source_tokens = tokenize_sentence(source_text, self.en_token_to_id)
        target_tokens = tokenize_sentence(target_text, self.fr_token_to_id)
        
        # Pad/truncate source and target to max_length
        pad_id_en = self.en_token_to_id["<PAD>"]
        pad_id_fr = self.fr_token_to_id["<PAD>"]
        source_ids = source_tokens[:self.max_length] + [pad_id_en] * max(0, self.max_length - len(source_tokens))
        target_ids = target_tokens[:self.max_length] + [pad_id_fr] * max(0, self.max_length - len(target_tokens))
        
        # Convert to tensors
        source_tensor = torch.tensor(source_ids, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=self.device)
        # Create attention masks
        source_mask = (source_tensor != pad_id_en).long()
        target_mask = make_tgt_mask(target_tensor.unsqueeze(0), pad=pad_id_fr)
        
        return {
            'src': source_tensor,
            'src_mask': source_mask.unsqueeze(0),
            'tgt': target_tensor,
            'tgt_mask': target_mask,
            'ntokens': (target_tensor != pad_id_fr).sum().item()
        }
