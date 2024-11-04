from torch.utils.data import Dataset
import torch
from model import autoregressive_mask

def make_tgt_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        Input Params: 
            - tgt (Tensor): target data. 
            - pad (int): padding id. 
        Returns: 
            - tgt_mask (Tensor): mask to hide padding and future words.

        HINTS:  
        1. Create a mask to hide padding in the target data. 
        2. Create an autogregressive mask to hide future words in the target data. 
        (Use the autoregressive_mask function in model.py, Section 2.8. of notebook)
        3. Combine the padding mask and the autoregressive mask. 
        """
        tgt_mask = None 

        # TODO: Implement the make_std_mask function.
        # YOUR CODE STARTS HERE
        padding_mask = (tgt != pad).to(tgt.device)
        tgt_size = tgt.shape[1]
        autoreg_mask = autoregressive_mask((tgt_size)).to(tgt.device)
        padding_mask = padding_mask.unsqueeze(1)
        padding_mask = padding_mask.expand(-1, tgt_size, -1) 
        
        tgt_mask = padding_mask & autoreg_mask
        # YOUR CODE ENDS HERE
        return tgt_mask

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, device, max_length=128, src_lang='fr', tgt_lang='en'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.src_lang= src_lang
        self.tgt_lang = tgt_lang
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        source_text = self.dataset[idx]["translation"][self.src_lang]
        target_text = self.dataset[idx]["translation"][self.tgt_lang]
        pad = self.tokenizer.get_pad_token_id()
        if self.src_lang != "en":
            source_encodings = self.tokenizer.tokenizer(text_target=source_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            target_encodings = self.tokenizer.tokenizer(target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        else:
            source_encodings = self.tokenizer.tokenizer(source_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            target_encodings = self.tokenizer.tokenizer(text_target=target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        tgt_mask = make_tgt_mask(target_encodings['input_ids'].to(self.device), pad=pad)
        return {
            'src': source_encodings['input_ids'].squeeze(0).to(self.device),
            'src_mask': source_encodings['attention_mask'].to(self.device),
            'tgt': target_encodings['input_ids'].squeeze(0).to(self.device),
            'tgt_mask': tgt_mask.squeeze(0).to(self.device),
            'ntokens': (target_encodings['input_ids'].squeeze(0).to(self.device) != pad).data.sum()
            
        }