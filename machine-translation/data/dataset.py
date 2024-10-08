from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        source_text = self.dataset[idx]["translation"]["en"]
        target_text = self.dataset[idx]["translation"]["fr"]

        source_encodings = self.tokenizer(source_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids': source_encodings['input_ids'].flatten(),
            'attention_mask': source_encodings['attention_mask'].flatten(),
            'labels': target_encodings['input_ids'].flatten()
        }

    def preprocess_target_sequence(self, tgt_sequence):
        """
        Going to retry training once more under MBart with the context window first
        Then we will add this in
        """
        # Prepend the start token
        decoder_input_ids = [start_token_id] + tgt_sequence
        # Append the end token if not already present
        if tgt_sequence[-1] != end_token_id:
            decoder_input_ids.append(end_token_id)
        # Truncate if necessary
        decoder_input_ids = decoder_input_ids[:max_length]
        # Pad if necessary
        padding_length = max_length - len(decoder_input_ids)
        if padding_length > 0:
            decoder_input_ids += [pad_token_id] * padding_length
        return decoder_input_ids


class NewTranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Determine the start and end token IDs
        if hasattr(tokenizer, 'lang_code_to_id'):
            # For MBart tokenizer
            self.start_token_id = tokenizer.lang_code_to_id['fr_XX']  # Replace 'fr_XX' with your target language code
        else:
            # For BERT tokenizer or others
            self.start_token_id = tokenizer.cls_token_id
        
        self.end_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        source_text = self.dataset[idx]["translation"]["en"]
        target_text = self.dataset[idx]["translation"]["fr"]

        # Tokenize the source text
        source_encodings = self.tokenizer(
            source_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenize the target text without adding special tokens
        target_encodings = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors='pt'
        )

        # Convert target input IDs to a list
        tgt_sequence = target_encodings['input_ids'].squeeze().tolist()
        # Remove any padding tokens from the end
        tgt_sequence = [token for token in tgt_sequence if token != self.pad_token_id]

        # Preprocess the target sequence
        decoder_input_ids = preprocess_target_sequence(
            tgt_sequence,
            self.start_token_id,
            self.end_token_id,
            self.max_length,
            self.pad_token_id
        )

        # Create attention masks
        decoder_attention_mask = [1 if token != self.pad_token_id else 0 for token in decoder_input_ids]

        # Convert lists back to tensors
        decoder_input_ids = torch.tensor(decoder_input_ids)
        decoder_attention_mask = torch.tensor(decoder_attention_mask)

        # Labels are the target sequence shifted to the right (for teacher forcing)
        labels = torch.tensor(decoder_input_ids.tolist())

        return {
            'input_ids': source_encodings['input_ids'].squeeze(),
            'attention_mask': source_encodings['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels
        }
