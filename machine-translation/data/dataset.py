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
        target_encodings = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids': source_encodings['input_ids'].flatten(),
            'attention_mask': source_encodings['attention_mask'].flatten(),
            'labels': target_encodings['input_ids'].flatten()
        }