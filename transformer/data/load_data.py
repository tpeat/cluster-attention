from datasets import load_dataset
from .dataset import TranslationDataset
from torch.utils.data import DataLoader

def make_dataloaders(tokenizer, device, batch_size=16, test_size=0.2):
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=test_size)

    train_data = books['train']
    valid_data = books['test']

    train_dataset = TranslationDataset(train_data, tokenizer, device)
    valid_dataset = TranslationDataset(valid_data, tokenizer, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, val_loader