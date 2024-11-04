from datasets import load_dataset
from torch.utils.data import DataLoader

from .dataset import TranslationDataset

def make_dataloaders(tokenizer, device, batch_size=16, test_size=0.2, return_datasets=False, src_lang="fr", tgt_lang="en"):
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=test_size)

    train_data = books['train']
    valid_data = books['test']

    train_dataset = TranslationDataset(train_data, tokenizer, device, src_lang=src_lang, tgt_lang=tgt_lang)
    valid_dataset = TranslationDataset(valid_data, tokenizer, device, src_lang=src_lang, tgt_lang=tgt_lang)

    if return_datasets: # required for distributed, to wrap as distsampler
        return train_dataset, valid_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, val_loader