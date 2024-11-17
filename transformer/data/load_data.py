from datasets import load_dataset
from torch.utils.data import DataLoader
from .dataset import TranslationDataset

from .vocab import build_vocabulary_from_datasets, Vocab


# Build English and French Vocabularies

def make_dataloaders(device, batch_size=16, test_size=0.2, return_datasets=False):
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=test_size)
    en_token_to_id, en_id_to_token = build_vocabulary_from_datasets(books["train"], books["test"], lang_key='en', vocab_size=50)
    fr_token_to_id, fr_id_to_token = build_vocabulary_from_datasets(books["train"], books["test"], lang_key='fr', vocab_size=50)
    vocab = Vocab(en_token_to_id, fr_token_to_id, en_id_to_token, fr_id_to_token)
    train_data = books['train']
    valid_data = books['test']

    train_dataset = TranslationDataset(train_data, device, en_token_to_id, fr_token_to_id)
    valid_dataset = TranslationDataset(valid_data, device, en_token_to_id,fr_token_to_id)

    if return_datasets: # required for distributed, to wrap as distsampler
        return train_dataset, valid_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, val_loader, vocab