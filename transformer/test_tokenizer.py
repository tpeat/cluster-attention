from data.load_data import make_dataloaders
import torch

def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) 
    train_loader, val_loader, vocab = make_dataloaders(device=device)
    
    for batch in train_loader:
        print("Source:", batch['src'].shape)
        print("Source Mask:", batch['src_mask'].shape)
        print("Target:", batch['tgt'].shape)
        print("Target Mask:", batch['tgt_mask'].shape)
        print("Number of Tokens:", batch['ntokens'])
        break  # Only print the first batch for verification
    
    print(vocab.en_token_to_id["<PAD>"])
    print(vocab.fr_token_to_id["<PAD>"])
    print( len(vocab.en_token_to_id))
    print(len(vocab.fr_token_to_id))
if __name__ == '__main__':
    main()