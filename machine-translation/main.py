import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.load_data import make_dataloaders
from data.tokenizer import make_tokenizer
from layers.transformer import Transformer
from engine.train import train, validate

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Training Script')

    parser.add_argument('--exp_name', type=str, default='exp1', help='Experiment name')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in Encoder and Decoder')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of the feedforward network')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training, defautl apple')
    parser.add_argument('--tokenizer-name', type=str, default='MBart', help='Tokenizer Type')
    parser.add_argument('--attn-type', type=str, default='baseline', help='Name of attention mechanism')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # check if specified device is available, defautls to mps if not
    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'mps'
    print(f"Using device {device}")

    print("Loading data")
    tokenizer = make_tokenizer(args.tokenizer_name)
    train_loader, val_loader = make_dataloaders(tokenizer, batch_size=args.batch_size)

    src_vocab_size = tokenizer.vocab_size
    tgt_vocab_size = tokenizer.vocab_size

    print("Building model")
    model = Transformer(
        tokenizer, src_vocab_size, tgt_vocab_size, args.d_model, args.num_layers,
        args.num_heads, args.d_ff, args.max_seq_length, args.attn_type, args.attn_type
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(log_dir=f'runs/{args.exp_name}')

    print("Beginning training")
    ### MAIN TRAINING LOOP
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        global_step = train(model, train_loader, criterion, optimizer, epoch, device, writer, global_step, tgt_vocab_size)
        val_loss = validate(model, val_loader, criterion, device, writer, epoch, tgt_vocab_size)

        # Save the model if validation loss decreases
        # TODO: save every k epochs and save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoints/{args.exp_name}_model_epoch_{epoch}.pt')
            print(f'Model saved at epoch {epoch}')

    writer.close()

if __name__ == '__main__':
    main()