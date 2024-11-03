import pandas as pd
import os
import torch
from bleu_score import get_bleu_score
from sequence_generator import SequenceGenerator, DecodingStrategy

def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model_path = "model_weights/model_epoch_0.pt"
    max_len = 128
    k = 5
    p = 0.9
    preds, refs, _ = get_bleu_score(model_path, device, max_len=max_len, k=k, p=p,
                                    decoding_strategy=DecodingStrategy.TOP_P,
                                    batch_size=16, validation=True)
    

if __name__ == '__main__':
    main()