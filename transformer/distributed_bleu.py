import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd

from data.load_data import make_dataloaders
from data.tokenizer import Tokenizer
from distributed_training import TrainingHyperParams, load_trained_model
from sequence_generator import SequenceGenerator, DecodingStrategy

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed BLEU Score Calculation")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights.')
    parser.add_argument('--output_file', type=str, default='test_predictions.csv', help='Name of the output CSV file.')
    parser.add_argument('--decoding_strategy', type=str, default='GREEDY', choices=['GREEDY', 'BEAM_SEARCH', 'TOP_K', 'TOP_P'], help='Decoding strategy to use.')
    parser.add_argument('--max_len', type=int, default=1, help='Maximum length for generation.')
    parser.add_argument('--k', type=int, default=None, help='Top-k sampling parameter.')
    parser.add_argument('--p', type=float, default=None, help='Top-p (nucleus) sampling parameter.')
    parser.add_argument('--beam_width', type=int, default=None, help='Beam width for beam search.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU.')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend (default: nccl).')
    parser.add_argument('--port', type=str, default='12355', help='Port number for distributed training.')
    parser.add_argument('--early_termination', type=int, default=None, help='max iteration before termianting')
    return parser.parse_args()

def setup_distributed(rank, world_size, backend, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()


def convert_and_join(int_to_str_dict, list_of_lists):
    converted_list_of_lists = [[int_to_str_dict[num] for num in sublist] for sublist in list_of_lists]
    out = [' '.join(sublist) for sublist in converted_list_of_lists]
    # Remove special tokens
    out = [s.replace(BOS_WORD, '').replace(EOS_WORD, '').replace(BLANK_WORD, '').strip() for s in out]
    return out


def bleu_score(reference_sentences, candidate_sentences):
    """
    Calculate the corpus BLEU score using NLTK.
    """
    tokenized_references = [[nltk.word_tokenize(ref)] for ref in reference_sentences]
    tokenized_candidates = [nltk.word_tokenize(candidate) for candidate in candidate_sentences]
    average_bleu_score = corpus_bleu(tokenized_references, tokenized_candidates)
    return average_bleu_score * 100

def get_decoding_strategy(strategy_str):
    """
    Map string to DecodingStrategy enum.
    """
    strategy_mapping = {
        'GREEDY': DecodingStrategy.GREEDY,
        'BEAM_SEARCH': DecodingStrategy.BEAM_SEARCH,
        'TOP_K': DecodingStrategy.TOP_K,
        'TOP_P': DecodingStrategy.TOP_P
    }
    return strategy_mapping.get(strategy_str.upper(), DecodingStrategy.GREEDY)

def main():
    args = parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK']) # master gpu

    setup_distributed(rank, world_size, args.backend, args.port)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    early_termination = int(args.early_termination)

    # Only the master process will perform certain actions like saving files and printing
    is_master = rank == 0

    if is_master:
        print(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}")

    TOKENIZER = Tokenizer()
    PAD_ID = TOKENIZER.get_pad_token_id()
    EOS_ID = TOKENIZER.get_eos_token_id()
    SOS_ID = TOKENIZER.get_sos_token_id()
    VOCAB_SIZE = TOKENIZER.get_vocab_size()
    training_hp = TrainingHyperParams()

    # pass return dataset flag to get back datasets, pass cpu to not move anything to gpu yet
    # defaults to fr -> en
    train_dataset, valid_dataset = make_dataloaders(tokenizer=TOKENIZER, device='cpu', return_datasets=True)

    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True
    )

    transformer = load_trained_model(model_path=args.model_path)
    transformer.to(device)

    # Wrap the model with DDP
    transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    decoding_strategy = get_decoding_strategy(args.decoding_strategy)
    sg = SequenceGenerator(
        model=transformer.module,  # Access the original model inside DDP
        pad_token=PAD_ID,
        sos_token=SOS_ID,
        eos_token=EOS_ID,
        max_len=training_hp.max_padding
    )

    all_predictions = []
    all_references = []

    transformer.eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_dataloader, desc=f'Rank {rank}: Decoding translations', disable=not is_master)):
            if i == 0:
                print("Source Text:", TOKENIZER.tokenizer.decode(batch['src'][0], skip_special_tokens=True))
                print("Target Text:", TOKENIZER.tokenizer.decode(batch['tgt'][0], skip_special_tokens=True))
                
            src = batch['src'].to(device, non_blocking=True)
            src_mask = batch['src_mask'].to(device, non_blocking=True)
            tgt = batch['tgt']  # Keep on CPU

            if decoding_strategy == DecodingStrategy.BEAM_SEARCH:
                pred = sg.beam_search(
                    src=src,
                    src_mask=src_mask,
                    beam_width=args.beam_width,
                )
                pred = [pred]
            else:
                pred = sg.generate(
                    src=src,
                    src_mask=src_mask,
                    strategy=decoding_strategy,
                    k=args.k,
                    p=args.p,
                )
            # can also use the .join candiadates method
            with TOKENIZER.tokenizer.as_target_tokenizer():
                candidate = [TOKENIZER.tokenizer.decode(id_list, skip_special_tokens=True) for id_list in pred]
                reference = [TOKENIZER.tokenizer.decode(id_list, skip_special_tokens=True) for id_list in tgt]
                print(f"Number of candidates: {len(candidate)}")
                print(f"Number of references: {len(reference)}")

                for cand, ref in zip(candidate[:5], reference[:5]):
                    print(f"Candidate: {cand}")
                    print(f"Reference: {ref}")
                    print("-" * 50)

            all_predictions.extend(candidate)
            all_references.extend(reference)

            if early_termination is not None and i >= early_termination:
                break

    # Gather all predictions and references from all processes to the master process
    gathered_predictions = [None for _ in range(world_size)]
    gathered_references = [None for _ in range(world_size)]

    dist.barrier()  # Ensure all processes have finished generating predictions

    try:
        dist.all_gather_object(gathered_predictions, all_predictions)
        dist.all_gather_object(gathered_references, all_references)
    except:
        # Fallback for older PyTorch versions or if all_gather_object is unavailable
        if is_master:
            print("Error: all_gather_object is not supported in this PyTorch version.")
            print("Please upgrade to PyTorch 1.8 or higher.")
        cleanup_distributed()
        return

    if is_master:
        # Flatten the lists
        flat_predictions = [pred for sublist in gathered_predictions for pred in sublist]
        flat_references = [ref for sublist in gathered_references for ref in sublist]

        # Calculate BLEU score
        bleu = bleu_score(flat_references, flat_predictions)
        print(f"BLEU Score: {bleu:.2f}")

        # Save predictions to CSV
        df = pd.DataFrame(flat_predictions, columns=['predicted'])
        output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, args.output_file)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    # cleanup
    cleanup_distributed()

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
