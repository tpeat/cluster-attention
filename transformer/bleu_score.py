from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import os

from data.load_data import make_dataloaders
from data.tokenizer import Tokenizer
from model_training import TrainingHyperParams, load_trained_model
from sequence_generator import SequenceGenerator, DecodingStrategy


def get_bleu_score(model_path, device, decoding_strategy=DecodingStrategy.GREEDY, max_len=1, k=None, p=None,
                   beam_width=None, batch_size=64, validation=True):
    TOKENIZER = Tokenizer()
    PAD_ID = TOKENIZER.get_pad_token_id()
    EOS_ID = TOKENIZER.get_eos_token_id()
    SOS_ID = TOKENIZER.get_sos_token_id()
    VOCAB_SIZE = TOKENIZER.get_vocab_size()
    training_hp = TrainingHyperParams()
    train_dataloader, valid_dataloader = make_dataloaders(tokenizer=TOKENIZER, device=device)
    

    transformer = load_trained_model(model_path=model_path)
    sg = SequenceGenerator(
        model=transformer,
        pad_token=PAD_ID,
        sos_token=SOS_ID,
        eos_token=EOS_ID,
        max_len=training_hp.max_padding
    )
    all_predictions = []
    all_references = []
    for i, data in tqdm(enumerate(valid_dataloader), desc='Decoding translations', total=len(valid_dataloader)):

        pred = sg.generate(
                src=data['src'],
                src_mask=data['src_mask'],
                strategy=decoding_strategy,
                k=k,
                p=p
        )
        candidate = [TOKENIZER.tokenizer.decode(id_list, skip_special_tokens=True) for id_list in pred]
        reference = [TOKENIZER.tokenizer.decode(id_list, skip_special_tokens=True) for id_list in data['tgt']]
        all_predictions.extend(candidate)
        if validation:
            all_references.extend(reference)

    # Calculate BLEU score
    bleu = None
    if validation: # if not validation set, we are not sharing references locally
        bleu = bleu_score(all_predictions, all_references)

    return all_predictions, all_references, bleu


def generate_test_set_predictions(model_path, device, file_name, decoding_strategy=DecodingStrategy.GREEDY, max_len=1,
                                  k=None, p=None, beam_width=None, batch_size=64):
    preds, refs, _ = get_bleu_score(model_path, device, max_len=max_len, k=k, p=p, beam_width=beam_width,
                                    decoding_strategy=decoding_strategy,
                                    batch_size=batch_size, validation=False)

    df = pd.DataFrame(preds, columns=['predicted'])
    path = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, f'{file_name}.csv'), index=False)


def bleu_score(reference_sentences, candidate_sentences):
    tokenized_references = [[nltk.word_tokenize(ref)] for ref in reference_sentences]
    tokenized_candidates = [nltk.word_tokenize(candidate) for candidate in candidate_sentences]
    average_bleu_score = corpus_bleu(tokenized_references, tokenized_candidates)
    return average_bleu_score * 100
