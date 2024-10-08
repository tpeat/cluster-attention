import torch
from data.tokenizer import make_tokenizer, get_start_end_pad_tokens
from layers.attention import make_attn_fn
from layers.transformer import Transformer

def load_model(model, model_path, device):
    # load model checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def translate(source_sentence, model, tokenizer, device, start_symbol, end_symbol, max_len=50, beam_width=5):
    """
    New decoder using different greedy decoder method
    """
    # encode
    src_tokens = tokenizer.encode(source_sentence, return_tensors='pt').to(device)
    # ifnerence op
    # translated_tokens = model.greedy_decode(src_tokens, max_len, end_symbol)
    translated_tokens = model.beam_search_decode(
        src=src_tokens,
        max_len=max_len,
        start_symbol=start_symbol,
        end_symbol=end_symbol,
        beam_width=beam_width
    )
    # decode   
    translated_sentence = tokenizer.decode(translated_tokens, skip_special_tokens=True)
    return translated_sentence


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print("Device:", device)
    
    print("Creating Tokenizer")
    tokenizer_name = 'MBart'
    tokenizer = make_tokenizer(tokenizer_name)
    start_symbol, end_symbol, padd_token_id = get_start_end_pad_tokens(tokenizer_name, tokenizer)

    src_vocab_size, tgt_vocab_size = tokenizer.vocab_size, tokenizer.vocab_size
    
    print("Building model")
    d_model=256
    num_layers=4
    num_heads=4
    d_ff=1024
    max_seq_length=128
    attn_type = "baseline"
    model = Transformer(
        tokenizer, src_vocab_size, tgt_vocab_size, d_model, num_layers,
        num_heads, d_ff, max_seq_length, attn_type, attn_type
    ).to(device)


    print("Loading model weights...")
    model_path = 'checkpoints/baseline_e30_mbart_model_epoch_20.pt'
    model = load_model(model, model_path, device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num params:", params)

    while True:
        sentence = input("Enter a sentence for prediction (or 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        translation = translate(sentence, model, tokenizer, device, start_symbol, end_symbol)
        # translation = generate_translation(model, sentence, tokenizer, device, max_seq_length)
        print(f"Prediction: {translation}\n")

if __name__ == "__main__":
    main()


