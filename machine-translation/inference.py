import torch
from data.tokenizer import make_tokenizer
from layers.attention import make_attn_fn
from layers.transformer import Transformer

def generate_translation(model, src_sentence, tokenizer, device, max_length=50):
    """
    Generates a translation for the given source sentence using the Transformer model.
    
    Args:
        model (Transformer): The trained Transformer model.
        src_sentence (str): The source sentence to translate.
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding.
        device (str): The device to perform computations on.
        max_length (int): Maximum length of the generated translation.
        
    Returns:
        str: The translated sentence.
    """
    model.eval()
    
    # Tokenize the source sentence
    src_tokens = tokenizer.encode(src_sentence)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)  # Shape: [1, src_len]
    
    # Generate source mask
    src_mask = model.make_src_mask(src_tensor)
    
    # Encode the source sentence
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
    
    # Initialize the target sequence with the <sos> token
    tgt_tokens = [tokenizer.cls_token_id]
    tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)  # Shape: [1, 1]
    
    for _ in range(max_length):
        # Generate target mask
        tgt_mask = model.make_tgt_mask(tgt_tensor)
        
        # Decode the current target sequence
        with torch.no_grad():
            dec_output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
        
        # Get the logits for the last token
        last_token_logits = dec_output[:, -1, :]  # Shape: [1, tgt_vocab_size]
        
        # Predict the next token (greedy decoding)
        next_token_id = last_token_logits.argmax(dim=-1).item()
        
        # Append the predicted token to the target sequence
        tgt_tokens.append(next_token_id)
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        
        # If the <eos> token is generated, stop
        if next_token_id == tokenizer.sep_token_id:
            break
    
    translation_tokens = tgt_tokens[1:-1]  # Remove [CLS] and [SEP]
    translation = tokenizer.decode(translation_tokens, skip_special_tokens=True)
    return translation


def load_model(model, model_path, device):
    # load model checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def translate(source_sentence, model, tokenizer, device, start_symbol, end_symbol, max_len=50):
    """
    New decoder using different greedy decoder method
    """
    # encode
    src_tokens = tokenizer.encode(source_sentence, return_tensors='pt').to(device)
    # ifnerence op
    translated_tokens = model.greedy_decode(src_tokens, max_len, start_symbol, end_symbol) 
    # decode   
    translated_sentence = tokenizer.decode(translated_tokens, skip_special_tokens=True)
    return translated_sentence
    

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print("Device:", device)
    
    print("Creating Tokenizer")
    tokenizer = make_tokenizer('opus')
    src_vocab_size, tgt_vocab_size = tokenizer.vocab_size, tokenizer.vocab_size

    start_symbol = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else tokenizer.bos_token_id
    end_symbol = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else tokenizer.eos_token_id
    
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
    model_path = 'checkpoints/baseline_e30_v2_model_epoch_16.pt'
    model = load_model(model, model_path, device)

    while True:
        sentence = input("Enter a sentence for prediction (or 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        translation = translate(sentence, model, tokenizer, device, start_symbol, end_symbol)
        # translation = generate_translation(model, sentence, tokenizer, device, max_seq_length)
        print(f"Prediction: {translation}\n")

if __name__ == "__main__":
    main()


