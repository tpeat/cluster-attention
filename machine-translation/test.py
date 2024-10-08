def make_tokenizer():
    from data.tokenizer import make_tokenizer
    tokenizer_name = 'MBart'
    tokenizer = make_tokenizer(tokenizer_name)
    return tokenizer

def tokenizer_tests():
    tokenizer = make_tokenizer()

    def test_differences_in_context_management():
        with tokenizer.as_target_tokenizer():
            print(tokenizer.encode("Je déteste ces conneries"))
        print(tokenizer.encode("Je déteste ces conneries"))

    def test_existence_of_lang_specific_ids():
        if tokenizer_name == "MBart":
            print([tokenizer.lang_code_to_id['fr_XX']])

    def test_attriute_existence():
        print(hasattr(tokenizer, 'cls_token_id'))

        print(hasattr(tokenizer, 'sep_token_id'))

        print(hasattr(tokenizer, 'bos_token_id'))

        print(hasattr(tokenizer, 'eos_token_id'))
        print(tokenizer.cls_token_id)
        print(tokenizer.sep_token_id)
        print(tokenizer.bos_token_id)
        print(tokenizer.eos_token_id)
        print(tokenizer.pad_token_id)

    def test_special_maps():
        print("Special Tokens Map:", tokenizer.special_tokens_map)
        print("All Special Token IDs:", tokenizer.all_special_ids)


    def test_creating_similar_sentence():
        sentence = "Hello I like to eat"

        if tokenizer_name == "MBart":
            sentence = [tokenizer.lang_code_to_id['en_XX']] + sentence
            print(sentence)

def test_translation():
    tokenizer = make_tokenizer()
    from layers.transformer import Transformer

    src_vocab_size, tgt_vocab_size = tokenizer.vocab_size, tokenizer.vocab_size

    device='cuda'
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


    def translate(source_sentence, model, tokenizer, device, max_len=50):
        # Tokenize the source sentence
        src_tokens = tokenizer.encode(source_sentence, return_tensors='pt').to(device)  # Shape: (1, src_len)
        
        end_symbol = tokenizer.eos_token_id
        
        # Generate translated tokens using greedy decoding
        translated_tokens = model.greedy_decode(src_tokens, max_len, end_symbol)
        
        # Convert token IDs to tokens and then to string
        translated_sentence = tokenizer.decode(translated_tokens, skip_special_tokens=True)
        
        return translated_sentence

    # Example usage:
    source_sentence = "Hello, how are you?"
    translated_sentence = translate(source_sentence, model, tokenizer, device)
    print(f"Translated Sentence: {translated_sentence}")



def test_dataset():
    from data.load_data import make_dataloaders

    tokenizer = make_tokenizer()

    trainloader, testloader = make_dataloaders(tokenizer)
    batch = next(iter(trainloader))
    print(batch['input_ids'][0])
    print(batch['labels'][0])
    


if __name__ == "__main__":
    test_dataset()