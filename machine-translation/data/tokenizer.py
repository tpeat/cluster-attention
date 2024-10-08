from transformers import BertTokenizer, MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer

def make_tokenizer(name="MBart"):
    if name == "Bert":
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    elif name == "MBart":
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
        tokenizer.src_lang = 'en_XX'
        tokenizer.tgt_lang = 'fr_XX'
    elif name == "opus":
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    else:
        print("Invalid tokenizer name")
    return tokenizer

def get_start_end_pad_tokens(tokenizer_name, tokenizer):
    if tokenizer_name == "Bert":
        start_symbol = tokenizer.cls_token_id
        end_symbol = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id
    elif tokenizer_name == "MBart":
        start_symbol = tokenizer.lang_code_to_id['fr_XX']
        end_symbol = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
    elif tokenizer_name == "opus":
        start_symbol = tokenizer.pad_token_id
        end_symbol = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
    return start_symbol, end_symbol, pad_token_id