from transformers import BertTokenizer, MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer

def make_tokenizer(name="MBart"):
    if name == "Bert":
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    elif name == "MBart":
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
    elif name == "opus":
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
    else:
        print("Invalid tokenizer name")
    return tokenizer