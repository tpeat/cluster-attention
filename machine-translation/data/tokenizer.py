from transformers import BertTokenizer, MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer

def make_tokenizer(name="MBart"):
    if name == "Bert":
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    elif name == "MBart":
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
    elif name == "opus":
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    else:
        print("Invalid tokenizer name")
    return tokenizer