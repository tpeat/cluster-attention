from transformers import BertTokenizer

def make_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    return tokenizer