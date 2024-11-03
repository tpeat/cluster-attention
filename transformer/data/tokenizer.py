from transformers import BertTokenizer, MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer


class Tokenizer:
    def __init__(self, name="opus"):
        self.tokenizer = self._make_tokenizer(name)

    def _make_tokenizer(self, name):
        if name == "Bert":
            return BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        elif name == "MBart":
            return MBartTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
        elif name == "opus":
            return AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        else:
            raise ValueError("Invalid tokenizer name provided")

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    def get_sos_token_id(self):
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        else:
            return self.get_pad_token_id()
    
    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size


