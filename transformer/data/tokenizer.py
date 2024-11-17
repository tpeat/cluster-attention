from transformers import BertTokenizer, MBart50TokenizerFast, AutoTokenizer


class Tokenizer:
    def __init__(self, name="MBart", src_lang="en_XX", tgt_lang="fr_XX"):
        self.name = name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = self._make_tokenizer(name)

    def _make_tokenizer(self, name):
        if name == "Bert":
            return BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        elif name == "MBart":
            tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            tokenizer.src_lang = "en_XX"  # Set source language to English
            tokenizer.tgt_lang = "fr_XX"
            return tokenizer
        elif name == "opus":
            return AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr", bos_token="<s>")
        else:
            raise ValueError("Invalid tokenizer name provided")

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    def get_sos_token_id(self):
        """This method is primarily used for getting the SOS to start decoding"""
        if self.name == "MBart":
            return self.tokenizer.lang_code_to_id[self.tgt_lang]
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
    
    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size


