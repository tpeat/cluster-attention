import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_model(local_checkpoint_path="pretrained_checkpoint"):

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_checkpoint_path, 
        output_attentions=True, 
        return_dict=True
    ).to(device)
    return model, tokenizer
