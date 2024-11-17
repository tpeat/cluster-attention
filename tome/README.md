# PiToMe Token Merging Setup

Required download of PiToMe into directory where you are working
git clone

https://github.com/hchautran/PiToMe/blob/main/algo/pitome/patch/bert.py

## Example usage
```python
from transformers import AlbertForMaskedLM, AlbertTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

# Load the model and tokenizer
checkpoint = 'albert-xlarge-v2'
depth=24
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained(checkpoint)
alm2_config = AlbertConfig.from_pretrained(checkpoint, num_hidden_layers=depth)
model = AlbertModel.from_pretrained(checkpoint, config=alm2_config)

margin=0.99
apply_patch_to_albert(model, trace_source=False, prop_attn=True, margin=margin, alpha=1.0, use_attn=False)

# Prepare your input data
text = "This is a sample input text."
inputs = tokenizer(text, return_tensors='pt')

# Forward pass with PiToMe applied
outputs = model(**inputs, output_hidden_states=True, return_dict=True)
len(outputs.hidden_states)
```