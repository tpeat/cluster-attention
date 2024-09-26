# Machine Translation

This project implements a Transformer model from scratch using PyTorch for machine translation tasks, specifically translating from English to French. The code includes components such as positional encoding, multi-head attention, encoder and decoder layers, and supports training with customizable hyperparameters via command-line arguments. 

## Installation
I'm trying to accumlate a list of all the package that we need in `requirements.txt` but I already had a lot installed so keep updating requirements as you find packages we need


```sh
pip install datasets
```

Loading the data looks like this
```python
from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")
# split dataset into train and test
books = books["train"].train_test_split(test_size=0.2)
```

Was getting some errors with this on pace, tried installing with conda
`conda install -c huggingface -c conda-forge datasets`

But I actually couldn't verify if this was working either

So then I built it from scratch by cloning, see docs [here](https://huggingface.co/docs/datasets/en/installation)

## Directory layout

```
machine-translation/
├── data/
│   ├── load_data.py          # Data loading and preprocessing scripts
│   ├── dataset.py            # Custom implementation of machine translation dataset
│   ├── tokenizer.py          # Tokenizer initialization script
├── data/
│   ├── train.py              # contains train and val methods
├── layers/
│   ├── attention.py          # Where most of our *attention* will go!
│   ├── pos_encoding.py     
│   ├── ffn.py     
│   ├── encoder.py     
│   ├── decoder.py   
│   ├── transformer.py     
├── checkpoints/              # Directory to save model checkpoints
├── runs/                     # TensorBoard logs directory
├── main.py                   # Main training script
├── launch.sh                 # Shell script to run training
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
```

## Launching training

Launch training by submitting a slurm job with the `pace_launch.sh` script. A local launch script is also provided in `launch.py`. Both scripts contain hyperparameter args for tuning.

Change these variables as needed:


```sh
EXP_NAME='experiment name'
D_MODEL=256
NUM_LAYERS=4
NUM_HEADS=4
D_FF=1024
MAX_SEQ_LENGTH=100
EPOCHS=5
LEARNING_RATE=5e-5
BATCH_SIZE=16
DEVICE='cuda'  # Change to 'cpu' or 'mps' as needed
```

`EXP_NAME` is particularly useful to track what configuration you are using

### Monitoring

You can monitor training using tensorboard

```sh
tensorboard --logdir runs
```
And then click on the localhost link it returns
