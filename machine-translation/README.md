# Machine Translation

## Installation

```sh
pip install datasets
```

```python
from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")
# split dataset into train and test
books = books["train"].train_test_split(test_size=0.2)
```