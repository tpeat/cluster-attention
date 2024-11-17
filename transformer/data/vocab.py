
import re
from collections import Counter


class Vocab():
    def __init__(self, en_token_to_id , fr_token_to_id, en_id_to_token, fr_id_to_token):
        self.en_token_to_id  = en_token_to_id
        self.fr_token_to_id = fr_token_to_id
        self.en_id_to_token = en_id_to_token
        self.fr_id_to_token = fr_id_to_token

def preprocess_text(text):
    """
    Preprocess text by:
    - Lowercasing
    - Removing punctuation
    - Splitting into tokens
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Simple whitespace-based tokenization
    return tokens


def combine_data(train_data, test_data, lang_key):
    """
    Combine translations from train and test datasets for a given language.
    """
    combined_texts = []
    for dataset in (train_data, test_data):
        for item in dataset:
            if lang_key in item['translation']:
                combined_texts.append(item['translation'][lang_key])
    return combined_texts

def build_vocabulary_from_datasets(train_data, test_data, lang_key, vocab_size=None):
    """
    Build a vocabulary from both train and test datasets for a specific language.
    
    Args:
        train_data: Training dataset (list of dicts).
        test_data: Testing dataset (list of dicts).
        lang_key: Language key to process (e.g., 'en' or 'fr').
        vocab_size: Optional maximum size of the vocabulary.
    
    Returns:
        token_to_id: Dictionary mapping tokens to IDs.
        id_to_token: Dictionary mapping IDs to tokens.
    """
    # Combine all translations for the given language
    print("building vocab...")
    combined_texts = combine_data(train_data, test_data, lang_key)
    
    # Tokenize all combined texts
    tokens = []
    for text in combined_texts:
        tokens.extend(preprocess_text(text))
    
    # Count token frequencies
    token_counts = Counter(tokens)
    
    # Keep the most frequent tokens
    if vocab_size:
        most_common_tokens = token_counts.most_common(vocab_size)
    else:
        most_common_tokens = token_counts.items()
    
    # Build token-to-ID and ID-to-token mappings
    token_to_id = {token: idx + 1 for idx, (token, _) in enumerate(most_common_tokens)}
    token_to_id["<UNK>"] = len(token_to_id) + 1  # Unknown token
    token_to_id["<PAD>"] = len(token_to_id) + 1  # Padding token
    token_to_id["<SOS>"] = len(token_to_id) + 1  # Start of sentence
    token_to_id["<EOS>"] = len(token_to_id) + 1 
    
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    
    return token_to_id, id_to_token


def tokenize_sentence(sentence, token_to_id, add_special_tokens=True):
    """
    Tokenizes a sentence and maps tokens to their corresponding IDs.
    
    Args:
        sentence (str): Input sentence to tokenize.
        token_to_id (dict): Mapping of tokens to IDs.
        add_special_tokens (bool): Whether to add <SOS> and <EOS> tokens.
    
    Returns:
        List[int]: List of token IDs.
    """
    # Preprocess the sentence into tokens
    tokens = preprocess_text(sentence) 
    
    # Map tokens to IDs, defaulting to <UNK> for unknown tokens
    unk_id = token_to_id["<UNK>"]
    token_ids = [token_to_id.get(token, unk_id) for token in tokens]
    
    # Add <SOS> and <EOS> tokens if specified
    if add_special_tokens:
        sos_id = token_to_id["<SOS>"]
        eos_id = token_to_id["<EOS>"]
        token_ids = [sos_id] + token_ids + [eos_id]
    
    return token_ids