import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from transformers import BertTokenizer
from transformers import XLNetTokenizer

text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."

# ------------------------------
# Word-based tokens (NLTK)
# ------------------------------
print("Word Based tokens:-")
tokens = word_tokenize(text)
print("Tokens:", tokens)

# If you want token IDs for word tokens, you need a custom vocab:
vocab = {word: idx for idx, word in enumerate(set(tokens))}
token_ids = [vocab[w] for w in tokens]
print("Token IDs (custom):", token_ids)

print("\n")

# ------------------------------
# Subword tokens (BERT)
# ------------------------------
print("Sub-Word tokens:-")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

tokens = bert_tokenizer.tokenize(text)
token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)