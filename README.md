# Pytorch Implementation of Low Rank Factorization for Compact Multi-Head Self-Attention

This is a PyTorch implementation of the __L__ ow Rank F __a__ ctorization for Compact __M__ ulti-Head __A__ ttention (LAMA) mechanism and the corresponding pooler introduced in the paper: "[Low Rank Factorization for Compact Multi-Head Self-Attention](https://arxiv.org/abs/1912.00835)".

Note that I am _not_ one of the authors on the paper.

## Usage

The only dependency is PyTorch. Installation instructions can be found [here](https://pytorch.org/get-started/locally/).

### LAMA

```python
import torch
from lama import LAMA

num_heads = 8      # Number of attention heads
hidden_dim = 768   # Dimension of each tokens hidden representation
max_seq_len = 100  # Maximum length of the input sequence

# Create a random input sequence
inputs = torch.randn(max_seq_len, hidden_dim)  

# Initialize the pooler
lama = LAMA(num_heads, hidden_dim)

output = lama(inputs)
print(output.size())  # (num_heads,  max_seq_len)
```

### LAMAPooler

```python
import torch
from lama_pooler import LAMAPooler

num_heads = 8      # Number of attention heads
hidden_dim = 768   # Dimension of each tokens hidden representation
max_seq_len = 100  # Maximum length of the input sequence

# Create a random input sequence
inputs = torch.randn(max_seq_len, hidden_dim)  

# Initialize the pooler
lama_pooler = LAMAPooler(num_heads, hidden_dim)

pooled_output = lama_pooler(inputs)
print(pooled_output.size())  # num_heads * hidden_dim
```