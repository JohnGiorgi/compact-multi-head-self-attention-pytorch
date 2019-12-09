import pytest
import torch

from modules.lama import LAMA
from modules.lama_pooler import LAMAPooler


@pytest.fixture
def lama():
    """Return a tuple of the args used to intialize ``LAMA`` and the initialized instance.
    """
    args = {
        'num_heads':  6,
        'input_dim':  128,
        'activation': torch.tanh,
        'normalize':  True,
        'bias':       False,
    }

    lama = LAMA(**args)

    return args, lama


@pytest.fixture
def lama_pooler():
    """Return a tuple of the args used to intialize ``LAMAPooler`` and the initialized instance.
    """
    args = {
        'num_heads':  6,
        'input_dim':  128,
        'activation': torch.tanh,
        'normalize':  True,
        'bias':       False,
    }

    lama_pooler = LAMAPooler(**args)

    return args, lama_pooler
