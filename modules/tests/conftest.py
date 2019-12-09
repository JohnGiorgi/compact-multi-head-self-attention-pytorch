import pytest
import torch

from modules.lama import LAMA
from modules.lama_pooler import LAMAPooler


@pytest.fixture
def lama():
    """Return a tuple of the args used to intialize ``LAMA`` and the initialized instance.
    """
    # This nested function lets us build the object on the fly in our unit tests
    def _initialize(num_heads=6, input_dim=128, activation=torch.tanh, normalize=True, bias=False):

        args = {
            'num_heads':  num_heads,
            'input_dim':  input_dim,
            'activation': activation,
            'normalize':  normalize,
            'bias':       bias,
        }

        lama = LAMA(**args)

        return args, lama

    return _initialize


@pytest.fixture()
def lama_pooler():
    """Return a tuple of the args used to intialize ``LAMAPooler`` and the initialized instance.
    """
    def _initialize(num_heads=6, input_dim=128, activation=torch.tanh, normalize=True, bias=False):

        args = {
            'num_heads':  num_heads,
            'input_dim':  input_dim,
            'activation': activation,
            'normalize':  normalize,
            'bias':       bias,
        }

        lama_pooler = LAMAPooler(**args)

        return args, lama_pooler

    return _initialize
