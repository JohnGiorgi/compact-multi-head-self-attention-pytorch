import pytest
import torch

from modules.lama import LAMA
from modules.lama_encoder import LAMAEncoder


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
        }

        lama = LAMA(**args)

        return args, lama

    return _initialize


@pytest.fixture
def lama_encoder():
    """Return a tuple of the args used to intialize ``LAMA`` and the initialized instance.
    """
    # This nested function lets us build the object on the fly in our unit tests
    def _initialize(num_heads=6, input_dim=128, activation=torch.tanh, output_dim=64):

        args = {
            'num_heads':  num_heads,
            'input_dim':  input_dim,
            'activation': activation,
            'output_dim': output_dim
        }

        lama_encoder = LAMAEncoder(**args)

        return args, lama_encoder

    return _initialize
