from typing import Callable

import torch
import torch.nn.functional as F

from lama import LAMA


class LAMAPooler(LAMA):
    """
    A thin wrapper around ``LAMA``, which applies LAMA to ``input`` and returns the flattened
    structed sentence embedding matrix.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    activation : ``Callable``, optional (default=``torch.tanh``)
        An activation function applied after the attention calculation. Default is
        ``torch.tanh``. Set to ``None`` to use a linear activation (i.e. no activation).
    normalize : ``bool``, optional (default: ``True``)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for each attention head.  If false, this is just computing a similarity score.
    bias : TODO.
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        activation: Callable = torch.tanh,
        normalize: bool = True,
        bias: bool = False
    ) -> None:
        super().__init__(num_heads, input_dim, activation, normalize, bias)

    def forward(self, input, mask=None):
        similarities = self._forward_internal(input, mask)

        if self._normalize:
            similarities = F.softmax(similarities, dim=0)

        sentence_embedding_matrix = similarities @ input
        return torch.flatten(sentence_embedding_matrix)
