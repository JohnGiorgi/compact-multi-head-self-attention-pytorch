from typing import Callable

import torch
from torch.nn import Linear
from torch.nn import Module

from .lama import LAMA


class LAMAEncoder(Module):
    """
    This class implements the low rank factorization multi-head self-attention mechanism described
    in `"Low Rank Factorization for Compact Multi-Head Self-Attention"
    <https://arxiv.org/abs/1912.00835>`_ by Mehta et al., 2019.

    Inputs:

    - inputs: shape ``(batch_size, max_sequence_length, input_dim)``
    - mask: shape ``(batch_size, max_sequence_length)``, should be 0 at timesteps where attention will be masked
        (e.g. pad tokens), and 1 otherwise.

    Output:

    - attention: shape ``(batch_size, num_heads, max_sequence_length)``.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    activation : ``Callable``, optional (default=``torch.tanh``)
        An activation function applied after the attention calculation. Default is
        ``torch.tanh``. Set to ``None`` to use linear activation (i.e. no activation).
    output_dim : ``Optional[int]``, optional (default=``None``)
        If not None, we'll apply the computed attention weights for each head to ``inputs``, concatenate the
        resulting features, and project them into a vector of this size, giving an output of
        ``(batch_size, output_dim)``. If this value is ``None``, we will just return the attention weights over
        each timestep in ``input``for each head, given an output of shape ``(batch_size, num_heads, max_seq_len)``.
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        activation: Callable = torch.tanh,
        output_dim: int = None
    ) -> None:
        super().__init__()
        self._attention = LAMA(num_heads, input_dim, activation, normalize=True)
        self._output_dim = output_dim

        if self._output_dim:
            self.projection_layer = Linear(num_heads * input_dim, self._output_dim)
        else:
            self.projection_layer = None

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None, pool: bool = False) -> torch.Tensor:
        self_attention_matrix = self._attention(inputs, mask)
        structured_sentence_embedding = self_attention_matrix @ inputs

        if self._output_dim:
            return self.projection_layer(structured_sentence_embedding.view(inputs.size(0), -1))
        else:
            return structured_sentence_embedding
