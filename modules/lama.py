import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter


class LAMA(Module):
    """
    This class implements the low rank factorization multi-head self-attention mechanism described
    in `"Low Rank Factorization for Compact Multi-Head Self-Attention"
    <https://arxiv.org/abs/1912.00835>`_ by Mehta et al., 2019.

    Inputs:

    - inputs: shape ``(batch_size, max_sequence_length, input_dim)``
    - mask: shape ``(batch_size, max_sequence_length)``, should be 0 at timesteps where attention
      will be masked (e.g. pad tokens), and 1 otherwise.

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
    normalize : ``bool``, optional (default: ``True``)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for each attention head.  If false, this is just computing a similarity score.
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        activation: Callable = torch.tanh,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self._activation = (lambda x: x) if activation is None else activation
        self._normalize = normalize
        self._p = Parameter(torch.Tensor(input_dim, num_heads))
        self._q = Parameter(torch.Tensor(input_dim, num_heads))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self._p, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self._q, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None, pool: bool = False) -> torch.Tensor:
        similarities = self._forward_internal(inputs, mask)

        if self._normalize:
            if mask is not None:
                # The 2nd dimension (num_heads) will be broadcasted
                similarities = similarities.masked_fill(mask.unsqueeze(1) == 0, -1e9)
            similarities = F.softmax(similarities, dim=-1)

        if pool:
            sequence_embedding_matrix = similarities @ inputs
            return sequence_embedding_matrix.view(inputs.size(0), -1)

        # If pool: (batch_size, num_heads * input_dim)
        # Else:    (batch_size, num_heads, max_seq_len)
        return similarities

    def _forward_internal(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # The global context vector for each input is the mean of the word embeddings
        if mask is not None:
            c = (torch.sum(inputs * mask.unsqueeze(-1), dim=1) /
                 torch.clamp(torch.sum(mask, dim=1, keepdims=True), min=1e-9))
        else:
            c = torch.mean(inputs, dim=1)

        # See Eq. 3.13 of https://arxiv.org/abs/1912.00835
        q_h = self._q.t() @ inputs.transpose(1, 2)
        p_c_g = self._p.t() @ c.unsqueeze(-1)

        alignment = self._activation(p_c_g * q_h)
        alignment /= torch.norm(alignment, dim=1, keepdim=True)  # l2 normalization

        # (batch_size, num_heads, max_seq_len)
        return alignment
