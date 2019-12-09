class LAMAPooler(Module):
    """
    This class implements the low rank factorization multi-head self-attention mechanism detailed
    in the paper `Low Rank Factorization for Compact Multi-Head Self-Attention
    <https://arxiv.org/abs/1912.00835>`_ .
    
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
        # TODO (John): How to get max_len without asking for it explicitly?
        bias: bool = False 
    ) -> None:
        super().__init__()
        self._activation = (lambda x: x) if activation is None else activation
        self._normalize = normalize
        self._p = Parameter(torch.Tensor(input_dim, num_heads))
        self._q = Parameter(torch.Tensor(input_dim, num_heads))
        self._c = Parameter(torch.Tensor(input_dim, 1))

        if bias:
            self._bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._p)
        torch.nn.init.xavier_uniform_(self._q)
        torch.nn.init.xavier_uniform_(self._c)
        if self._bias is not None:
            self._bias.data.fill_(0)

    def forward(self, input, mask=None):
        similarities = self._forward_internal(input, mask)
        
        if self._normalize:
            similarities = F.softmax(similarities, dim=0)

        sentence_embedding_matrix = similarities @ input
        return torch.flatten(sentence_embedding_matrix)

    def _forward_internal(self, input, mask=None):
        # TODO (John): Missing L2 norm
        scores = self._activation((self._p.t() @ self._c) * (self._q.t() @ input.t()))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return scores