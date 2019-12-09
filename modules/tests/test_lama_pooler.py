import torch


class TestLAMAPooler(object):
    """Collects all unit tests for `modules.lama_pooler.LAMAPooler`.
    """
    def test_attributes_after_initialization(self, lama_pooler):
        args, lama_pooler = lama_pooler()

        assert lama_pooler._activation == args['activation']
        assert lama_pooler._normalize == args['normalize']
        assert lama_pooler._p.size() == (args['input_dim'], args['num_heads'])
        assert lama_pooler._q.size() == (args['input_dim'], args['num_heads'])
        assert lama_pooler._c.size() == (args['input_dim'], 1)
        # TODO (John): Switch this when we figure out the bias and make it default to True
        assert lama_pooler._bias is None

    def test_output_forward_without_mask(self, lama_pooler):
        args, lama_pooler = lama_pooler()

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        pooled_output = lama_pooler(inputs)

        assert pooled_output.size() == (batch_size, args['num_heads'] * args['input_dim'])
