import torch


class TestLAMAPooler(object):
    """Collects all unit tests for `modules.lama_pooler.LAMAPooler`.
    """
    def test_attributes_after_initialization(self, lama_pooler):
        args, lama_pooler = lama_pooler

        assert lama_pooler._activation == args['activation']
        assert lama_pooler._normalize == args['normalize']
        assert lama_pooler._p.size() == (args['input_dim'], args['num_heads'])
        assert lama_pooler._q.size() == (args['input_dim'], args['num_heads'])
        assert lama_pooler._c.size() == (args['input_dim'], 1)
        # TODO (John): Switch this when we figure out the bias and make it default to True
        assert lama_pooler._bias is None

    def test_output_size_without_mask(self, lama_pooler):
        args, lama_pooler = lama_pooler

        max_seq_len = 50  # Maximum length of the input sequence

        inputs = torch.randn(max_seq_len, args['input_dim'])
        pooled_output = lama_pooler(inputs)

        assert len(pooled_output.size()) == 1
        assert pooled_output.size(0) == (args['num_heads'] * args['input_dim'])
