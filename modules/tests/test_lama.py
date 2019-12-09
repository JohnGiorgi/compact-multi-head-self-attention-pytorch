import torch


class TestLAMA(object):
    """Collects all unit tests for `modules.lama.LAMA`.
    """
    def test_attributes_after_initialization(self, lama):
        args, lama = lama

        assert lama._activation == args['activation']
        assert lama._normalize == args['normalize']
        assert lama._p.size() == (args['input_dim'], args['num_heads'])
        assert lama._q.size() == (args['input_dim'], args['num_heads'])
        assert lama._c.size() == (args['input_dim'], 1)
        # TODO (John): Switch this when we figure out the bias and make it default to True
        assert lama._bias is None

    def test_output_size_without_mask(self, lama):
        args, lama = lama

        max_seq_len = 50  # Maximum length of the input sequence

        inputs = torch.randn(max_seq_len, args['input_dim'])
        output = lama(inputs)

        assert output.size() == (args['num_heads'], max_seq_len)
