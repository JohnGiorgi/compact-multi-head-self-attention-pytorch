import torch


class TestLAMAEncoder(object):
    """Collects all unit tests for `modules.lama_encoder.lama_encoder`.
    """
    def test_attributes_after_initialization(self, lama_encoder):
        args, lama_encoder = lama_encoder()

        assert lama_encoder._attention._activation == args['activation']
        assert lama_encoder._attention._p.size() == (args['input_dim'], args['num_heads'])
        assert lama_encoder._attention._q.size() == (args['input_dim'], args['num_heads'])

        assert lama_encoder._output_dim == args['output_dim']

    def test_output_shape_forward_without_mask_without_projection(self, lama_encoder):
        args, lama_encoder = lama_encoder(output_dim=None)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        output = lama_encoder(inputs)

        assert output.size() == (batch_size, args['num_heads'], args['input_dim'])

    def test_output_shape_forward_without_mask_with_projection(self, lama_encoder):
        args, lama_encoder = lama_encoder()

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        output = lama_encoder(inputs)

        assert output.size() == (batch_size, args['output_dim'])

    def test_output_shape_forward_with_mask_without_projection(self, lama_encoder):
        args, lama_encoder = lama_encoder(output_dim=None)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        mask = torch.ones(batch_size, max_seq_len)
        mask[:, -1] = 0  # Zero-out the last timestep of each sequence
        output = lama_encoder(inputs, mask)

        assert output.size() == (batch_size, args['num_heads'], args['input_dim'])

    def test_output_shape_forward_with_mask_with_projection(self, lama_encoder):
        args, lama_encoder = lama_encoder()

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        mask = torch.ones(batch_size, max_seq_len)
        mask[:, -1] = 0  # Zero-out the last timestep of each sequence
        output = lama_encoder(inputs, mask)

        assert output.size() == (batch_size, args['output_dim'])
