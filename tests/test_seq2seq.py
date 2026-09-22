import unittest

import torch

from text_dialogue_agents.seq2seq import BahdanauAttention, Seq2SeqDecoder, Seq2SeqEncoder, encoder_to_decoder_state


class Seq2SeqTests(unittest.TestCase):
    def test_attention_weights_sum_to_one(self):
        attention = BahdanauAttention(12)
        query = torch.randn(3, 12)
        values = torch.randn(3, 7, 12)
        context, weights = attention(query, values)
        self.assertEqual(context.shape, (3, 12))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3, 1))

    def test_encoder_decoder_shapes(self):
        encoder = Seq2SeqEncoder(40, 8, 6, dropout=0.0)
        decoder = Seq2SeqDecoder(40, 8, 6, dropout=0.0)
        source = torch.randint(0, 40, (2, 5))
        outputs, forward, backward = encoder(source)
        self.assertEqual(outputs.shape, (2, 5, 12))
        hidden = encoder_to_decoder_state(forward, backward)
        logits, next_hidden, weights = decoder(torch.ones((2, 1), dtype=torch.long), hidden, outputs)
        self.assertEqual(logits.shape, (2, 40))
        self.assertEqual(next_hidden.shape, (1, 2, 12))
        self.assertEqual(weights.shape, (2, 5, 1))


if __name__ == "__main__":
    unittest.main()
