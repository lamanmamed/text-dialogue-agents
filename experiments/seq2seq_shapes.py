"""Run one encoder and decoder step to show the tensor flow."""

import torch

from text_dialogue_agents.seq2seq import Seq2SeqDecoder, Seq2SeqEncoder, encoder_to_decoder_state

encoder = Seq2SeqEncoder(vocab_size=500, embedding_size=50, hidden_size=64)
decoder = Seq2SeqDecoder(vocab_size=500, embedding_size=50, encoder_hidden_size=64)
source = torch.randint(1, 500, (4, 20))
outputs, forward, backward = encoder(source)
hidden = encoder_to_decoder_state(forward, backward)
logits, hidden, attention = decoder(torch.ones((4, 1), dtype=torch.long), hidden, outputs)
print("Encoder outputs:", tuple(outputs.shape))
print("Decoder logits:", tuple(logits.shape))
print("Attention:", tuple(attention.shape))
