"""Check the two dialogue-act architectures on synthetic token IDs."""

import torch

from text_dialogue_agents.dialogue_acts import BiLSTMDialogueActTagger, ContextCNNBiLSTM

utterances = torch.randint(1, 100, (8, 30))
baseline = BiLSTMDialogueActTagger(vocab_size=100, num_classes=43)
print("Utterance model:", tuple(baseline(utterances).shape))

windows = torch.randint(1, 100, (8, 7, 30))
context = ContextCNNBiLSTM(vocab_size=100, num_classes=43)
print("Context model:", tuple(context(windows).shape))
