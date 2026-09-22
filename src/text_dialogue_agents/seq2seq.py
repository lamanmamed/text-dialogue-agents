"""GRU encoder-decoder components and attention for response generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


class Seq2SeqEncoder(nn.Module):
    """Two-layer bidirectional GRU encoder used in the dialogue-generation experiments."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        *,
        dropout: float = 0.2,
        padding_idx: int = 0,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding or nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, token_ids: torch.Tensor, hidden: torch.Tensor | None = None):
        embedded = self.dropout(self.embedding(token_ids.long()))
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden[-2], hidden[-1]

    def initial_hidden(self, batch_size: int, *, device=None) -> torch.Tensor:
        return torch.zeros(4, batch_size, self.hidden_size, device=device)


class BahdanauAttention(nn.Module):
    """Additive attention over encoder time steps."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.score_projection = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, values: torch.Tensor):
        if query.ndim == 3:
            query = query.squeeze(0)
        score = self.score_projection(
            torch.tanh(self.query_projection(query.unsqueeze(1)) + self.value_projection(values))
        )
        weights = F.softmax(score, dim=1)
        context = torch.sum(weights * values, dim=1)
        return context, weights


class Seq2SeqDecoder(nn.Module):
    """Attention decoder with two GRU stages."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        encoder_hidden_size: int,
        *,
        dropout: float = 0.2,
        padding_idx: int = 0,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.state_size = 2 * encoder_hidden_size
        self.embedding = embedding or nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.attention = BahdanauAttention(self.state_size)
        self.gru1 = nn.GRU(embedding_size + self.state_size, self.state_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gru2 = nn.GRU(self.state_size, self.state_size, batch_first=True)
        self.output = nn.Linear(self.state_size, vocab_size)

    def forward(self, token_ids: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        query = hidden.squeeze(0) if hidden.ndim == 3 else hidden
        context, weights = self.attention(query, encoder_outputs)
        embedded = self.embedding(token_ids.long())
        decoder_input = torch.cat([context.unsqueeze(1), embedded], dim=-1)
        first, hidden1 = self.gru1(decoder_input, hidden if hidden.ndim == 3 else hidden.unsqueeze(0))
        second, hidden2 = self.gru2(self.dropout(first), hidden1)
        return self.output(second.squeeze(1)), hidden2, weights


def encoder_to_decoder_state(forward_hidden: torch.Tensor, backward_hidden: torch.Tensor) -> torch.Tensor:
    return torch.cat([forward_hidden, backward_hidden], dim=-1).unsqueeze(0)


@dataclass
class BeamHypothesis:
    token_ids: list[int]
    negative_log_probability: float
    hidden: torch.Tensor
    ended: bool = False


def beam_search_decode(
    decoder: Seq2SeqDecoder,
    encoder_outputs: torch.Tensor,
    initial_hidden: torch.Tensor,
    *,
    start_id: int,
    end_id: int,
    beam_size: int = 5,
    max_steps: int = 30,
) -> list[int]:
    """Decode one example by keeping the beam_size lowest negative-log-probability paths."""
    if encoder_outputs.shape[0] != 1:
        raise ValueError("beam_search_decode currently expects batch size 1")
    live = [BeamHypothesis([start_id], 0.0, initial_hidden)]
    finished: list[BeamHypothesis] = []

    for _ in range(max_steps):
        candidates: list[BeamHypothesis] = []
        for hypothesis in live:
            if hypothesis.ended:
                candidates.append(hypothesis)
                continue
            token = torch.tensor([[hypothesis.token_ids[-1]]], device=encoder_outputs.device)
            logits, hidden, _ = decoder(token, hypothesis.hidden, encoder_outputs)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            values, indices = torch.topk(log_probs, beam_size)
            for value, index in zip(values.tolist(), indices.tolist()):
                candidates.append(
                    BeamHypothesis(
                        hypothesis.token_ids + [index],
                        hypothesis.negative_log_probability - value,
                        hidden,
                        index == end_id,
                    )
                )

        candidates.sort(key=lambda item: item.negative_log_probability)
        next_live = []
        for hypothesis in candidates[:beam_size]:
            if hypothesis.ended:
                finished.append(hypothesis)
            else:
                next_live.append(hypothesis)
        live = next_live
        if len(finished) >= beam_size or not live:
            break

    choices = finished or live
    if not choices:
        return []
    best = min(choices, key=lambda item: item.negative_log_probability)
    tokens = best.token_ids[1:]
    if end_id in tokens:
        tokens = tokens[: tokens.index(end_id)]
    return tokens
