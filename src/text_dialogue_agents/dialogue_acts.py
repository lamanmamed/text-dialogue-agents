"""Dialogue-act classification models based on the SWDA experiments."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def simplify_swda_tag(tag_text: str) -> str:
    """Collapse Switchboard tags to the 43-tag DAMSL-style mapping used in the experiments."""
    tags = re.split(r"\s*[,;]\s*", tag_text)
    simplified = []
    for tag in tags:
        if tag in ("qy^d", "qw^d", "b^m"):
            pass
        elif tag == "nn^e":
            tag = "ng"
        elif tag == "ny^e":
            tag = "na"
        else:
            tag = re.sub(r"(.)\^.*", r"\1", tag)
            tag = re.sub(r"[\(\)@*]", "", tag)
            if tag in ("qr", "qy"):
                tag = "qy"
            elif tag in ("fe", "ba"):
                tag = "ba"
            elif tag in ("oo", "co", "cc"):
                tag = "oo_co_cc"
            elif tag in ("fx", "sv"):
                tag = "sv"
            elif tag in ("aap", "am"):
                tag = "aap_am"
            elif tag in ("arp", "nd"):
                tag = "arp_nd"
            elif tag in ("fo", "o", "fw", '"', "by", "bc"):
                tag = 'fo_o_fw_"_by_bc'
        simplified.append(tag)
    return simplified[0]


def pad_sequences(sequences: Sequence[Sequence[int]], max_length: int, pad_id: int = 0) -> np.ndarray:
    """Right-pad or truncate token-id sequences to one length."""
    output = np.full((len(sequences), max_length), pad_id, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        values = list(sequence)[:max_length]
        output[row, : len(values)] = values
    return output


def balanced_class_weights(labels: Sequence[int], num_classes: int | None = None) -> np.ndarray:
    """Compute n_samples / (n_classes * class_count) for each observed class."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("labels cannot be empty")
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("Every requested class must occur at least once")
    return labels.size / (num_classes * counts.astype(np.float64))


class BiLSTMDialogueActTagger(nn.Module):
    """Two-layer BiLSTM that predicts one dialogue-act label from one utterance."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        *,
        embedding_size: int = 100,
        hidden_size: int | None = None,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()
        hidden_size = hidden_size or num_classes
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.bilstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids.long())
        _, (hidden, _) = self.bilstm(embedded)
        utterance = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(utterance)


def build_context_windows(
    utterances: Sequence[Sequence[int]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    window_size: int = 7,
    pad_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed windows of neighbouring utterances around each target utterance."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")
    x = np.asarray(utterances, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("utterances must have shape (n_utterances, sequence_length)")
    if len(x) != len(y):
        raise ValueError("utterances and labels must have the same length")

    half = window_size // 2
    padded_utterance = np.full(x.shape[1], pad_id, dtype=np.int64)
    windows = []
    for index in range(len(x)):
        window = []
        for offset in range(-half, half + 1):
            source = index + offset
            window.append(x[source] if 0 <= source < len(x) else padded_utterance)
        windows.append(window)
    return np.asarray(windows, dtype=np.int64), y.copy()


class ContextCNNBiLSTM(nn.Module):
    """Encode each utterance with a CNN, then model neighbouring utterances with a BiLSTM."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        *,
        embedding_size: int = 100,
        filter_sizes: Sequence[int] = (3, 4, 5),
        num_filters: int = 64,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, num_filters, kernel_size=(size, embedding_size)),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(),
                )
                for size in filter_sizes
            ]
        )
        self.utterance_projection = nn.Linear(num_filters * len(filter_sizes), embedding_size)
        self.utterance_dropout = nn.Dropout(dropout)
        self.context_lstm = nn.LSTM(
            embedding_size,
            embedding_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.context_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * embedding_size, num_classes)

    def _encode_utterances(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids.long()).unsqueeze(1)
        pooled = []
        for conv in self.conv_blocks:
            features = conv(embedded)
            pooled.append(F.max_pool2d(features, kernel_size=features.shape[2:]))
        combined = torch.cat(pooled, dim=1).squeeze(-1).squeeze(-1)
        return self.utterance_dropout(self.utterance_projection(combined))

    def forward(self, context_windows: torch.Tensor) -> torch.Tensor:
        if context_windows.ndim != 3:
            raise ValueError("context_windows must have shape (batch, window, sequence_length)")
        batch, window, sequence_length = context_windows.shape
        flat = context_windows.reshape(batch * window, sequence_length)
        utterance_vectors = self._encode_utterances(flat)
        utterance_vectors = utterance_vectors.reshape(batch, window, self.embedding_size)
        contextualized, _ = self.context_lstm(utterance_vectors)
        center = contextualized[:, window // 2, :]
        return self.classifier(self.context_dropout(center))


def per_class_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[int, float]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    result = {}
    for label in np.unique(true):
        mask = true == label
        result[int(label)] = float((pred[mask] == true[mask]).mean())
    return result
