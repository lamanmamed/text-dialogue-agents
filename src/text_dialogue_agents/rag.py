"""Prompt construction, retrieval-query rewriting, and lexical evaluation for dialogue RAG."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable


def format_dialogue_prompt(history: list[tuple[str, str]], current_query: str) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and knowledgeable assistant engaging in a multi-turn conversation. "
                "Provide factual, informative responses based on the conversation context."
            ),
        }
    ]
    for speaker, utterance in history:
        messages.append({"role": "user" if speaker == "A" else "assistant", "content": utterance})
    messages.append({"role": "user", "content": current_query})
    return messages


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = Counter(predicted) & Counter(gold)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall)


def evaluate_responses(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("predictions cannot be empty")
    scores = [token_f1(prediction, reference) for prediction, reference in zip(predictions, references)]
    result = {"average_f1": 100 * sum(scores) / len(scores)}
    try:
        import sacrebleu
    except ImportError:
        return result
    result["bleu"] = float(sacrebleu.corpus_bleu(predictions, [references]).score)
    return result


def passage_to_text(passage: dict, max_chars: int = 1200) -> str:
    title = str(passage.get("title", "")).strip()
    text = str(passage.get("text", passage.get("contents", ""))).strip()
    combined = f"Title: {title}\nText: {text}" if title else text
    return combined[:max_chars]


def build_retrieval_query(history: list[tuple[str, str]], current_query: str, max_turns: int = 6) -> str:
    """Baseline query: recent dialogue followed by the current user question."""
    recent = history[-max_turns:] if max_turns > 0 else history
    parts = [f"{'User' if speaker == 'A' else 'Assistant'}: {utterance.strip()}" for speaker, utterance in recent]
    parts.append(f"Current question: {current_query.strip()}")
    return "\n".join(parts)


def extract_recent_entities(
    history: list[tuple[str, str]],
    *,
    max_turns: int = 6,
    max_entities: int = 8,
) -> list[str]:
    """Extract simple capitalized-name candidates from recent turns."""
    recent = history[-max_turns:] if max_turns > 0 else history
    text = " ".join(utterance for _, utterance in recent)
    candidates = re.findall(r"\b(?:[A-Z][a-zA-Z&\-]*)(?:\s+[A-Z][a-zA-Z&\-]*)*\b", text)
    seen = set()
    entities = []
    for candidate in candidates:
        candidate = candidate.strip()
        if len(candidate) > 1 and candidate not in seen:
            seen.add(candidate)
            entities.append(candidate)
    return entities[:max_entities]


def resolve_simple_references(history: list[tuple[str, str]], current_query: str) -> str:
    """Resolve the pronouns handled by the improved query experiment."""
    query = current_query.strip()
    entities = extract_recent_entities(history, max_entities=10)
    if re.search(r"\bthese two\b", query, re.IGNORECASE) and len(entities) >= 2:
        query = re.sub(
            r"\bthese two\b",
            f"{entities[-2]} and {entities[-1]}",
            query,
            flags=re.IGNORECASE,
        )
    if entities:
        query = re.sub(r"\bhe\b", entities[-1], query, flags=re.IGNORECASE)
        query = re.sub(r"\bshe\b", entities[-1], query, flags=re.IGNORECASE)
    return query


def build_retrieval_query_v2(
    history: list[tuple[str, str]],
    current_query: str,
    *,
    max_turns: int = 6,
    max_chars: int = 1200,
) -> str:
    """Put a rewritten question and named entities before recent dialogue context."""
    recent = history[-max_turns:] if max_turns > 0 else history
    rewritten = resolve_simple_references(history, current_query)
    entities = extract_recent_entities(history, max_turns=max_turns)
    parts = [f"Current question: {rewritten}"]
    if entities:
        parts.append("Key entities: " + ", ".join(entities))
    parts.append("Recent dialogue:")
    parts.extend(
        f"{'User' if speaker == 'A' else 'Assistant'}: {utterance.strip()}"
        for speaker, utterance in recent
    )
    return "\n".join(parts)[:max_chars].strip()


def format_rag_prompt(
    history: list[tuple[str, str]],
    current_query: str,
    retrieved_passages: Iterable[str],
) -> list[dict]:
    evidence = "\n\n".join(retrieved_passages)
    system = (
        "You are a helpful assistant. Use the retrieved evidence when it is relevant. "
        "If the evidence does not support an answer, do not invent a fact."
    )
    if evidence:
        system += "\n\nRetrieved evidence:\n" + evidence
    messages = [{"role": "system", "content": system}]
    for speaker, utterance in history:
        messages.append({"role": "user" if speaker == "A" else "assistant", "content": utterance})
    messages.append({"role": "user", "content": current_query})
    return messages
