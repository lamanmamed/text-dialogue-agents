"""Show the difference between the original and rewritten retrieval queries."""

from text_dialogue_agents.rag import build_retrieval_query, build_retrieval_query_v2

history = [
    ("A", "Tell me about Sérgio Santos."),
    ("B", "Sérgio Santos is a footballer."),
    ("A", "He moved clubs in 2009."),
]
question = "What team did he move from?"
print("BASELINE QUERY\n", build_retrieval_query(history, question))
print("\nIMPROVED QUERY\n", build_retrieval_query_v2(history, question))
