import unittest

from text_dialogue_agents.rag import (
    build_retrieval_query_v2,
    extract_recent_entities,
    normalize_answer,
    resolve_simple_references,
    token_f1,
)


class RAGTests(unittest.TestCase):
    def test_answer_normalization(self):
        self.assertEqual(normalize_answer("The Hotel, Dubai!"), "hotel dubai")

    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("red blue", "red green"), 0.5)

    def test_entity_extraction_and_pronoun_rewrite(self):
        history = [("A", "Tell me about Sergio Santos."), ("B", "Sergio Santos played football.")]
        entities = extract_recent_entities(history)
        self.assertIn("Sergio Santos", entities)
        rewritten = resolve_simple_references(history, "What club did he leave?")
        self.assertNotIn(" he ", " " + rewritten.lower() + " ")

    def test_improved_query_starts_with_current_question(self):
        history = [("A", "Tell me about Gevora Hotel."), ("B", "It is in Dubai.")]
        query = build_retrieval_query_v2(history, "How tall is it?")
        self.assertTrue(query.startswith("Current question:"))
        self.assertIn("Key entities:", query)


if __name__ == "__main__":
    unittest.main()
