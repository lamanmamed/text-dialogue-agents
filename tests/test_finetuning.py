import unittest

from text_dialogue_agents.finetuning import full_sft_settings, lora_settings, token_f1


class FineTuningTests(unittest.TestCase):
    def test_full_sft_learning_rate(self):
        self.assertEqual(full_sft_settings()["learning_rate"], 2e-5)

    def test_lora_recipe(self):
        adapter, training = lora_settings()
        self.assertEqual(adapter["r"], 8)
        self.assertEqual(adapter["lora_alpha"], 16)
        self.assertEqual(training["learning_rate"], 1e-4)

    def test_shared_f1_metric(self):
        self.assertEqual(token_f1("hello there", "hello there"), 1.0)


if __name__ == "__main__":
    unittest.main()
