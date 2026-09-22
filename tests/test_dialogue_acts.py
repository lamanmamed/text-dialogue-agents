import unittest

import numpy as np
import torch

from text_dialogue_agents.dialogue_acts import (
    BiLSTMDialogueActTagger,
    ContextCNNBiLSTM,
    balanced_class_weights,
    build_context_windows,
    pad_sequences,
    simplify_swda_tag,
)


class DialogueActTests(unittest.TestCase):
    def test_tag_simplification(self):
        self.assertEqual(simplify_swda_tag("ny^e"), "na")
        self.assertEqual(simplify_swda_tag("co"), "oo_co_cc")
        self.assertEqual(simplify_swda_tag("b^m"), "b^m")

    def test_padding(self):
        result = pad_sequences([[1, 2], [3]], 3)
        np.testing.assert_array_equal(result, [[1, 2, 0], [3, 0, 0]])

    def test_balanced_weights_upweight_rare_class(self):
        weights = balanced_class_weights([0, 0, 0, 1])
        self.assertGreater(weights[1], weights[0])

    def test_bilstm_output_shape(self):
        model = BiLSTMDialogueActTagger(30, 5, embedding_size=8, hidden_size=6)
        self.assertEqual(model(torch.randint(0, 30, (4, 7))).shape, (4, 5))

    def test_context_windows_pad_edges(self):
        x = np.array([[1, 1], [2, 2], [3, 3]])
        windows, labels = build_context_windows(x, [0, 1, 2], window_size=3)
        self.assertEqual(windows.shape, (3, 3, 2))
        np.testing.assert_array_equal(windows[0, 0], [0, 0])
        np.testing.assert_array_equal(windows[1, 1], [2, 2])
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_context_model_output_shape(self):
        model = ContextCNNBiLSTM(50, 6, embedding_size=10, filter_sizes=(2, 3), num_filters=4)
        x = torch.randint(0, 50, (3, 5, 8))
        self.assertEqual(model(x).shape, (3, 6))


if __name__ == "__main__":
    unittest.main()
