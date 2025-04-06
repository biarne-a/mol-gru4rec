import unittest

from preprocess.__main__ import _generate_examples_from_complete_sequences


class TestGenerateExamplesFromCompleteSequences(unittest.TestCase):
    def test_generate_examples_with_proportion_sliding_window(self):
        complete_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_context_len = 5
        proportion_sliding_window = 1.0
        expected_output = [
            {"input_ids": [6, 7, 8, 9, 10], "sample_type": 2},
            {"input_ids": [5, 6, 7, 8, 9], "sample_type": 1},
            {"input_ids": [4, 5, 6, 7, 8], "sample_type": 0},
            {"input_ids": [1, 2, 3], "sample_type": 0},
        ]
        result = _generate_examples_from_complete_sequences(complete_sequence, max_context_len, proportion_sliding_window)
        self.assertEqual(result, expected_output)

    def test_generate_examples_with_sliding_window_step_size_override(self):
        complete_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_context_len = 5
        sliding_window_step_size_override = 2
        expected_output = [
            {"input_ids": [6, 7, 8, 9, 10], "sample_type": 2},
            {"input_ids": [5, 6, 7, 8, 9], "sample_type": 1},
            {"input_ids": [4, 5, 6, 7, 8], "sample_type": 0},
            {"input_ids": [2, 3, 4, 5, 6], "sample_type": 0},
            {"input_ids": [1, 2, 3, 4], "sample_type": 0},
            {"input_ids": [1, 2], "sample_type": 0},
        ]
        result = _generate_examples_from_complete_sequences(
            complete_sequence, max_context_len, sliding_window_step_size_override=sliding_window_step_size_override
        )
        self.assertEqual(result, expected_output)


if __name__ == '__main__':
    unittest.main()