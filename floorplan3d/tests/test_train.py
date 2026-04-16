"""Unit tests for the label-masking pure helpers in train.py.

These helpers decide which input tokens contribute to the training loss.
A silent failure in either of them would mask nothing and teach the
model to reproduce the system prompt / image tokens as supervision —
a mediocre loss curve that looks like a data problem. The helpers are
testable without torch or transformers installed, so these tests run in
the same suite as the rest of the project.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from train import _find_subseq, _mask_prompt_cutoff  # type: ignore


class TestFindSubseq(unittest.TestCase):
    def test_locates_match_at_start(self):
        self.assertEqual(_find_subseq([1, 2, 3, 4], [1, 2]), 0)

    def test_locates_match_in_middle(self):
        self.assertEqual(_find_subseq([9, 1, 2, 3, 4], [2, 3]), 2)

    def test_locates_match_at_end(self):
        self.assertEqual(_find_subseq([1, 2, 3, 4], [3, 4]), 2)

    def test_returns_first_occurrence_when_repeated(self):
        self.assertEqual(_find_subseq([1, 2, 1, 2, 1, 2], [1, 2]), 0)

    def test_missing_returns_negative_one(self):
        self.assertEqual(_find_subseq([1, 2, 3], [9, 9]), -1)

    def test_empty_needle_returns_zero(self):
        # Matches Python's str.find convention. Downstream code treats
        # an empty needle as an error (see _mask_prompt_cutoff) so the
        # permissive behaviour here is fine in isolation.
        self.assertEqual(_find_subseq([1, 2, 3], []), 0)

    def test_needle_longer_than_haystack(self):
        self.assertEqual(_find_subseq([1], [1, 2, 3]), -1)


class TestMaskPromptCutoff(unittest.TestCase):
    def test_cutoff_is_past_the_boundary(self):
        # assistant_token_ids = [10, 11, 12] found at index 3 → cutoff 6.
        cutoff = _mask_prompt_cutoff(
            input_ids=[1, 2, 3, 10, 11, 12, 99, 100],
            assistant_token_ids=[10, 11, 12],
        )
        self.assertEqual(cutoff, 6)

    def test_raises_when_boundary_missing(self):
        # Silent fallback would train on prompt tokens; an explicit raise
        # is the only safe behaviour when the chat template drifts.
        with self.assertRaises(RuntimeError) as ctx:
            _mask_prompt_cutoff(
                input_ids=[1, 2, 3, 4, 5],
                assistant_token_ids=[99, 100],
            )
        self.assertIn("assistant-role boundary", str(ctx.exception))

    def test_raises_on_empty_assistant_tokens(self):
        # An empty token list would produce cutoff=0 and train on
        # everything including the system prompt. Tokenizer drift
        # producing a 0-length sequence is rare but real — raise early.
        with self.assertRaises(RuntimeError):
            _mask_prompt_cutoff(input_ids=[1, 2, 3], assistant_token_ids=[])


if __name__ == "__main__":
    unittest.main()
