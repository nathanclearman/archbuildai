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

from train import (  # type: ignore
    FloorPlanDS,
    MAX_TARGET_CHARS_PER_TOKEN,
    _filter_oversized_samples,
    _find_subseq,
    _mask_prompt_cutoff,
    _split_eval,
)


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


class _FakeSample:
    """Stand-in for dataset.Sample with only the field the filter cares
    about. Avoids pulling the real Sample class (and its heavier deps)
    into a unit test that only exercises a char-length filter."""
    def __init__(self, target_json: str):
        self.target_json = target_json


class TestFilterOversizedSamples(unittest.TestCase):
    def test_keeps_short_samples(self):
        samples = [_FakeSample("x" * 100) for _ in range(3)]
        kept, dropped, budget = _filter_oversized_samples(samples, max_length=1024)
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, 0)
        self.assertEqual(budget, int(1024 * MAX_TARGET_CHARS_PER_TOKEN))

    def test_drops_oversized_samples(self):
        # budget = 100 * 4.0 = 400 chars.
        short = _FakeSample("x" * 100)
        big = _FakeSample("x" * 500)
        kept, dropped, budget = _filter_oversized_samples([short, big], max_length=100)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(budget, 400)
        self.assertIs(kept[0], short)

    def test_boundary_sample_is_kept(self):
        # Exactly at the char budget is inclusive. Predictable behaviour
        # matters more than the specific choice here.
        exactly = _FakeSample("x" * 400)
        kept, dropped, _ = _filter_oversized_samples([exactly], max_length=100)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 0)


class TestSplitEval(unittest.TestCase):
    """The eval split is the only way the Trainer distinguishes
    overfitting from generalisation; silently zero-sizing it would
    disable checkpoint selection without a visible error. These tests
    pin the edge-case handling the smoke inside train.main() can't
    exercise.
    """

    def test_empty_input_returns_empty_splits(self):
        train, evl = _split_eval([], 0.1, seed=0)
        self.assertEqual(train, [])
        self.assertEqual(evl, [])

    def test_standard_ten_percent_split(self):
        samples = list(range(100))
        train, evl = _split_eval(samples, 0.1, seed=0)
        self.assertEqual(len(train), 90)
        self.assertEqual(len(evl), 10)
        # Train and eval are disjoint: a sample that lands in eval
        # should never also land in train (and vice versa). The previous
        # implementation took a prefix slice off the upstream-shuffled
        # list and this test still passes under internal-shuffle too.
        self.assertEqual(set(train).intersection(evl), set())
        # Union covers every input sample — nothing dropped.
        self.assertEqual(sorted(train + evl), samples)

    def test_small_corpus_still_gets_nonzero_eval(self):
        # 5 samples * 0.1 = 0.5 → int(0.5) = 0. Floor to 1 so eval is
        # never empty on a non-empty corpus — otherwise the Trainer
        # silently skips all eval steps.
        train, evl = _split_eval(list(range(5)), 0.1, seed=0)
        self.assertEqual(len(evl), 1)
        self.assertEqual(len(train), 4)

    def test_single_sample_eval_split_yields_empty_train(self):
        # Pathological but legal: 1 sample, 0.1 split. Eval gets it,
        # train gets nothing. Documented edge case — build_samples
        # would normally raise long before we got here, but the split
        # shouldn't add its own opaque failure.
        train, evl = _split_eval([42], 0.1, seed=0)
        self.assertEqual(train, [])
        self.assertEqual(evl, [42])

    def test_same_seed_produces_same_split(self):
        # The split must be reproducible across runs for checkpoint
        # resumption to make sense — otherwise resuming a training run
        # might see a different eval set than the original.
        samples = list(range(50))
        a_train, a_eval = _split_eval(samples, 0.1, seed=42)
        b_train, b_eval = _split_eval(samples, 0.1, seed=42)
        self.assertEqual(a_train, b_train)
        self.assertEqual(a_eval, b_eval)

    def test_different_seed_produces_different_split(self):
        # A smoke check that seeding actually affects the shuffle —
        # guards against a future refactor that accidentally ignores
        # the seed kwarg.
        samples = list(range(50))
        _, eval_a = _split_eval(samples, 0.1, seed=0)
        _, eval_b = _split_eval(samples, 0.1, seed=1)
        self.assertNotEqual(eval_a, eval_b)

    def test_shuffle_does_not_mutate_input(self):
        # The previous prefix-slice implementation was view-safe by
        # accident; the new implementation shuffles a copy. Pin it so
        # a future optimisation that skips the copy doesn't silently
        # reorder the caller's list.
        samples = list(range(20))
        original = list(samples)
        _split_eval(samples, 0.1, seed=0)
        self.assertEqual(samples, original)


class TestFloorPlanDSPicklable(unittest.TestCase):
    """FloorPlanDS must be defined at module scope, not nested in main(),
    so it can be pickled when dataloader_num_workers > 0 under the
    `spawn` multiprocessing start method (macOS / Windows). A nested
    class raises PicklingError; a module-scope class round-trips.
    """

    def test_class_qualname_is_module_scoped(self):
        # A nested class would be <locals>.FloorPlanDS; module-scope is
        # just FloorPlanDS. This check catches a regression the moment
        # someone moves the class back inside main().
        self.assertEqual(FloorPlanDS.__qualname__, "FloorPlanDS")

    def test_class_is_picklable(self):
        import pickle
        pickle.loads(pickle.dumps(FloorPlanDS))


if __name__ == "__main__":
    unittest.main()
