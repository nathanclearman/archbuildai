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
    PROMPT_OVERHEAD_TOKENS,
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
    """The filter budget now subtracts PROMPT_OVERHEAD_TOKENS before
    multiplying by the chars-per-token floor. This is the fix for the
    step-496 crash: the previous formula `max_length * 4.0` ignored
    ~1500 tokens of prompt overhead (system + user + image tokens),
    admitting samples that tokenized over budget.
    """

    def _expected_budget(self, max_length):
        return int(max(0, max_length - PROMPT_OVERHEAD_TOKENS) * MAX_TARGET_CHARS_PER_TOKEN)

    def test_keeps_short_samples(self):
        # max_length = 6144 (new default); overhead 1500 → 4644 avail ×
        # 3.0 chars/token = 13,932 char budget. 100-char samples pass.
        samples = [_FakeSample("x" * 100) for _ in range(3)]
        kept, dropped, budget = _filter_oversized_samples(samples, max_length=6144)
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, 0)
        self.assertEqual(budget, self._expected_budget(6144))

    def test_drops_oversized_samples(self):
        # Budget at max_length=4000: (4000-1500)*3.0 = 7500 chars.
        short = _FakeSample("x" * 100)
        big = _FakeSample("x" * 8000)
        kept, dropped, budget = _filter_oversized_samples([short, big], max_length=4000)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(budget, 7500)
        self.assertIs(kept[0], short)

    def test_boundary_sample_is_kept(self):
        # Exactly at the char budget is inclusive.
        budget = self._expected_budget(4000)  # 7500
        exactly = _FakeSample("x" * budget)
        kept, dropped, _ = _filter_oversized_samples([exactly], max_length=4000)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 0)

    def test_max_length_below_overhead_rejects_everything(self):
        # Pathological but defined: if someone sets max_length below
        # the prompt overhead, available tokens is clamped to 0 and
        # the char budget is 0 — every sample is rejected. The previous
        # formula would have accepted samples up to (max_length * 4)
        # chars even though there was no room for the target at all.
        tiny = _FakeSample("x")
        kept, dropped, budget = _filter_oversized_samples([tiny], max_length=500)
        self.assertEqual(len(kept), 0)
        self.assertEqual(dropped, 1)
        self.assertEqual(budget, 0)

    def test_regression_step_496_crash(self):
        # The exact scenario that crashed the user's 4-hour run: a 9500-
        # char target that the old filter (max_length*4 = 16384) admitted,
        # but that tokenized to 4101 total tokens under the 4096 budget.
        # Under the new filter with max_length=4096, overhead=1500:
        # available = 2596 tokens × 3.0 chars = 7788 char budget.
        # 9500 > 7788 → correctly rejected pre-flight.
        borderline = _FakeSample("x" * 9500)
        kept, dropped, _ = _filter_oversized_samples([borderline], max_length=4096)
        self.assertEqual(len(kept), 0, msg="regression: old formula admitted this")
        self.assertEqual(dropped, 1)


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
