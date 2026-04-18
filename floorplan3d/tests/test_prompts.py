"""
Guard the train/inference prompt invariant.

SYSTEM_PROMPT and USER_PROMPT were previously duplicated copy-paste
between train.py and inference.py. A drift between the two — even a
stray whitespace change on one side — silently degrades eval: the model
is supervised against one prefix but conditioned on another at
sampling time. These tests fail loudly the moment either import path
goes out of sync with the canonical constants in prompts.py.

Importing train.py pulls PIL (module-level), so we guard that import
with a skip — the prompts module itself must remain dep-free and
independently importable.
"""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))


class TestPromptsModule(unittest.TestCase):
    """The canonical prompts module must be dep-free and importable."""

    def test_prompts_module_has_no_heavy_deps(self):
        # Check in a fresh subprocess so we don't disturb this process's
        # module cache (other tests import `train` and `inference`, which
        # re-export the prompts by identity — popping `prompts` from
        # sys.modules mid-suite breaks that identity invariant).
        probe = (
            "import sys;"
            f"sys.path.insert(0, {str(MODEL_DIR)!r});"
            "import prompts;"
            "assert 'torch' not in sys.modules, 'prompts pulled torch';"
            "assert 'transformers' not in sys.modules, 'prompts pulled transformers';"
            "assert 'PIL' not in sys.modules, 'prompts pulled PIL';"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(
            result.returncode, 0,
            f"prompts must be dep-free. stdout={result.stdout!r} "
            f"stderr={result.stderr!r}",
        )

    def test_system_prompt_content_stable(self):
        import prompts  # type: ignore
        self.assertIsInstance(prompts.SYSTEM_PROMPT, str)
        # Substrings that matter for downstream behavior — the model is
        # explicitly told to emit JSON only. If any of these are removed
        # the generated output will regress to prose / fenced code.
        for required in (
            "floor plan vectorization",
            "scale",
            "walls",
            "doors",
            "windows",
            "rooms",
            "JSON only",
        ):
            self.assertIn(required, prompts.SYSTEM_PROMPT)

    def test_user_prompt_content_stable(self):
        import prompts  # type: ignore
        self.assertEqual(prompts.USER_PROMPT, "Vectorize this floor plan.")


class TestInferenceReexportsPrompts(unittest.TestCase):
    """inference.py must re-export the canonical prompts by identity.

    Identity (not equality) matters — an accidental copy-paste that
    reintroduces a local literal with the same content today would drift
    silently tomorrow. Identity fails the moment someone replaces the
    import with a literal.
    """

    def test_inference_reexports_by_identity(self):
        import prompts  # type: ignore
        import inference  # type: ignore
        self.assertIs(inference.SYSTEM_PROMPT, prompts.SYSTEM_PROMPT)
        self.assertIs(inference.USER_PROMPT, prompts.USER_PROMPT)


class TestTrainReexportsPrompts(unittest.TestCase):
    """train.py must re-export the canonical prompts by identity.

    train.py imports PIL at module scope so FloorPlanDS is picklable
    under spawn. Skip if PIL isn't installed in this env — the identity
    invariant is covered by inference in that case, and CI environments
    that can import PIL will still exercise it.
    """

    def test_train_reexports_by_identity(self):
        try:
            import PIL  # type: ignore  # noqa: F401
        except ImportError:
            self.skipTest("PIL not installed — train.py unimportable here")
        import prompts  # type: ignore
        train = importlib.import_module("train")
        self.assertIs(train.SYSTEM_PROMPT, prompts.SYSTEM_PROMPT)
        self.assertIs(train.USER_PROMPT, prompts.USER_PROMPT)


if __name__ == "__main__":
    unittest.main()
