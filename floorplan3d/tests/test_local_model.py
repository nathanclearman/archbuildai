"""
Tests for the LocalModelClient python-interpreter resolver.

The production bug this guards against: the previous default
`python_bin=sys.executable` silently routed Blender-initiated predicts to
Blender's bundled Python, which has no torch. The subprocess failed at
`import torch` with returncode=1 and the user saw a generic "Model
inference failed" instead of actionable guidance to set FP3D_PYTHON.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "blender_addon" / "api"),
)

import local_model  # type: ignore
from local_model import (  # type: ignore
    _FALLBACK_PYTHON_CANDIDATES,
    _is_blender_python,
    _resolve_python_bin,
    LocalModelClient,
)


class TestIsBlenderPython(unittest.TestCase):
    """Pure predicate over an interpreter path."""

    def test_linux_blender_layout(self):
        self.assertTrue(
            _is_blender_python("/opt/blender-4.0/python/bin/python3.10")
        )

    def test_macos_blender_layout(self):
        self.assertTrue(
            _is_blender_python(
                "/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.10"
            )
        )

    def test_system_python_not_matched(self):
        self.assertFalse(_is_blender_python("/usr/bin/python3"))
        self.assertFalse(_is_blender_python("/opt/homebrew/bin/python3.11"))

    def test_user_path_with_blender_word_in_leaf_only(self):
        # User's own virtualenv named `python3-blender-work` sitting in a
        # non-Blender directory should NOT be classified as Blender's
        # interpreter. The matcher looks at ancestor parts, not the leaf.
        self.assertFalse(_is_blender_python("/home/nate/venvs/python3-blender"))

    def test_user_directory_containing_blender_does_match(self):
        # Conservative: any ancestor containing "blender" triggers the
        # probe/skip. False positives here just push the resolver to try
        # python3/python next, which is a safe fallback — false negatives
        # would silently route to a non-ML interpreter.
        self.assertTrue(
            _is_blender_python("/home/nate/blender-projects/.venv/bin/python")
        )


class TestResolvePythonBin(unittest.TestCase):
    """Resolution order: FP3D_PYTHON → sys.executable → python3 → python."""

    def setUp(self):
        # The module-level cache persists across calls; tests that inject
        # a custom probe bypass the cache by design, but we still clear
        # it in setUp so a prior test that somehow triggered the default
        # path can't leak a cached result into the next test.
        local_model._RESOLVED_PYTHON_CACHE.clear()

    def tearDown(self):
        local_model._RESOLVED_PYTHON_CACHE.clear()

    def test_env_var_wins_without_probing(self):
        # Explicit opt-in is trusted — no subprocess spawned. User took
        # the action; respect it. If the path is broken, predict() will
        # fail at run time with the actual stderr, not a probe summary.
        def probe_never_called(_):
            raise AssertionError("probe must not run when FP3D_PYTHON is set")

        with patch.dict("os.environ", {"FP3D_PYTHON": "/custom/py"}):
            self.assertEqual(
                _resolve_python_bin(probe=probe_never_called),
                "/custom/py",
            )

    def test_sys_executable_when_probe_passes(self):
        env_empty = {k: v for k, v in __import__("os").environ.items()
                     if k != "FP3D_PYTHON"}

        with patch.dict("os.environ", env_empty, clear=True), \
             patch("local_model._is_blender_python", return_value=False):
            resolved = _resolve_python_bin(probe=lambda _: True)
        self.assertEqual(resolved, sys.executable)

    def test_blender_python_is_skipped(self):
        # Inside Blender, sys.executable is Blender's bundled python. The
        # resolver must skip it and try python3 / python even though it
        # would probe-pass in a test environment.
        env_empty = {k: v for k, v in __import__("os").environ.items()
                     if k != "FP3D_PYTHON"}
        probed: list[str] = []

        def probe(candidate):
            probed.append(candidate)
            return candidate == "python3"

        with patch.dict("os.environ", env_empty, clear=True), \
             patch("local_model._is_blender_python", return_value=True):
            resolved = _resolve_python_bin(probe=probe)
        self.assertEqual(resolved, "python3")
        self.assertNotIn(sys.executable, probed)
        self.assertEqual(probed[0], "python3")

    def test_falls_through_to_python_when_python3_missing(self):
        env_empty = {k: v for k, v in __import__("os").environ.items()
                     if k != "FP3D_PYTHON"}

        def probe(candidate):
            return candidate == "python"

        with patch.dict("os.environ", env_empty, clear=True), \
             patch("local_model._is_blender_python", return_value=True):
            resolved = _resolve_python_bin(probe=probe)
        self.assertEqual(resolved, "python")

    def test_raises_with_blender_diagnosis_when_sys_exec_is_blender(self):
        # When sys.executable is Blender's bundled python, the error
        # message should name Blender so the user knows the default
        # won't work and to point FP3D_PYTHON at their ML env.
        env_empty = {k: v for k, v in __import__("os").environ.items()
                     if k != "FP3D_PYTHON"}
        with patch.dict("os.environ", env_empty, clear=True), \
             patch("local_model._is_blender_python", return_value=True):
            with self.assertRaises(RuntimeError) as cm:
                _resolve_python_bin(probe=lambda _: False)
        msg = str(cm.exception)
        self.assertIn("FP3D_PYTHON", msg)
        self.assertIn("Blender", msg)

    def test_raises_with_path_diagnosis_when_sys_exec_is_not_blender(self):
        # When sys.executable is a regular system python that just
        # happens to lack torch, the message must NOT accuse it of
        # being Blender's — that would send the user hunting for a
        # problem they don't have. It should name "PATH" / "ML env"
        # so the user knows to install or activate one.
        env_empty = {k: v for k, v in __import__("os").environ.items()
                     if k != "FP3D_PYTHON"}
        with patch.dict("os.environ", env_empty, clear=True), \
             patch("local_model._is_blender_python", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                _resolve_python_bin(probe=lambda _: False)
        msg = str(cm.exception)
        self.assertIn("FP3D_PYTHON", msg)
        self.assertNotIn("Blender", msg)
        self.assertIn("PATH", msg)

    def test_fallback_candidates_order(self):
        # python3 before python — Ubuntu 22.04 / macOS default; `python`
        # on some older distros still points at 2.x.
        self.assertEqual(_FALLBACK_PYTHON_CANDIDATES, ("python3", "python"))


class TestResolverCache(unittest.TestCase):
    """The resolver memoizes the default-probe path to avoid re-probing on
    every LocalModelClient() construction — the operator builds a fresh
    client per Generate click, so without the cache the user eats 15-45 s
    of `import torch` probe time on every click."""

    def setUp(self):
        local_model._RESOLVED_PYTHON_CACHE.clear()

    def tearDown(self):
        local_model._RESOLVED_PYTHON_CACHE.clear()

    def test_default_probe_is_memoized(self):
        env_empty = {k: v for k, v in __import__("os").environ.items()
                     if k != "FP3D_PYTHON"}
        calls = [0]

        def probe(_):
            calls[0] += 1
            return True

        # Monkey-patch the module's default probe so the function signature
        # default picks it up — `probe is _probe_python` is the cache gate.
        with patch.dict("os.environ", env_empty, clear=True), \
             patch("local_model._is_blender_python", return_value=False), \
             patch("local_model._probe_python", side_effect=probe):
            # Call twice with no explicit probe arg → uses the (patched)
            # module default → qualifies for caching.
            r1 = _resolve_python_bin()
            r2 = _resolve_python_bin()

        self.assertEqual(r1, r2)
        self.assertEqual(
            calls[0], 1,
            "resolver must probe only once under repeated default-probe calls",
        )

    def test_injected_probe_bypasses_cache(self):
        # Injected probes are test seams — every call must see a fresh
        # resolution. If the cache swallowed the second call, a test that
        # flips probe behaviour between calls would get a stale answer.
        env_empty = {k: v for k, v in __import__("os").environ.items()
                     if k != "FP3D_PYTHON"}
        calls = [0]

        def probe(_):
            calls[0] += 1
            return True

        with patch.dict("os.environ", env_empty, clear=True), \
             patch("local_model._is_blender_python", return_value=False):
            _resolve_python_bin(probe=probe)
            _resolve_python_bin(probe=probe)

        self.assertEqual(calls[0], 2)
        # And nothing should have leaked into the cache.
        self.assertFalse(local_model._RESOLVED_PYTHON_CACHE)

    def test_env_var_path_does_not_populate_cache(self):
        # FP3D_PYTHON is trusted; there's nothing to memoize on that
        # path (no probe was performed). Writing into the cache would
        # also tie the cached value to env=None, which is not what the
        # env-var branch returns.
        with patch.dict("os.environ", {"FP3D_PYTHON": "/custom/py"}):
            _resolve_python_bin()
        self.assertFalse(local_model._RESOLVED_PYTHON_CACHE)


class TestPredictForwardsFlags(unittest.TestCase):
    """LocalModelClient.predict assembles a subprocess command. The
    specific flags it forwards (--cv-only, --refine, --quantize) are
    the user-facing contract between the Blender add-on panel and
    inference.py. A regression that drops a flag (e.g. --quantize not
    being wired through) would silently OOM on a 16 GB GPU with the
    Blender user having no way to fix it from the UI.
    """

    def _run_predict_with_mocked_subprocess(self, **kwargs):
        """Invoke predict() with subprocess.run faked to capture the
        command. Returns the cmd list and the fake client (for path
        assertions). Uses a tmp file to satisfy the isfile check and
        a stub returning empty JSON so no real inference runs."""
        import tempfile
        from unittest.mock import MagicMock

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name

        # Weights dir and inference script must exist for predict() to
        # proceed past its guard clauses; use the real in-repo paths.
        client = LocalModelClient(
            weights_dir=local_model.DEFAULT_WEIGHTS_DIR,
            python_bin="/dev/null/fake-python",
        )

        captured: dict = {}

        def fake_run(cmd, **_):
            captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = "{}"
            result.stderr = ""
            return result

        with patch("local_model.subprocess.run", side_effect=fake_run), \
                patch("local_model.INFERENCE_SCRIPT") as script:
            script.exists.return_value = True
            script.__str__ = lambda self: "/fake/inference.py"
            client.predict(img_path, **kwargs)

        return captured["cmd"]

    def test_quantize_flag_forwarded(self):
        cmd = self._run_predict_with_mocked_subprocess(quantize=True)
        self.assertIn("--quantize", cmd)

    def test_quantize_default_is_false(self):
        # quantize=False is the important default: the primary inference
        # target (M4 Max) both doesn't need it and measurably loses
        # quality with it. A silent default of True would degrade every
        # Blender-initiated predict.
        cmd = self._run_predict_with_mocked_subprocess()
        self.assertNotIn("--quantize", cmd)

    def test_cv_only_and_refine_still_forwarded(self):
        # Regression guard: when adding --quantize we re-ran the flag
        # routing. Confirm the older flags still land in the command.
        cmd = self._run_predict_with_mocked_subprocess(cv_only=True, refine=True)
        self.assertIn("--cv-only", cmd)
        self.assertIn("--refine", cmd)


if __name__ == "__main__":
    unittest.main()
