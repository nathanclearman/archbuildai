"""
Tests for the LocalModelClient python-interpreter resolver.

The production bug this guards against: the previous default
`python_bin=sys.executable` silently routed Blender-initiated predicts to
Blender's bundled Python, which has no torch. The subprocess failed at
`import torch` with returncode=1 and the user saw a generic "Model
inference failed" instead of actionable guidance to set FP3D_PYTHON.
"""

import json
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


def _make_fake_daemon(response_payload=None):
    """Build a MagicMock standing in for subprocess.Popen(...)'s return
    value with daemon-protocol-compatible stdin/stdout/stderr streams.

    The mock answers ONE READY handshake on stderr and any number of
    successful JSON responses on stdout (one per request). Tests that
    need a different daemon behaviour (death between calls, malformed
    JSON, slow ready) build their own mock — this is the happy-path
    default.
    """
    from unittest.mock import MagicMock
    payload = response_payload if response_payload is not None else {"ok": True, "result": {}}
    response_line = json.dumps(payload) + "\n"

    m = MagicMock()
    m.poll.return_value = None  # alive
    m.returncode = None

    # stderr: one "READY\n" then empties (no further diagnostics).
    # The client's _wait_for_ready_unlocked stops on the first READY.
    m.stderr = MagicMock()
    m.stderr.readline.side_effect = ["READY\n"] + [""] * 1000

    # stdin: capture written bytes for assertions; .closed is needed
    # by _close_daemon_unlocked's "close if not already closed" check.
    m.stdin = MagicMock()
    m.stdin.closed = False
    m.stdin._written = []

    def stdin_write(s):
        m.stdin._written.append(s)
        return len(s)

    m.stdin.write.side_effect = stdin_write

    # stdout: same canned response per readline() call. _close_daemon
    # never reads stdout, so an EOF tail is unnecessary.
    m.stdout = MagicMock()
    m.stdout.readline.side_effect = [response_line] * 1000

    # wait() must not raise — the test path through close() calls it.
    m.wait.return_value = 0
    return m


class TestPredictForwardsFlags(unittest.TestCase):
    """LocalModelClient.predict assembles a subprocess command. The
    specific flags it forwards (--cv-only, --refine, --quantize) are
    the user-facing contract between the Blender add-on panel and
    inference.py. A regression that drops a flag (e.g. --quantize not
    being wired through) would silently OOM on a 16 GB GPU with the
    Blender user having no way to fix it from the UI.
    """

    def _run_predict_with_mocked_subprocess(self, **kwargs):
        """Invoke predict() with BOTH subprocess.run (one-shot CV path)
        and subprocess.Popen (daemon VLM path) faked. Captures the
        cmd from whichever path predict() actually takes so the same
        helper covers both contracts.
        """
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

        def fake_popen(cmd, **_):
            captured["cmd"] = cmd
            return _make_fake_daemon()

        with patch("local_model.subprocess.run", side_effect=fake_run), \
                patch("local_model.subprocess.Popen", side_effect=fake_popen), \
                patch("local_model.INFERENCE_SCRIPT") as script:
            script.exists.return_value = True
            script.__str__ = lambda self: "/fake/inference.py"
            try:
                client.predict(img_path, **kwargs)
            finally:
                # Reset the daemon handle so __del__ during teardown
                # doesn't try to wait/kill a MagicMock with non-mock
                # subprocess machinery underneath.
                client._daemon = None

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

    def test_serve_flag_in_daemon_path(self):
        # Daemon path must spawn with --serve. A regression that drops
        # this flag would have the daemon try one-shot inference with
        # no --image arg and exit immediately, looking to the client
        # like a startup failure.
        cmd = self._run_predict_with_mocked_subprocess()
        self.assertIn("--serve", cmd)

    def test_cv_only_path_does_not_use_serve(self):
        # CV-only path must NOT spawn the daemon — that's the whole
        # point of routing it through one-shot subprocess.run.
        cmd = self._run_predict_with_mocked_subprocess(cv_only=True)
        self.assertNotIn("--serve", cmd)
        self.assertIn("--cv-only", cmd)


class TestDaemonLifecycle(unittest.TestCase):
    """Daemon protocol contract: spawn once on first VLM predict(),
    reuse across subsequent calls, respawn on death, restart on
    quantize-config change, clean up on close().
    """

    def _make_client(self):
        """Build a client with mocked filesystem guards. Returns the
        client and a captured-state dict the tests inspect."""
        client = LocalModelClient(
            weights_dir=local_model.DEFAULT_WEIGHTS_DIR,
            python_bin="/dev/null/fake-python",
        )
        return client

    def _patches(self, popen_side_effect):
        """Yields the patch context manager set for daemon-path tests."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(
            patch("local_model.subprocess.Popen", side_effect=popen_side_effect)
        )
        script = stack.enter_context(patch("local_model.INFERENCE_SCRIPT"))
        script.exists.return_value = True
        script.__str__ = lambda self: "/fake/inference.py"
        return stack

    def _tmp_image(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            return f.name

    def test_daemon_reused_across_calls(self):
        # Two predict() calls must result in exactly ONE Popen — the
        # whole point of the daemon is to amortize the 30-90s load
        # cost across all calls in a session.
        spawn_count = [0]

        def fake_popen(_cmd, **_):
            spawn_count[0] += 1
            return _make_fake_daemon()

        img = self._tmp_image()
        client = self._make_client()
        try:
            with self._patches(fake_popen):
                client.predict(img)
                client.predict(img)
        finally:
            client._daemon = None

        self.assertEqual(
            spawn_count[0], 1,
            "daemon must be reused across predict() calls — got "
            f"{spawn_count[0]} Popen invocations for 2 predicts",
        )

    def test_daemon_respawns_after_death(self):
        # A daemon that died between calls (OOM, crash) must respawn
        # transparently. Without this, every Blender session would
        # require an explicit re-init after the first GPU OOM.
        spawn_count = [0]
        daemons: list = []

        def fake_popen(_cmd, **_):
            spawn_count[0] += 1
            d = _make_fake_daemon()
            daemons.append(d)
            return d

        img = self._tmp_image()
        client = self._make_client()
        try:
            with self._patches(fake_popen):
                client.predict(img)
                # Simulate daemon death between calls: poll() now
                # returns a non-None exit code.
                daemons[0].poll.return_value = 137  # SIGKILL
                client.predict(img)
        finally:
            client._daemon = None

        self.assertEqual(
            spawn_count[0], 2,
            "daemon must respawn after the previous one died",
        )

    def test_daemon_restarts_on_quantize_change(self):
        # quantize affects model loading, not per-request behavior.
        # A running daemon loaded with quantize=False cannot serve
        # a quantize=True request without restarting.
        spawn_count = [0]
        spawned_cmds: list = []

        def fake_popen(cmd, **_):
            spawn_count[0] += 1
            spawned_cmds.append(cmd)
            return _make_fake_daemon()

        img = self._tmp_image()
        client = self._make_client()
        try:
            with self._patches(fake_popen):
                client.predict(img, quantize=False)
                client.predict(img, quantize=True)
        finally:
            client._daemon = None

        self.assertEqual(spawn_count[0], 2)
        self.assertNotIn("--quantize", spawned_cmds[0])
        self.assertIn("--quantize", spawned_cmds[1])

    def test_request_payload_carries_image_and_refine(self):
        # Per-request fields (image path, refine flag) ride on the
        # JSON request, not CLI flags — the daemon parses them out
        # of the line written to its stdin.
        spawned: list = []

        def fake_popen(_cmd, **_):
            d = _make_fake_daemon()
            spawned.append(d)
            return d

        img = self._tmp_image()
        client = self._make_client()
        try:
            with self._patches(fake_popen):
                client.predict(img, refine=True)
        finally:
            client._daemon = None

        written = "".join(spawned[0].stdin._written)
        req = json.loads(written.strip())
        self.assertEqual(req["image"], img)
        self.assertTrue(req["refine"])

    def test_close_shuts_down_daemon(self):
        # close() must terminate the daemon — without it the daemon
        # outlives Blender (orphaned to init on Unix) and holds the
        # GPU pinned until the user manually kills it.
        spawned: list = []

        def fake_popen(_cmd, **_):
            d = _make_fake_daemon()
            spawned.append(d)
            return d

        img = self._tmp_image()
        client = self._make_client()
        with self._patches(fake_popen):
            client.predict(img)
            client.close()

        self.assertIsNone(client._daemon)
        # close() must close stdin (EOF signals the daemon to exit
        # its read loop) and wait for the process to terminate.
        spawned[0].stdin.close.assert_called()
        spawned[0].wait.assert_called()

    def test_error_response_raises(self):
        # Daemon responses with ok=false must surface as an exception
        # at the client. Silently returning the error dict would let
        # downstream code consume "rooms": missing as if the model
        # had returned an empty plan.
        def fake_popen(_cmd, **_):
            return _make_fake_daemon(
                response_payload={"ok": False, "error": "boom"}
            )

        img = self._tmp_image()
        client = self._make_client()
        try:
            with self._patches(fake_popen):
                with self.assertRaises(RuntimeError) as cm:
                    client.predict(img)
                self.assertIn("boom", str(cm.exception))
        finally:
            client._daemon = None

    def test_startup_failure_diagnoses_and_cleans_up(self):
        # A daemon that exits before printing READY (corrupt weights,
        # OOM at load) must produce a fixable error message AND leave
        # the client in a clean state so the next predict() can try
        # again (e.g. with --quantize set differently).
        def fake_popen(_cmd, **_):
            from unittest.mock import MagicMock
            m = MagicMock()
            m.poll.return_value = 1  # already exited
            m.returncode = 1
            m.stderr = MagicMock()
            # No READY; only a traceback to read on drain.
            m.stderr.readline.side_effect = [""]
            m.stderr.read.return_value = "CUDA out of memory\n"
            m.stdin = MagicMock()
            m.stdin.closed = False
            m.wait.return_value = 1
            return m

        img = self._tmp_image()
        client = self._make_client()
        with self._patches(fake_popen):
            with self.assertRaises(RuntimeError) as cm:
                client.predict(img)
            self.assertIn("CUDA out of memory", str(cm.exception))
            self.assertIsNone(client._daemon)


if __name__ == "__main__":
    unittest.main()


class TestShippablePaths(unittest.TestCase):
    """The shipped add-on must work without the repo: bundled runtime dir,
    Blender's own interpreter as a last-resort candidate, and a status probe
    the preferences panel can render."""

    def setUp(self):
        local_model._RESOLVED_PYTHON_CACHE.clear()

    def tearDown(self):
        local_model._RESOLVED_PYTHON_CACHE.clear()
        local_model.reconfigure()

    def test_no_home_directory_guessing(self):
        import inspect
        src = inspect.getsource(local_model._resolve_model_dir)
        self.assertNotIn("Desktop", src)
        self.assertNotIn("home()", src)

    def test_model_dir_env_override_and_reconfigure(self):
        with patch.dict("os.environ", {"FP3D_QWEN_MODEL_DIR": "/tmp/fp3d-runtime", "FP3D_WEIGHTS_DIR": "/tmp/fp3d-w"}):
            local_model.reconfigure()
            # compare resolved paths: macOS maps /tmp -> /private/tmp
            self.assertEqual(local_model.INFERENCE_SCRIPT, Path("/tmp/fp3d-runtime").resolve() / "inference.py")
            self.assertEqual(local_model.DEFAULT_WEIGHTS_DIR, Path("/tmp/fp3d-w").resolve())
        local_model.reconfigure()
        self.assertNotEqual(str(local_model.DEFAULT_WEIGHTS_DIR), "/tmp/fp3d-w")

    def test_bundled_vlm_dir_is_inside_the_addon(self):
        self.assertEqual(local_model.BUNDLED_VLM_DIR.name, "vlm")
        self.assertEqual(local_model.BUNDLED_VLM_DIR.parent.name, "blender_addon")

    def test_blender_python_is_tried_last_not_never(self):
        env_empty = {k: v for k, v in __import__("os").environ.items() if k != "FP3D_PYTHON"}
        probed = []

        def probe(c):
            probed.append(c)
            return c == sys.executable

        with patch.dict("os.environ", env_empty, clear=True), \
                patch("local_model._is_blender_python", return_value=True):
            resolved = _resolve_python_bin(probe=probe)
        self.assertEqual(resolved, sys.executable)
        self.assertEqual(probed[-1], sys.executable)
        self.assertEqual(probed[0], "python3")

    def test_base_model_cache_detection(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {"HF_HUB_CACHE": d}):
            self.assertFalse(local_model.is_base_model_cached("Qwen/Qwen2.5-VL-7B-Instruct"))
            snap = Path(d) / "models--Qwen--Qwen2.5-VL-7B-Instruct" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            self.assertTrue(local_model.is_base_model_cached("Qwen/Qwen2.5-VL-7B-Instruct"))

    def test_environment_status_shape(self):
        st = local_model.environment_status(probe=lambda c: False)
        for key in ("inference_script", "python", "python_error", "base_model_cached", "adapter", "ready", "base_model"):
            self.assertIn(key, st)
        self.assertIsNone(st["python"])
        self.assertIn("Preferences", st["python_error"])
        self.assertFalse(st["ready"])


class TestBackendSelection(unittest.TestCase):
    def test_env_override_wins(self):
        with patch.dict("os.environ", {"FP3D_VLM_BACKEND": "torch"}):
            self.assertEqual(local_model.preferred_backend(), "torch")
            self.assertIn("torch", local_model._probe_imports())
        with patch.dict("os.environ", {"FP3D_VLM_BACKEND": "mlx"}):
            self.assertEqual(local_model.preferred_backend(), "mlx")
            self.assertIn("mlx_vlm", local_model._probe_imports())

    def test_status_reports_backend_and_matching_model(self):
        with patch.dict("os.environ", {"FP3D_VLM_BACKEND": "mlx"}):
            st = local_model.environment_status(probe=lambda c: False)
        self.assertEqual(st["backend"], "mlx")
        self.assertEqual(st["base_model"], local_model.MLX_BASE_MODEL)
