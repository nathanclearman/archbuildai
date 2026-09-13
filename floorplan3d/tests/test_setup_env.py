"""Tests for the first-run environment setup helpers (api/setup_env.py)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blender_addon" / "api"))
import setup_env  # type: ignore


class TestPackageLists(unittest.TestCase):
    def test_every_requirement_has_an_import_name(self):
        for req, imp in setup_env.CV_PACKAGES + setup_env.VLM_PACKAGES:
            self.assertTrue(req and imp, (req, imp))
            self.assertNotIn(" ", imp)

    def test_transformers_pinned_below_5(self):
        req = next(r for r, i in setup_env.VLM_PACKAGES if i == "transformers")
        self.assertIn("<5", req)


class TestMissingPackages(unittest.TestCase):
    def test_probe_reports_only_missing(self):
        pkgs = [("json-is-stdlib", "json"), ("definitely-not-installed", "fp3d_no_such_module_xyz")]
        missing = setup_env.missing_packages(sys.executable, pkgs)
        self.assertEqual(missing, ["definitely-not-installed"])

    def test_bad_interpreter_reports_everything_missing(self):
        pkgs = [("a", "json"), ("b", "os")]
        self.assertEqual(setup_env.missing_packages("/nonexistent/python", pkgs), ["a", "b"])


class TestCommands(unittest.TestCase):
    def test_pip_command_targets_user_site(self):
        cmd = setup_env.pip_install_command("/py", ["torch>=2.3", "peft"])
        self.assertEqual(cmd[:4], ["/py", "-m", "pip", "install"])
        self.assertIn("--user", cmd)
        self.assertIn("--no-input", cmd)
        self.assertEqual(cmd[-2:], ["torch>=2.3", "peft"])

    def test_install_skips_pip_when_nothing_missing(self):
        log = []
        with patch.object(setup_env, "missing_packages", return_value=[]), \
                patch.object(setup_env, "run_streaming") as run:
            code = setup_env.install_packages("/py", setup_env.CV_PACKAGES, log.append)
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn("already installed", log[-1])

    def test_run_streaming_feeds_lines_and_returns_exit_code(self):
        log = []
        code = setup_env.run_streaming([sys.executable, "-c", "print('one'); print('two'); raise SystemExit(3)"], log.append)
        self.assertEqual(code, 3)
        self.assertEqual(log[1:], ["one", "two"])

    def test_download_uses_snapshot_download(self):
        captured = {}
        with patch.object(setup_env, "run_streaming", side_effect=lambda cmd, cb, env=None: captured.setdefault("cmd", cmd) and 0):
            setup_env.download_base_model("/py", print, model_id="Org/Model")
        self.assertEqual(captured["cmd"][0], "/py")
        self.assertIn("snapshot_download('Org/Model'", captured["cmd"][2])


if __name__ == "__main__":
    unittest.main()
