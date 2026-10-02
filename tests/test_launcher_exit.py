from pathlib import Path
import signal
import unittest

from launcher_fixture import ROOT, run_launcher

WAIT = {"mode": "wait"}


class LauncherExitTests(unittest.TestCase):
    def check_result(self, result, code, combined=False, model=False, browsers=0):
        self.assertFalse(result["timed_out"], "Launcher hung after a controlled exit or shutdown signal")
        self.assertEqual(result["code"], code)
        self.assertEqual(result["alive"], [], "Owned children survived launcher exit")
        expected = ["serve"]
        if combined:
            expected.append("--allow-websocket-origin=127.0.0.1:5000")
        expected.append("scripts/bokeh_app.py")
        if model or combined:
            expected.append("--args")
        if model:
            expected.extend(["--model", "model directory/checkpoint.keras"])
        self.assertEqual(result["children"]["bokeh"]["args"], expected)
        for child in result["children"].values():
            self.assertEqual(Path(child["cwd"]), ROOT / "acoustic-camera")
        if "flask" in result["children"]:
            self.assertEqual(result["children"]["flask"]["args"], ["scripts/flask_app.py"])
        self.assertEqual(len(result["browsers"]), browsers)
        if browsers:
            url = "http://127.0.0.1:5000" if combined else "http://localhost:5006/bokeh_app"
            self.assertEqual(result["browsers"], [url] * browsers)

    def test_no_flask_failure_reaches_public_exit(self):
        self.check_result(run_launcher({"code": 7}), 7)

    def test_no_flask_model_failure_reaches_public_exit(self):
        self.check_result(run_launcher({"code": 7}, model=True), 7, model=True)

    def test_no_flask_success(self):
        self.check_result(run_launcher({"code": 0}), 0)

    def test_no_flask_model_success(self):
        self.check_result(run_launcher({"code": 0}, model=True), 0, model=True)

    def test_early_bokeh_failure_does_not_start_flask(self):
        result = run_launcher({"code": 7}, WAIT, model=True, timeout=8)
        self.check_result(result, 7, combined=True, model=True)
        self.assertNotIn("flask", result["children"])

    def test_flask_startup_failure_stops_bokeh(self):
        result = run_launcher(WAIT, {"code": 7}, timeout=8)
        self.check_result(result, 7, combined=True)
        self.assertIn("bokeh", result["stopped"])

    def test_later_bokeh_failure_stops_flask(self):
        result = run_launcher({"mode": "after_browser", "code": 7}, WAIT)
        self.check_result(result, 7, combined=True, browsers=1)
        self.assertIn("flask", result["stopped"])

    def test_later_flask_model_failure_stops_bokeh(self):
        result = run_launcher(WAIT, {"mode": "after_browser", "code": 7}, model=True)
        self.check_result(result, 7, combined=True, model=True, browsers=1)
        self.assertIn("bokeh", result["stopped"])

    def test_later_no_flask_model_failure_after_browser(self):
        result = run_launcher({"mode": "after_browser", "code": 7}, model=True)
        self.check_result(result, 7, model=True, browsers=1)

    def test_orderly_bokeh_exit_does_not_start_flask(self):
        result = run_launcher({"code": 0}, WAIT)
        self.check_result(result, 0, combined=True)
        self.assertNotIn("flask", result["children"])

    def test_nonzero_flask_exit_is_not_masked_by_bokeh_zero(self):
        result = run_launcher({"mode": "after_flask", "code": 0}, {"code": 7})
        self.check_result(result, 7, combined=True)

    def test_flask_output_is_drained_and_failure_preserved(self):
        result = run_launcher(WAIT, {"mode": "burst", "code": 7}, timeout=8)
        self.check_result(result, 7, combined=True)
        self.assertGreaterEqual(result["stdout"].count("O"), 524288)
        self.assertGreaterEqual(result["stderr"].count("E"), 524288)

    def test_signaled_child_status_is_normalized(self):
        self.check_result(run_launcher({"mode": "signal"}), 143)

    def test_sigterm_public_command_stops_no_flask_child(self):
        result = run_launcher(WAIT, stop_signal=signal.SIGTERM, timeout=8)
        self.check_result(result, 143)
        self.assertIn("bokeh", result["stopped"])

    def test_sigint_public_command_stops_both_children(self):
        result = run_launcher(WAIT, WAIT, stop_signal=signal.SIGINT)
        self.check_result(result, 130, combined=True)
        self.assertEqual(sorted(result["stopped"]), ["bokeh", "flask"])

    def test_stubborn_sibling_is_killed_and_reaped(self):
        result = run_launcher({"mode": "wait", "stubborn": True}, {"code": 7})
        self.check_result(result, 7, combined=True)


if __name__ == "__main__":
    unittest.main()
