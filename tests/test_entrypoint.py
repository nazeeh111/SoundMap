import importlib.util
from pathlib import Path
import signal
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("entry", ROOT / "soundmap.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

class EntrypointTests(unittest.TestCase):
    def test_dispatch_preserves_arguments_and_failure_code(self):
        for mode, path in entry.COMMANDS.items():
            child = Mock()
            child.wait.return_value = 7
            child.poll.return_value = 7
            with self.subTest(mode=mode), patch.object(entry.subprocess, "Popen", return_value=child) as call:
                args = ["--input", "path with spaces/input.dat"]
                handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
                self.assertEqual(entry.main([mode, *args]), 7)
                for signum, handler in handlers.items():
                    self.assertEqual(signal.getsignal(signum), handler)
                self.assertEqual(call.call_args.args[0][2:], args)
                self.assertEqual(Path(call.call_args.args[0][1]), ROOT / path)
                expected = (ROOT / path).parent if mode == "run" else ROOT
                self.assertEqual(call.call_args.kwargs["cwd"], expected)

if __name__ == "__main__":
    unittest.main()
