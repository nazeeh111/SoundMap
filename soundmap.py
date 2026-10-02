"""SoundMap: Microphone-array measurements into sound-source maps."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys

COMMANDS = {'run': 'acoustic-camera/start.py'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    script = root / COMMANDS[args.command]
    cwd = script.parent if args.command == "run" else root
    child = None
    stopping_signal = None
    previous_handlers = {}
    cleaned = True

    def forward_signal(signum, frame):
        nonlocal stopping_signal
        stopping_signal = signum
        if child is not None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, forward_signal)
        child = subprocess.Popen([sys.executable, str(script), *args.arguments], cwd=cwd)
        if stopping_signal is not None:
            child.send_signal(stopping_signal)
        status = child.wait()
    finally:
        if child is not None and child.poll() is None:
            try:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"Could not reap launcher: {type(error).__name__}", file=sys.stderr)
                cleaned = False
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if stopping_signal is not None:
        return 128 + stopping_signal
    status = status if status >= 0 else 128 - status
    return 1 if not cleaned and status == 0 else status


if __name__ == "__main__":
    raise SystemExit(main())
