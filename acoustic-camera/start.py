import subprocess
import sys
import time
import signal
import webbrowser
from config import ConfigManager
import argparse
import bokeh.server.server


CONFIG_PATH = "config/config.json"
config = ConfigManager(CONFIG_PATH)

flask_process = None
bokeh_process = None
received_signal = None

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Path to an explicit checkpoint (.keras)")
parser.add_argument("--no-flask", action="store_true", help="Start only the Bokeh app without Flask")
args, unknown = parser.parse_known_args()


def start_flask():
    global flask_process
    # Inherit output: unread PIPEs can block a child before it can exit.
    flask_process = subprocess.Popen([sys.executable, "scripts/flask_app.py"])
    print("Flask process started at http://127.0.0.1:5000.")


def start_bokeh():
    global bokeh_process
    command = [sys.executable, "-m", "bokeh", "serve"]
    if not args.no_flask:
        command.append("--allow-websocket-origin=127.0.0.1:5000")
    command.append("scripts/bokeh_app.py")
    if args.model:
        command.extend(["--args", "--model", args.model])
    elif not args.no_flask:
        command.append("--args")
    bokeh_process = subprocess.Popen(command)


def stop_processes():
    processes = [("Flask", flask_process), ("Bokeh", bokeh_process)]
    cleaned = True
    # Stop all owned children before waiting for either of them.
    for name, process in processes:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            except OSError as error:
                print(f"Could not terminate {name}: {type(error).__name__}", file=sys.stderr)
                cleaned = False
    for name, process in processes:
        if process is None:
            continue
        try:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"Could not reap {name}: {type(error).__name__}", file=sys.stderr)
            cleaned = False
    return cleaned


def handle_exit(signum, frame):
    global received_signal
    # Cleanup belongs to the main flow, not a reentrant signal handler.
    received_signal = signum


def child_exit_status():
    if received_signal is not None:
        return 128 + received_signal
    statuses = [process.poll() for process in (bokeh_process, flask_process)
                if process is not None]
    # A simultaneous orderly exit must not hide the other child's failure.
    for status in statuses:
        if status is not None and status != 0:
            return status if status >= 0 else 128 - status
    if 0 in statuses:
        return 0
    return None


def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    status = 1
    try:
        start_bokeh()
        time.sleep(2)
        status = child_exit_status()
        if status is None and not args.no_flask:
            start_flask()
            time.sleep(2)
            status = child_exit_status()
        if status is None:
            # This checks process survival, not HTTP or application readiness.
            print("Processes started. Press Ctrl+C to exit.")
            status = child_exit_status()
            if status is None:
                url = "http://localhost:5006/bokeh_app" if args.no_flask else "http://127.0.0.1:5000"
                webbrowser.open(url)
            while status is None:
                time.sleep(1)
                status = child_exit_status()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        status = 1
    finally:
        cleaned = stop_processes()
    if not cleaned and status == 0:
        return 1
    return status


if __name__ == "__main__":
    sys.exit(main())
