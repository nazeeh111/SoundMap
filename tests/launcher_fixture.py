import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]


def run_launcher(bokeh, flask=None, model=False, stop_signal=None, timeout=20):
    """Run the real public launchers with controlled children, never devices."""
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        (fixture / "bokeh" / "server").mkdir(parents=True)
        for name in ["bokeh/__init__.py", "bokeh/server/__init__.py", "bokeh/server/server.py"]:
            (fixture / name).write_text("")
        (fixture / "sitecustomize.py").write_text(textwrap.dedent("""
            import json
            import os
            import sys
            import types
            if os.path.basename(sys.argv[0]) == 'flask_app.py':
                os.execv(sys.executable, [sys.executable,
                    os.environ['SOUNDMAP_TEST_FLASK'], *sys.argv])
            config = types.ModuleType('config')
            config.ConfigManager = lambda path: None
            sys.modules['config'] = config
            browser = types.ModuleType('webbrowser')
            def open_browser(url):
                with open(os.environ['SOUNDMAP_TEST_BROWSER'], 'a') as output:
                    output.write(json.dumps(url) + '\\n')
                return False
            browser.open = open_browser
            sys.modules['webbrowser'] = browser
        """))
        child_program = textwrap.dedent("""
            import json
            import os
            from pathlib import Path
            import signal
            import sys
            import time
            kind = KIND
            root = Path(os.environ['SOUNDMAP_TEST_FIXTURE'])
            behavior = json.loads(os.environ['SOUNDMAP_TEST_BEHAVIORS'])[kind]
            (root / (kind + '.json')).write_text(json.dumps({
                'pid': os.getpid(), 'parent': os.getppid(),
                'args': sys.argv[1:], 'cwd': os.getcwd()}))
            def stop(signum, frame):
                (root / (kind + '.stopped')).write_text(str(signum))
                raise SystemExit(0)
            signal.signal(signal.SIGTERM,
                signal.SIG_IGN if behavior.get('stubborn') else stop)
            mode = behavior.get('mode', 'exit')
            if mode == 'signal':
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)
            if mode == 'burst':
                sys.stdout.write('O' * 524288)
                sys.stdout.flush()
                sys.stderr.write('E' * 524288)
                sys.stderr.flush()
            if mode in ('after_browser', 'after_flask'):
                event = root / ('browser.jsonl' if mode == 'after_browser' else 'flask.json')
                while not event.exists():
                    time.sleep(0.02)
            if mode == 'wait':
                while True:
                    time.sleep(0.1)
            raise SystemExit(behavior.get('code', 0))
        """
        )
        (fixture / "bokeh" / "__main__.py").write_text(child_program.replace("KIND", repr("bokeh")))
        (fixture / "fake_flask.py").write_text(child_program.replace("KIND", repr("flask")))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(fixture), environment.get("PYTHONPATH")]))
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["SOUNDMAP_TEST_FIXTURE"] = str(fixture)
        environment["SOUNDMAP_TEST_FLASK"] = str(fixture / "fake_flask.py")
        environment["SOUNDMAP_TEST_BROWSER"] = str(fixture / "browser.jsonl")
        environment["SOUNDMAP_TEST_BEHAVIORS"] = json.dumps({"bokeh": bokeh, "flask": flask})
        arguments = [] if flask is not None else ["--no-flask"]
        if model:
            arguments.extend(["--model", "model directory/checkpoint.keras"])
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "soundmap.py"), "run", *arguments],
            cwd=fixture, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            if stop_signal is not None:
                wanted = fixture / ("flask.json" if flask is not None else "bokeh.json")
                deadline = time.monotonic() + 8
                while not wanted.exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise AssertionError("Controlled child did not start")
                    time.sleep(0.02)
                process.send_signal(stop_signal)
            timed_out = False
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=5)
            records = {kind: json.loads((fixture / (kind + ".json")).read_text())
                       for kind in ["bokeh", "flask"] if (fixture / (kind + ".json")).exists()}
            browsers = []
            if (fixture / "browser.jsonl").exists():
                browsers = [json.loads(line) for line in (fixture / "browser.jsonl").read_text().splitlines()]
            alive = []
            for kind, record in records.items():
                try:
                    os.kill(record["pid"], 0)
                except ProcessLookupError:
                    continue
                alive.append(kind)
            return {"code": process.returncode, "timed_out": timed_out,
                    "children": records, "browsers": browsers,
                    "alive": alive, "stdout": stdout, "stderr": stderr,
                    "stopped": [kind for kind in records if (fixture / (kind + ".stopped")).exists()]}
        finally:
            # This isolated test group contains only its controlled processes.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
