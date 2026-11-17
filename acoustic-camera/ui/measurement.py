"""Own one live measurement until its processor has actually stopped."""
from threading import Event, Lock, Thread


class MeasurementOwner:
    # Every dashboard document in this process shares the physical acquisition.
    _acquisition_gate = Lock()

    def __init__(self, processor, *, gate=None):
        self.processor = processor
        self._gate = gate if gate is not None else self._acquisition_gate
        self._lock = Lock()
        self._state = 'idle'
        self._error = None
        self._thread = None
        self._stop = None
        self._mode = None

    def snapshot(self):
        with self._lock:
            return self._state, self._error

    def start(self, mode):
        if mode not in ('model', 'beamforming'):
            raise ValueError('Unknown measurement mode')
        with self._lock:
            # A failed startup or cleanup may have left processor-owned resources.
            if self._state != 'idle' or (self._thread and self._thread.is_alive()):
                return False
            if not self._gate.acquire(blocking=False):
                return False
            self._state = 'starting'
            self._error = None
            self._mode = mode
            self._stop = Event()
            self._thread = Thread(target=self._run, args=(mode, self._stop), daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._state = 'failed'
                self._gate.release()
                raise
            return True

    def stop(self):
        with self._lock:
            if self._state not in ('starting', 'running', 'stopping'):
                return False
            self._state = 'stopping'
            self._stop.set()
            # Only a fully started generation has a current processor event.
            if getattr(self, '_started', False):
                getattr(self.processor, self._mode + '_stop_event').set()
            return True

    def _run(self, mode, stop):
        try:
            getattr(self.processor, 'start_' + mode)()
            with self._lock:
                self._started = True
                if stop.is_set():
                    getattr(self.processor, mode + '_stop_event').set()
                else:
                    self._state = 'running'
            stop.wait()
            getattr(self.processor, 'stop_' + mode)()
        except Exception as exc:
            with self._lock:
                self._error = f'{type(exc).__name__}: {exc}'
                self._state = 'failed'
        else:
            with self._lock:
                self._state = 'idle'
                self._gate.release()
        finally:
            with self._lock:
                self._started = False
