"""Queue cancellation contracts from actual methods, without device imports."""
import ast
import itertools
from pathlib import Path
import queue
import threading
import time
import types
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'acoustic-camera/data_processing/process.py'


def methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Processor')
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {'_csm_generator', '_predictor', 'stop_model'}]
    namespace = {'queue': queue, 'time': time}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class WatchedQueue(queue.Queue):
    def __init__(self):
        super().__init__(1)
        self.entered_full = threading.Event()
        self.timed_out = threading.Event()

    def put(self, item, *args, **kwargs):
        if self.full():
            self.entered_full.set()
        try:
            return super().put(item, *args, **kwargs)
        except queue.Full:
            self.timed_out.set()
            raise


class QueueStopContractTests(unittest.TestCase):
    def setUp(self):
        self.api = methods()
        self.stop = threading.Event()
        self.q = WatchedQueue()
        self.produced = []
        self.removed = []
        self.gates = []
        self.workers = []
        self.thread_errors = []

        def values():
            for value in itertools.count():
                self.produced.append(value)
                yield value

        self.obj = types.SimpleNamespace(
            csm_gen=types.SimpleNamespace(result=lambda num: values()), csm_num=1,
            csm_queue=self.q, model_stop_event=self.stop, min_queue_size=2,
            log_data=False, fft=object(), writeH5=object(),
            sample_splitter=types.SimpleNamespace(remove_object=lambda *x: self.removed.append(x)),
        )
        self.addCleanup(self.cleanup_workers)

    def start(self, name, target):
        def checked_target():
            try:
                target()
            except BaseException as error:
                self.thread_errors.append((name, error))
                self.stop.set()

        thread = threading.Thread(name='soundmap-contract-' + name, target=checked_target, daemon=True)
        self.workers.append(thread)
        thread.start()
        return thread

    def start_processing(self):
        self.obj.compute_prediction_thread = self.start('predictor', lambda: self.api['_predictor'](self.obj))
        self.obj.csm_thread = self.start('producer', lambda: self.api['_csm_generator'](self.obj))

    def stop_processing(self):
        return self.start('stopper', lambda: self.api['stop_model'](self.obj))

    def cleanup_workers(self):
        self.stop.set()
        for gate in self.gates:
            gate.set()
        # Release only this fixture's old unbounded enqueue after a baseline failure.
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        for worker in self.workers:
            worker.join(2)
        self.assertFalse(any(t.is_alive() for t in self.workers), 'owned worker leaked')
        self.assertEqual(self.thread_errors, [], 'owned worker raised unexpectedly')

    def test_actual_waiting_predictor_and_full_queue_stop(self):
        self.q.put('queued')
        self.start_processing()
        self.assertTrue(self.q.entered_full.wait(1))
        stopper = self.stop_processing()
        stopper.join(.7)
        self.assertFalse(stopper.is_alive(), 'producer join blocked after actual predictor exited')
        self.assertTrue(self.q.empty())
        self.assertEqual(self.removed, [(self.obj.fft,)])

    def test_ordinary_backpressure_preserves_fifo_and_pending_item(self):
        self.q.put('queued')
        self.obj.csm_thread = self.start('producer', lambda: self.api['_csm_generator'](self.obj))
        self.obj.compute_prediction_thread = self.start('consumer-wait', self.stop.wait)
        self.assertTrue(self.q.timed_out.wait(.7), 'full queue never reached a cancellable retry')
        self.assertEqual(self.produced, [0])
        self.assertEqual(self.q.qsize(), 1)
        self.assertEqual(self.q.get(timeout=1), 'queued')
        delivered = [self.q.get(timeout=1) for _ in range(8)]
        self.assertEqual(delivered, list(range(8)))
        self.assertEqual(self.q.maxsize, 1)
        self.assertEqual(self.produced[:8], delivered)
        stopper = self.stop_processing()
        stopper.join(.7)
        self.assertFalse(stopper.is_alive())
        self.assertTrue(self.q.empty())

    def test_empty_queue_stop_cleans_logging_branch(self):
        self.stop.set()
        self.obj.log_data = True
        self.obj.save_time_samples_thread = self.start('logging-wait', self.stop.wait)
        self.start_processing()
        stopper = self.stop_processing()
        stopper.join(.7)
        self.assertFalse(stopper.is_alive())
        self.assertEqual(self.produced, [])
        self.assertTrue(self.q.empty())
        self.assertEqual(self.removed, [(self.obj.fft, self.obj.writeH5)])

    def test_stop_during_successful_enqueue_still_clears_queue(self):
        entered, release = threading.Event(), threading.Event()
        self.gates.append(release)
        original_put = self.q.put

        def delayed_put(item, *args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise AssertionError('owned enqueue gate was not released')
            return original_put(item, *args, **kwargs)

        self.q.put = delayed_put
        self.start_processing()
        self.assertTrue(entered.wait(1))
        stopper = self.stop_processing()
        self.assertTrue(self.stop.wait(1))
        release.set()
        stopper.join(.7)
        self.assertFalse(stopper.is_alive())
        self.assertTrue(self.q.empty())
        self.assertEqual(self.produced, [0])

    def test_blocked_iterator_remains_outside_queue_cancellation(self):
        entered, release = threading.Event(), threading.Event()
        self.gates.append(release)

        def controlled_iterator():
            entered.set()
            if not release.wait(2):
                raise AssertionError('owned iterator gate was not released')
            yield 'pending'

        self.obj.csm_gen.result = lambda num: controlled_iterator()
        self.start_processing()
        self.assertTrue(entered.wait(1))
        stopper = self.stop_processing()
        self.assertTrue(self.stop.wait(1))
        stopper.join(.15)
        self.assertTrue(stopper.is_alive(), 'probe must observe the separate next(gen) limitation')
        release.set()
        stopper.join(.7)
        self.assertFalse(stopper.is_alive())
        self.assertTrue(self.q.empty())


if __name__ == '__main__':
    unittest.main()
