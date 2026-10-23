"""Actual processor methods with controlled queues/iterators; no device imports."""
import ast
import itertools
from pathlib import Path
import queue
import threading
import types
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'acoustic-camera/data_processing/process.py'


def processor_methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Processor')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {'_csm_generator', 'stop_model'}]
    namespace = {'queue': queue}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class ObservedQueue(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=1)
        self.blocked = threading.Event()
        self.full_timeout = threading.Event()

    def put(self, item, *args, **kwargs):
        if self.full():
            self.blocked.set()
        try:
            return super().put(item, *args, **kwargs)
        except queue.Full:
            self.full_timeout.set()
            raise


class ModelQueueTests(unittest.TestCase):
    def setUp(self):
        methods = processor_methods()
        self.q = ObservedQueue()
        self.stop = threading.Event()
        self.consumed = []
        self.removed = []

        def values():
            for value in itertools.count():
                self.consumed.append(value)
                yield value

        self.obj = types.SimpleNamespace(
            csm_gen=types.SimpleNamespace(result=lambda num: values()), csm_num=1,
            csm_queue=self.q, model_stop_event=self.stop, log_data=False, fft=object(),
            sample_splitter=types.SimpleNamespace(remove_object=lambda *x: self.removed.append(x)),
        )
        self.producer = threading.Thread(target=methods['_csm_generator'], args=(self.obj,), daemon=True)
        self.consumer = threading.Thread(target=self.stop.wait, daemon=True)
        self.obj.csm_thread = self.producer
        self.obj.compute_prediction_thread = self.consumer
        self.stop_method = methods['stop_model']
        self.stopper = None
        self.addCleanup(self.release_workers)

    def release_workers(self):
        self.stop.set()
        # Also releases the old implementation after a failed regression assertion.
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        for worker in (self.producer, self.consumer, self.stopper):
            if worker is not None and worker.ident is not None:
                worker.join(2)
                self.assertFalse(worker.is_alive(), 'controlled worker leaked')

    def fill_and_start(self):
        self.q.put('previous')
        self.consumer.start()
        self.producer.start()
        self.assertTrue(self.q.blocked.wait(2), 'producer never reached full queue')

    def test_actual_stop_returns_with_full_queue(self):
        self.fill_and_start()
        self.stopper = threading.Thread(target=self.stop_method, args=(self.obj,), daemon=True)
        self.stopper.start()
        self.stopper.join(1)
        self.assertFalse(self.stopper.is_alive(), 'stop_model blocked joining full-queue producer')
        self.assertFalse(self.producer.is_alive())
        self.assertEqual(self.q.qsize(), 0)
        self.assertEqual(self.removed, [(self.obj.fft,)])

    def test_backpressure_retains_pending_item_across_timeout(self):
        self.fill_and_start()
        self.assertTrue(self.q.full_timeout.wait(2), 'enqueue never became cancellable')
        self.assertEqual(self.q.maxsize, 1)
        self.assertEqual(self.q.qsize(), 1)
        self.assertEqual(self.consumed, [0], 'full queue must not advance or drop pending data')
        self.assertEqual(self.q.get(timeout=1), 'previous')
        self.assertEqual(self.q.get(timeout=1), 0)

    def test_preexisting_stop_does_not_consume_data(self):
        self.stop.set()
        self.producer.start()
        self.producer.join(1)
        self.assertFalse(self.producer.is_alive())
        self.assertEqual(self.consumed, [])
        self.assertTrue(self.q.empty())

    def test_non_full_queue_error_is_not_swallowed(self):
        class BrokenQueue:
            def put(self, *args, **kwargs):
                raise RuntimeError('queue adapter failed')
        self.obj.csm_queue = BrokenQueue()
        with self.assertRaisesRegex(RuntimeError, 'queue adapter failed'):
            processor_methods()['_csm_generator'](self.obj)


if __name__ == '__main__':
    unittest.main()
