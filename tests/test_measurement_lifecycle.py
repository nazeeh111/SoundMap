"""Synthetic lifecycle integration; no capture, model, network or device access."""
import sys
import time
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'acoustic-camera'))
from ui.measurement import MeasurementOwner
from ui.dashboard import Dashboard
from bokeh.models import Button, Div
from data_processing.process import Processor
from queue import Queue
import numpy as np


class ProcessorFixture:
    def __init__(self, startup=False, failing=False):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.starting = threading.Event()
        self.allow_start = threading.Event()
        if not startup:
            self.allow_start.set()
        self.failing = failing
        self.started = 0
        self.cleaned = 0
        self.writes = 0
        self.source_thread = None

    def start_model(self):
        self.started += 1
        self.starting.set()
        self.allow_start.wait(3)
        self.model_stop_event = threading.Event()
        def produce():
            self.entered.set()
            self.release.wait(3)
            if not self.model_stop_event.is_set():
                self.writes += 1
        self.source_thread = threading.Thread(target=produce)
        self.source_thread.start()

    def stop_model(self):
        self.model_stop_event.set()
        self.source_thread.join()
        if self.failing:
            raise RuntimeError('<cleanup failed>')
        self.cleaned += 1

    start_beamforming = start_model
    def stop_beamforming(self):
        self.beamforming_stop_event.set()
        self.source_thread.join()
        self.cleaned += 1


def wait_for(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('timed out')


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.p = ProcessorFixture()
        self.owner = MeasurementOwner(self.p, gate=threading.Lock())
    def tearDown(self):
        self.p.allow_start.set(); self.p.release.set()
        self.owner.stop()
        if self.owner._thread:
            self.owner._thread.join(3)
            self.assertFalse(self.owner._thread.is_alive())
        if self.p.source_thread:
            self.p.source_thread.join(3)
            self.assertFalse(self.p.source_thread.is_alive())
    def test_stalled_stop_keeps_owner_and_restart_works(self):
        self.assertTrue(self.owner.start('model'))
        self.assertTrue(self.p.entered.wait(1))
        wait_for(lambda: self.owner.snapshot()[0] == 'running')
        before = time.monotonic(); self.assertTrue(self.owner.stop())
        self.assertLess(time.monotonic() - before, .2)
        self.assertEqual(self.owner.snapshot()[0], 'stopping')
        self.assertFalse(self.owner.start('model'))
        self.assertTrue(self.p.source_thread.is_alive())
        self.p.release.set()
        wait_for(lambda: self.owner.snapshot()[0] == 'idle')
        self.owner._thread.join(1)
        self.assertEqual((self.p.writes, self.p.cleaned), (0, 1))
        self.p.entered.clear(); self.p.release.clear()
        self.assertTrue(self.owner.start('model'))
        self.assertTrue(self.p.entered.wait(1))
        self.p.release.set(); self.owner.stop(); self.owner._thread.join(2)
        self.assertEqual(self.p.started, 2)
        self.assertEqual(self.p.cleaned, 2)
    def test_stop_during_start_waits_for_setup_without_resetting_old_event(self):
        self.p.allow_start.clear()
        self.assertTrue(self.owner.start('model'))
        self.assertTrue(self.p.starting.wait(1))
        self.assertTrue(self.owner.stop())
        self.assertEqual(self.owner.snapshot()[0], 'stopping')
        self.assertFalse(self.owner.start('model'))
        self.p.allow_start.set()
        self.assertTrue(self.p.entered.wait(1))
        wait_for(lambda: self.p.model_stop_event.is_set())
        self.p.release.set(); self.owner._thread.join(2)
        self.assertEqual(self.owner.snapshot()[0], 'idle')
        self.assertEqual(self.p.writes, 0)
    def test_failed_cleanup_is_not_reported_stopped_or_restartable(self):
        self.p.failing = True
        self.owner.start('model'); self.assertTrue(self.p.entered.wait(1))
        self.owner.stop(); self.p.release.set(); self.owner._thread.join(2)
        self.assertEqual(self.owner.snapshot()[0], 'failed')
        self.assertIn('cleanup failed', self.owner.snapshot()[1])
        self.assertFalse(self.owner.start('model'))
    def test_real_dashboard_callbacks_return_and_report_stopping(self):
        d = Dashboard.__new__(Dashboard)
        d.document = __import__('bokeh.plotting',fromlist=['curdoc']).curdoc()
        d.measurement = self.owner
        d.method = 0
        d.measurement_button = Button(label='Stop')
        d.measurement_status = Div(text='')
        d.estimation_callback = d.overflow_callback = d.beamforming_callback = None
        self.owner.start('model'); self.assertTrue(self.p.entered.wait(1))
        wait_for(lambda: self.owner.snapshot()[0] == 'running')
        before = time.monotonic(); d.start_measurement()
        self.assertLess(time.monotonic() - before, .2)
        self.assertTrue(d.measurement_button.disabled)
        self.assertIn('Stopping', d.measurement_status.text)
        self.assertFalse(self.owner.start('model'))
        self.p.release.set(); self.owner._thread.join(2)
        self.assertEqual(self.owner.snapshot()[0], 'idle')

class ActualProcessor(unittest.TestCase):
    def exercise(self, mode):
        entered, release = threading.Event(), threading.Event()
        class Source:
            def result(self, num):
                entered.set()
                release.wait(3)
                yield np.array([[1.0]])
        removed = []
        p = Processor.__new__(Processor)
        p.model_on, p.ckpt_path, p.device = True, 'synthetic', 1
        p.log_data = False
        p.csm_num, p.csm_buffer_size, p.min_queue_size = 1, 1, 2
        p.results_folder = 'unused'
        p.writeH5 = SimpleNamespace(name='')
        p.fft = object(); p.lastOut = object()
        p.sample_splitter = SimpleNamespace(register_object=lambda *args: None,
                                             remove_object=lambda *args: removed.append(args))
        p._generators = lambda: None
        p._setup_model = lambda: None
        p._get_result_filenames = lambda _: ('unused', 'unused')
        p.csm_gen = p.bf_out = Source()
        p.beamforming_grid = SimpleNamespace(z=1)
        p.beamforming_result_lock = threading.Lock()
        p.grid_dim = (1, 1)
        p.beamforming_results = {'results': 'old', 'max_x': [0], 'max_y': [0], 'max_s': [0]}
        owner = MeasurementOwner(p, gate=threading.Lock())
        try:
            self.assertTrue(owner.start(mode))
            self.assertTrue(entered.wait(1))
            wait_for(lambda: owner.snapshot()[0] == 'running')
            start = time.monotonic(); self.assertTrue(owner.stop())
            self.assertLess(time.monotonic() - start, .2)
            self.assertEqual(owner.snapshot()[0], 'stopping')
            self.assertFalse(owner.start(mode))
            release.set()
            owner._thread.join(3)
            self.assertFalse(owner._thread.is_alive())
            self.assertEqual(owner.snapshot()[0], 'idle')
            self.assertEqual(len(removed), 1)
            if mode == 'model':
                self.assertTrue(p.csm_queue.empty())
                self.assertFalse(p.csm_thread.is_alive())
                self.assertFalse(p.compute_prediction_thread.is_alive())
            else:
                self.assertEqual(p.beamforming_results['results'], 'old')
                self.assertFalse(p.beamforming_thread.is_alive())
        finally:
            release.set()
            owner.stop()
            if owner._thread:
                owner._thread.join(3)
    def test_actual_producers_publish_new_revisions(self):
        p = Processor.__new__(Processor)
        p.result_lock = threading.Lock(); p.beamforming_result_lock = threading.Lock()
        p._model_result_revision = p._beamforming_result_revision = 0
        p.model_stop_event = threading.Event(); p.beamforming_stop_event = threading.Event()
        p.csm_queue = Queue(); p.csm_queue.put(np.array([1.])); p.min_queue_size = 1
        p._preprocess_csm = lambda _: (np.array([1.]), 1.0)
        p.model = SimpleNamespace(predict=lambda *args, **kwargs: (np.array([1., 2.]),np.array([[1., 1.],[2., 2.],[3., 3.]]),None))
        p._save_results = p.model_stop_event.set
        p._predictor()
        self.assertEqual(p.get_model_snapshot()[0], 1)
        p.bf_out = SimpleNamespace(result=lambda num: iter([np.array([[1.]])]))
        p.grid_dim = (1,1); p.x_min=p.y_min=0; p.increment=1
        p.beamforming_results = {'results':None, 'max_x':[0], 'max_y':[0], 'max_s':0}
        p._beamforming_generator()
        self.assertEqual(p.get_beamforming_snapshot()[0], 1)
    def test_real_model_methods_blocked_in_next(self):
        self.exercise('model')
    def test_real_beamforming_methods_blocked_in_next(self):
        self.exercise('beamforming')

if __name__ == '__main__':
    unittest.main()
