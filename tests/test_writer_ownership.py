"""Allocated writers outlive option changes and retain acquisition ownership."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as S
import unittest

import numpy as np
from bokeh.models import Div, RadioButtonGroup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'acoustic-camera'))
from data_processing.process import Processor
from ui.dashboard import Dashboard
from ui.measurement import MeasurementOwner

def until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('controlled worker timed out')
        time.sleep(.002)

class Source:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.name = ''
    def result(self, num):
        self.entered.set()
        if not self.release.wait(10):
            raise AssertionError('fixture source not released')
        yield np.array([[1.]])

class WriterOwnership(unittest.TestCase):
    def exercise(self, mode, logging, queued):
        p = Processor.__new__(Processor)
        p.model_on, p.ckpt_path, p.device, p.log_data = True, 'fixture', 1, logging
        p.fft, p.lastOut = object(), object()
        p.csm_block_size, p.csm_num, p.csm_buffer_size, p.min_queue_size = 128, 1, 1, 2
        p._generators = p._setup_model = lambda: None
        p._get_result_filenames = lambda _: ('unused', 'unused')
        p.beamforming_grid, p.grid_dim = S(z=1), (1, 1)
        p.beamforming_result_lock = threading.Lock()
        p._beamforming_result_revision = 0
        p.beamforming_results = {'results': 'old', 'max_x': [0], 'max_y': [0], 'max_s': [0]}
        primary, writer = Source(), Source()
        p.writeH5, p.bf_out, p.csm_gen = writer, primary, primary
        registered = []
        def remove(*objects):
            for obj in objects:
                registered.remove(obj)
        p.sample_splitter = S(register_object=lambda *xs: registered.extend(xs), remove_object=remove)
        gate = threading.Lock()
        owner = MeasurementOwner(p, gate=gate)
        other = MeasurementOwner(S(), gate=gate)
        d = Dashboard.__new__(Dashboard)
        d.processor, d.measurement = p, owner
        d.data_folder_display = Div(visible=logging)
        d.save_time_data = RadioButtonGroup(labels=['Yes', 'No'], active=0 if logging else 1)
        d.save_time_data.on_change('active', d.toggle_save)
        try:
            self.assertTrue(owner.start(mode))
            self.assertTrue(primary.entered.wait(1))
            until(lambda: owner.snapshot()[0] == 'running')
            if logging:
                self.assertTrue(writer.entered.wait(1))
            if queued:
                d.save_time_data.disabled = True
                d.save_time_data.active = 1 if logging else 0
                self.assertEqual(p.log_data, logging)
                self.assertEqual(d.save_time_data.active, 0 if logging else 1)
                self.assertEqual(d.data_folder_display.visible, logging)
            else:
                # Processor cleanup remains safe even without the UI guard.
                p.log_data = not logging
            self.assertTrue(owner.stop())
            if queued:
                d.save_time_data.active = 1 if logging else 0
                self.assertEqual(d.save_time_data.active, 0 if logging else 1)
            primary.release.set()
            worker = p.csm_thread if mode == 'model' else p.beamforming_thread
            worker.join(1)
            self.assertFalse(worker.is_alive())
            writer_thread = p.save_time_samples_thread if mode == 'model' else p.save_time_samples_beamforming_thread
            if logging:
                self.assertTrue(writer_thread.is_alive())
                self.assertEqual(owner.snapshot()[0], 'stopping')
                self.assertIn(writer, registered)
                self.assertFalse(other.start('model'))
                writer.release.set()
            else:
                self.assertIsNone(writer_thread)
                self.assertFalse(writer.entered.is_set())
            owner._thread.join(2)
            self.assertFalse(owner._thread.is_alive())
            self.assertEqual(owner.snapshot(), ('idle', None))
            self.assertEqual(registered, [])
            self.assertTrue(gate.acquire(blocking=False))
            gate.release()
            if queued:
                d.save_time_data.active = 1 if logging else 0
                self.assertEqual(p.log_data, not logging)
        finally:
            primary.release.set()
            writer.release.set()
            owner.stop()
            if owner._thread:
                owner._thread.join(3)
                self.assertFalse(owner._thread.is_alive())

    def test_direct_option_change_does_not_change_allocated_cleanup(self):
        for mode in ('model', 'beamforming'):
            for logging in (True, False):
                with self.subTest(mode=mode, logging=logging):
                    self.exercise(mode, logging, queued=False)

    def test_actual_queued_save_callbacks_restore_choice_until_idle(self):
        for mode in ('model', 'beamforming'):
            for logging in (True, False):
                with self.subTest(mode=mode, logging=logging):
                    self.exercise(mode, logging, queued=True)

if __name__ == '__main__':
    unittest.main()
