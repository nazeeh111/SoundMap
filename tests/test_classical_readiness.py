"""Real Bokeh/Acoular document construction, with acquisition explicitly disabled."""
import importlib.util
from pathlib import Path
import os
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ClassicalReadiness(unittest.TestCase):
    def test_classical_document_without_checkpoint_or_device(self):
        # Deliberately no TensorFlow, cv2 or model stubs: import the real modules.
        sys.path.insert(0, str(ROOT/'acoustic-camera'))
        old_cwd, old_argv = Path.cwd(), sys.argv
        try:
            os.chdir(ROOT/'acoustic-camera')
            sys.argv = ['bokeh_app.py', '--no-device']
            spec = importlib.util.spec_from_file_location('readiness_app', 'scripts/bokeh_app.py')
            app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app)
            from bokeh.document import Document
            doc = Document()
            dashboard = app.build_document(doc, no_device=True)
            self.assertEqual(doc.title, 'SoundMap')
            self.assertTrue(doc.roots)
            self.assertIsNone(dashboard.processor.dev)
            self.assertFalse(dashboard.model_on)
            doc.validate()
            self.assertNotIn('tensorflow', sys.modules)
            self.assertNotIn('cv2', sys.modules)
            # Construct the classical controls with a real inert Acoular source,
            # without selecting/reading a microphone or starting processing.
            import acoular as ac
            import numpy as np
            dashboard.processor.dev = ac.TimeSamples(data=np.zeros((1,16)), sample_freq=51200)
            classical = app.Dashboard(dashboard.config, dashboard.processor, False, (1.0,1.0))
            self.assertFalse(classical.model_on)
            self.assertTrue(classical.get_layout().children)
            recorded = doc.to_json()['roots'][0]['attributes']['children'][0]['attributes']['text']
            self.assertIn('Device-free session', recorded)
            self.assertIn('In your checkout, extract', recorded)
            self.assertIn('band-4000.npz', recorded)
            self.assertIn('band-5000.json', recorded)
            self.assertFalse(hasattr(dashboard, 'acoustic_camera_plot'))
            from ui import recordings
            with TemporaryDirectory() as empty, patch.object(recordings, 'ASSETS', Path(empty)):
                unavailable = recordings.recorded_view('Acquisition disabled')
                self.assertIn('The archived images are unavailable', unavailable)
                self.assertIn('band-4000.npz is unavailable', unavailable)
                self.assertIn('full report archive is missing', unavailable)
                self.assertNotIn('In your checkout, extract', unavailable)
                self.assertNotIn('data:image/png', unavailable)
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
            sys.path.remove(str(ROOT/'acoustic-camera'))

    def test_explicit_unavailable_model_fails(self):
        sys.path.insert(0, str(ROOT/'acoustic-camera'))
        try:
            from config.runtime import resolve_checkpoint
            self.assertIsNone(resolve_checkpoint(None))
            with self.assertRaisesRegex(ValueError, 'No numbered .keras checkpoint'):
                resolve_checkpoint(ROOT/'missing-model')
        finally:
            sys.path.remove(str(ROOT/'acoustic-camera'))


if __name__ == '__main__':
    unittest.main()
