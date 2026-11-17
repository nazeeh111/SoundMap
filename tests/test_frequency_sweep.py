"""Shared graph sweeps against independently executed scalar-band baselines."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import acoular as ac
import numpy as np
from evaluation.offline import source_map, source_maps, save_maps
import test_offline_map


class FrequencySweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_offline_map.OfflineMapTests.setUpClass()

    def fixture(self, directory):
        signal = Path(directory) / 'input.h5'
        fixture = test_offline_map.OfflineMapTests()
        fixture.make_signal(signal, [.5, .3, 1.5])
        return signal, fixture.geometry

    def test_shared_graph_matches_scalar_maps_and_fft_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            signal, geometry = self.fixture(directory)
            options = dict(z=1.5, block_size=256, bounds=(.3, .7, .1, .5))
            # One input snapshot each, and one spectral graph for the sweep.
            import shutil
            calc_csm = ac.PowerSpectra.calc_csm
            csm_calls = []
            def counted(instance):
                csm_calls.append(instance)
                return calc_csm(instance)
            with patch.object(ac.PowerSpectra, 'calc_csm', counted), patch('evaluation.offline.ac.PowerSpectra', wraps=ac.PowerSpectra) as spectral, patch('evaluation.offline.shutil.copyfileobj', wraps=shutil.copyfileobj) as copy:
                powers, coordinates, reports = source_maps(signal, geometry, frequencies=[4096., 4500.], **options)
            self.assertEqual(spectral.call_count, 1)
            self.assertEqual(len(csm_calls), 1)
            self.assertEqual(copy.call_count, 2)
            for f, power, report in zip([4096., 4500.], powers, reports):
                baseline, xyz, single = source_map(signal, geometry, frequency=f, **options)
                np.testing.assert_array_equal(power, baseline)
                np.testing.assert_array_equal(coordinates, xyz)
                self.assertEqual(report['input_sha256'], single['input_sha256'])
                expected = np.arange(129)*64.
                expected = expected[(expected >= f*2**(-1/6)) & (expected < f*2**(1/6))]
                self.assertEqual(report['frequency_band']['fft_frequencies_hz'], expected.tolist())
                self.assertEqual(report['frequency_band']['fft_spacing_hz'], 64.)
            self.assertEqual(reports[0]['compute_seconds'], reports[1]['compute_seconds'])
            result = save_maps(Path(directory)/'sweep', powers, coordinates, reports)
            self.assertEqual(len(result['bands']), 2)
            for i, power in enumerate(powers, 1):
                with np.load(Path(directory)/f'sweep/band-{i:02d}/map.npz') as saved:
                    np.testing.assert_array_equal(saved['power'], power)
            with self.assertRaises(FileExistsError):
                save_maps(Path(directory)/'sweep', powers, coordinates, reports)

    def test_late_render_failure_publishes_nothing_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            signal, geometry = self.fixture(directory)
            powers, coordinates, reports = source_maps(signal, geometry, frequencies=[4096., 4500.], z=1.5, block_size=256, bounds=(.3,.7,.1,.5))
            from bokeh.embed import file_html
            calls = 0
            def render(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('second band render failed')
                return file_html(*args, **kwargs)
            output = Path(directory)/'sweep'
            with patch('bokeh.embed.file_html', side_effect=render):
                with self.assertRaisesRegex(RuntimeError, 'second band'):
                    save_maps(output, powers, coordinates, reports)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).glob('.soundmap*')), [])
            save_maps(output, powers, coordinates, reports)
            self.assertTrue((output/'sweep.json').exists())

    def test_bad_bands_fail_before_spectral_graph_and_restore_cache_after_late_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            signal, geometry = self.fixture(directory)
            options = dict(z=1.5, block_size=128, bounds=(.3,.7,.1,.5))
            with patch('evaluation.offline.ac.PowerSpectra') as graph:
                for frequencies in ([], [4096.]*2, [float('nan')], list(range(33)), [1.], [4096., 8000.]):
                    with self.assertRaises(ValueError):
                        source_maps(signal, geometry, frequencies=frequencies, **options)
                graph.assert_not_called()
            old = ac.config.global_caching
            original = ac.BeamformerBase.synthetic
            try:
                ac.config.global_caching = 'readonly'
                calls = 0
                def fail_late(instance, *args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError('second band compute failed')
                    return original(instance, *args, **kwargs)
                with patch.object(ac.BeamformerBase, 'synthetic', fail_late):
                    with self.assertRaisesRegex(RuntimeError, 'second band compute'):
                        source_maps(signal, geometry, frequencies=[4096.,4500.], **options)
                self.assertEqual(ac.config.global_caching, 'readonly')
                powers, _, _ = source_maps(signal, geometry, frequencies=[4096.,4500.], **options)
                self.assertEqual(len(powers), 2)
            finally:
                ac.config.global_caching = old
