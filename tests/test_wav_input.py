"""Verify PCM packing and the shared beamforming path against equivalent HDF5."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import xml.etree.ElementTree as ET

import acoular as ac
import h5py
import numpy as np

from evaluation.offline import source_maps
from scripts.offline_map import main


class WavInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.signal = self.root / 'recording.WAV'
        self.geometry = Path(ac.__file__).parent / 'xml/minidsp_uma-16_mirrored.xml'
        positions = np.array([[float(point.attrib[axis]) for axis in 'xyz']
                              for point in ET.parse(self.geometry).findall('.//pos')])
        self.target = [.5, .3, 1.5]
        distance = np.linalg.norm(positions - self.target, axis=1)
        t = np.arange(1024)[:, None] / 16384
        self.tone = .7 * np.sin(2 * np.pi * 4096 * (t - distance / 343)) / distance
        self.options = dict(frequencies=[4096, 4608], z=1.5, bounds=(.3, .7, .1, .5),
                            block_size=256, start=73, stop=911, invalid_channels=[1, 7])

    def write_wav(self, bits, counts):
        if bits == 8:
            raw = (counts + 128).astype(np.uint8).tobytes()
        elif bits == 24:
            # Pack PCM using Python's signed integer representation.
            raw = b''.join(int(value).to_bytes(3, 'little', signed=True) for value in counts.flat)
        else:
            raw = counts.astype(f'<i{bits // 8}').tobytes()
        with wave.open(str(self.signal), 'wb') as recording:
            recording.setparams((counts.shape[1], bits // 8, 16384, 0, 'NONE', ''))
            recording.writeframes(raw)

    def as_extensible(self, valid_bits):
        standard = self.signal.read_bytes()
        pcm_subtype = bytes.fromhex('0100000000001000800000aa00389b71')
        # Include an odd-length metadata chunk with a pad before the format.
        chunks = (b'JUNK\x01\0\0\0x\0fmt ' + (40).to_bytes(4, 'little') + b'\xfe\xff' + standard[22:36]
                  + (22).to_bytes(2, 'little') + valid_bits.to_bytes(2, 'little')
                  + bytes(4) + pcm_subtype + standard[36:])
        return b'RIFF' + (len(chunks) + 4).to_bytes(4, 'little') + b'WAVE' + chunks

    def test_ordered_calibrated_sweep_matches_equivalent_hdf5(self):
        calibration = self.root / 'calibration.xml'
        xml = ET.Element('Calib')
        for i in range(self.tone.shape[1]):
            ET.SubElement(xml, 'pos', Name=str(i), factor=str(.8 + i / 20))
        ET.ElementTree(xml).write(calibration)
        hdf5 = self.root / 'reference.h5'
        for bits, valid_bits in ((8, 8), (16, 16), (24, 24), (24, 20), (32, 32)):
            with self.subTest(bits=bits, valid_bits=valid_bits):
                scale = 2 ** (bits - 1)
                shift = bits - valid_bits
                counts = np.rint(self.tone * (scale >> shift)).astype(np.int64) << shift
                self.write_wav(bits, counts)
                if shift:
                    self.signal.write_bytes(self.as_extensible(valid_bits))
                with h5py.File(hdf5, 'w') as file:
                    data = file.create_dataset('time_data', data=counts / scale)
                    data.attrs['sample_freq'] = 16384
                expected, grid, _ = source_maps(hdf5, self.geometry, calibration=calibration, **self.options)
                actual, actual_grid, reports = source_maps(self.signal, self.geometry,
                                                            calibration=calibration, **self.options)
                np.testing.assert_array_equal(actual_grid, grid)
                np.testing.assert_array_equal(actual, expected)
                peak = np.unravel_index(np.argmax(actual[0]), actual[0].shape)
                np.testing.assert_allclose(actual_grid[:, *peak], self.target, atol=.05)
                for report in reports:
                    self.assertEqual(report['input_format'], 'wav-pcm')
                    self.assertEqual(report['pcm_bits'], bits)
                    self.assertEqual(report['sample_start'], 73)
                    self.assertEqual(report['sample_stop'], 911)
                    self.assertEqual(report['excluded_zero_based_channels'], [1, 7])
                    self.assertEqual(report['input_sha256'], hashlib.sha256(self.signal.read_bytes()).hexdigest())

    def test_invalid_inputs_leave_no_cli_output_and_allow_retry(self):
        output = self.root / 'result'
        args = ['--input', str(self.signal), '--geometry', str(self.geometry), '--frequency', '4096',
                '--z', '1.5', '--bounds', '.3', '.7', '.1', '.5', '--block-size', '256',
                '--output', str(output)]
        quantized = np.rint(self.tone * 32768).astype(np.int64)
        self.write_wav(16, quantized)
        valid = self.signal.read_bytes()
        malformed_length = valid[:4] + (128).to_bytes(4, 'little') + valid[8:]
        unsupported_float = valid[:20] + (3).to_bytes(2, 'little') + valid[22:]
        alignment_mismatch = valid[:32] + (33).to_bytes(2, 'little') + valid[34:]
        unsupported_bits = valid[:34] + (15).to_bytes(2, 'little') + valid[36:]
        partial_frame = (valid[:4] + (len(valid) - 6).to_bytes(4, 'little') + valid[8:40]
                         + (len(valid) - 43).to_bytes(4, 'little') + valid[44:] + bytes(2))
        for bad in (valid[:-1], malformed_length, unsupported_float, alignment_mismatch, unsupported_bits,
                    partial_frame, self.as_extensible(32), b'this is not a wave file'):
            self.signal.write_bytes(bad)
            with self.assertRaises(SystemExit) as exit_code:
                main(args)
            self.assertEqual(exit_code.exception.code, 2)
            self.assertFalse(output.exists())
        self.write_wav(16, quantized[:, :2])
        with self.assertRaisesRegex(ValueError, 'channel count'):
            source_maps(self.signal, self.geometry, **self.options)
        self.signal.write_bytes(valid)
        with patch('evaluation.offline._MAX_PCM_BYTES', 32):
            with self.assertRaisesRegex(ValueError, 'Decoded WAV selection'):
                source_maps(self.signal, self.geometry, **self.options)
        self.assertEqual(main(args), 0)
        self.assertTrue((output / 'map.npz').exists())
        self.assertTrue((output / 'map.html').exists())


if __name__ == '__main__':
    unittest.main()
