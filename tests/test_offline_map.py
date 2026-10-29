"""Independent spherical-wave coordinates through the production Acoular path."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import acoular as ac
import h5py
import numpy as np
from evaluation.offline import source_map, save_map


class OfflineMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = Path(ac.__file__).parent/'xml/minidsp_uma-16_mirrored.xml'
        cls.microphones = np.array([[float(p.attrib[a]) for a in ('x','y','z')]
                                   for p in ET.parse(cls.geometry).findall('.//pos')])

    def make_signal(self, path, position):
        frequency, rate = 4096., 16384.
        times = np.arange(8192)/rate
        distance = np.linalg.norm(self.microphones-np.array(position),axis=1)
        # Independent delayed point-source equation, not Acoular's simulator.
        data = .01*np.sin(2*np.pi*frequency*(times[:,None]-distance[None,:]/343.))/distance[None,:]
        with h5py.File(path,'w') as file:
            ds=file.create_dataset('time_data',data=data)
            ds.attrs['sample_freq']=rate
        return data

    def test_changed_input_peak_and_fresh_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            peaks=[]
            for i, position in enumerate(([.5,.3,1.5],[-.4,.6,1.5])):
                signal=directory/f'input-{i}.h5'
                self.make_signal(signal,position)
                power, coordinates, report=source_map(signal,self.geometry,frequency=4096.,z=1.5,block_size=256)
                actual=coordinates[:,*np.unravel_index(np.argmax(power),power.shape)]
                np.testing.assert_allclose(actual,position,atol=.05)
                peaks.append(actual)
                saved=save_map(directory/f'map-{i}',power,coordinates,report,peaks=1)
                self.assertEqual(saved['peaks'][0]['position_m'],actual.tolist())
                for name in ('map.npz','map.html','measurements.json'):
                    self.assertGreater((directory/f'map-{i}'/name).stat().st_size,100)
                before=hashlib.sha256((directory/f'map-{i}'/'map.npz').read_bytes()).hexdigest()
                with self.assertRaises(FileExistsError):
                    save_map(directory/f'map-{i}',power,coordinates,report)
                self.assertEqual(before,hashlib.sha256((directory/f'map-{i}'/'map.npz').read_bytes()).hexdigest())
            self.assertGreater(np.linalg.norm(peaks[0]-peaks[1]),.5)

    def test_invalid_sample_rate_channels_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.h5'
            self.make_signal(path,[.5,.3,1.5])
            with h5py.File(path,'a') as file:
                file['time_data'].attrs['sample_freq']=float('nan')
            with self.assertRaisesRegex(ValueError,'sample_freq'):
                source_map(path,self.geometry,frequency=4096.,z=1.5)
            self.make_signal(path,[.5,.3,1.5])
            with h5py.File(path,'a') as file: file['time_data'][0,0]=float('nan')
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                source_map(path,self.geometry,frequency=4096.,z=1.5)
            with h5py.File(path,'w') as file:
                ds=file.create_dataset('time_data',data=np.ones((2048,2)))
                ds.attrs['sample_freq']=16384.
            with self.assertRaisesRegex(ValueError,'channel count'):
                source_map(path,self.geometry,frequency=4096.,z=1.5)


if __name__ == '__main__':
    unittest.main()
