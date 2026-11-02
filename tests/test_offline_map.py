"""Independent spherical-wave coordinates through the production Acoular path."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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

    def test_cache_policy_restored_on_result_and_no_signal_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.h5'
            self.make_signal(path,[.5,.3,1.5])
            original_policy = ac.config.global_caching
            try:
                ac.config.global_caching = 'readonly'
                power, _, report = source_map(path,self.geometry,frequency=4096.,z=1.5,
                                               block_size=256,bounds=(.3,.7,.1,.5))
                self.assertGreater(power.max(),0)
                self.assertEqual(ac.config.global_caching,'readonly')
                # Independent actual Acoular CSM allocation verifies the model;
                # this small check does not allocate an oversized spectral matrix.
                samples = ac.TimeSamples(data=np.zeros((512,16)),sample_freq=16384.)
                spectra = ac.PowerSpectra(source=samples,block_size=256,cached=False)
                csm = spectra.calc_csm()
                self.assertEqual(report['estimated_spectral_working_bytes'],4*csm.nbytes)
                ac.config.global_caching = 'none'
                grid = ac.RectGrid(x_min=.3,x_max=.7,y_min=.1,y_max=.5,z=1.5,increment=.05)
                steer = ac.SteeringVector(mics=ac.MicGeom(file=str(self.geometry)),grid=grid)
                beamformer = ac.BeamformerBase(freq_data=spectra,steer=steer,cached=False)
                _ = beamformer.result
                self.assertEqual(report['estimated_beamformer_result_bytes'],beamformer._ac.nbytes)
                self.assertEqual(report['estimated_working_bytes'],4*csm.nbytes+beamformer._ac.nbytes)
                ac.config.global_caching = 'readonly'
                with h5py.File(path,'a') as file:
                    file['time_data'][:] = 0
                with self.assertRaisesRegex(ValueError,'no positive finite'):
                    source_map(path,self.geometry,frequency=4096.,z=1.5,
                               block_size=256,bounds=(.3,.7,.1,.5))
                self.assertEqual(ac.config.global_caching,'readonly')
            finally:
                ac.config.global_caching = original_policy

    def test_spectral_memory_bound_before_large_engine_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            geometry=directory/'array.xml'
            root=ET.Element('MicArray')
            for i in range(128):
                ET.SubElement(root,'pos',Name=str(i),x=str(i*.001),y='0',z='0')
            ET.ElementTree(root).write(geometry)
            path=directory/'input.h5'
            with h5py.File(path,'w') as file:
                data=file.create_dataset('time_data',shape=(4096,128),dtype='f4',fillvalue=0)
                data.attrs['sample_freq']=16384.
            # Valid input dimensions and a tiny spatial grid would previously
            # admit over2GiB of CSM work buffers. Rejection needs no such allocation.
            with self.assertRaisesRegex(ValueError,'spectral working buffers'):
                source_map(path,geometry,frequency=4096.,z=1.5,block_size=4096,
                           bounds=(-.1,.1,-.1,.1),increment=.1)

    def test_frequency_grid_result_memory_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.h5'
            with h5py.File(path,'w') as file:
                data=file.create_dataset('time_data',shape=(4096,16),dtype='f4',fillvalue=0)
                data.attrs['sample_freq']=16384.
            # Fewer than250000points and4Mmic/gridpairs, small CSM buffers,
            # but an approximately3.9GB all-frequency beamforming result.
            with self.assertRaisesRegex(ValueError,'beamformer result'):
                source_map(path,self.geometry,frequency=4096.,z=1.5,block_size=4096,
                           bounds=(-1,1,-1,1),increment=.0041)

    def test_graph_failure_restores_policy_and_closes_recording(self):
        # Failure injection checks cleanup only; the numerical tests above run
        # the actual spectral/beamforming engines without replacing them.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.h5'
            self.make_signal(path,[.5,.3,1.5])
            original_policy=ac.config.global_caching
            try:
                ac.config.global_caching='readonly'
                with patch('evaluation.offline.ac.PowerSpectra',side_effect=RuntimeError('injected graph construction failure')):
                    with self.assertRaisesRegex(RuntimeError,'injected graph'):
                        source_map(path,self.geometry,frequency=4096.,z=1.5,block_size=256)
                self.assertEqual(ac.config.global_caching,'readonly')
                import tables
                self.assertFalse(tables.file._open_files.get_handlers_by_name(str(path.resolve())))
                with h5py.File(path,'a') as file:
                    file.attrs['cleanup_verified']=True
            finally:
                ac.config.global_caching=original_policy


if __name__ == '__main__':
    unittest.main()
