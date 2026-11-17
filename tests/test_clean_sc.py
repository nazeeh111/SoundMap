"""Independent spherical random-source fixtures through actual Acoular methods."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import itertools
import warnings
import xml.etree.ElementTree as ET

import acoular as ac
import h5py
import numpy as np
from evaluation import offline


class CleanScTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry=Path(ac.__file__).parent/'xml/minidsp_uma-16_mirrored.xml'
        cls.mics=np.array([[float(row.attrib[a]) for a in ('x','y','z')] for row in ET.parse(cls.geometry).findall('.//pos')])

    def signal(self,path,positions,amplitudes,seed=273):
        rate=16384.;n=16384;frequencies=np.fft.rfftfreq(n,1/rate)
        data=np.zeros((n,len(self.mics)));rng=np.random.default_rng(seed)
        for position,amplitude in zip(positions,amplitudes):
            spectrum=np.fft.rfft(rng.normal(size=n));spectrum[(frequencies<3500)|(frequencies>4700)]=0
            distance=np.linalg.norm(self.mics-np.array(position),axis=1)
            for channel,d in enumerate(distance):
                data[:,channel]+=amplitude*np.fft.irfft(spectrum*np.exp(-2j*np.pi*frequencies*d/343.),n=n)/d
        data+=.01*rng.normal(size=data.shape)
        with h5py.File(path,'w') as file:
            samples=file.create_dataset('time_data',data=data);samples.attrs['sample_freq']=rate

    def options(self):return dict(frequency=4096.,z=1.5,block_size=256,bounds=(-.8,.8,-.7,.7),increment=.05)

    def test_two_source_changed_input_localization_uses_actual_clean_sc(self):
        cases=[([[-.4,.3,1.5],[.5,-.3,1.5]],[1.,.7]),([[-.5,-.2,1.5],[.4,.4,1.5]],[1.,.7]),([[-.4,.3,1.5],[.5,-.3,1.5]],[1.,.3])]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for index,(positions,amplitudes) in enumerate(cases):
                path=root/f'input-{index}.h5';self.signal(path,positions,amplitudes)
                with patch('evaluation.offline.ac.BeamformerCleansc',wraps=ac.BeamformerCleansc) as algorithm:
                    power,xyz,report=offline.source_map(path,self.geometry,method='clean-sc',**self.options())
                self.assertEqual(algorithm.call_count,1)
                self.assertEqual(algorithm.call_args.kwargs['n_iter'],0)
                saved=offline.save_map(root/f'map-{index}',power,xyz,report,peaks=2,separation=.2)
                self.assertIn('Acoular BeamformerCleansc',(root/f'map-{index}'/'map.html').read_text())
                peaks=[row['position_m'] for row in saved['peaks']]
                error=min(max(np.linalg.norm(np.array(p)-np.array(q)) for p,q in zip(order,positions)) for order in itertools.permutations(peaks))
                self.assertLessEqual(error,.05)
                self.assertEqual(report['algorithm'],'clean-sc')
                self.assertEqual(report['algorithm_parameters']['max_iterations_per_frequency'],32)
                self.assertGreater(report['estimated_clean_working_bytes'],0)

    def test_method_default_and_explicit_conventional_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.h5';self.signal(path,[[-.4,.3,1.5],[.5,-.3,1.5]],[1.,.7])
            default,xyz,report=offline.source_map(path,self.geometry,**self.options())
            explicit,other,other_report=offline.source_map(path,self.geometry,method='conventional',**self.options())
            np.testing.assert_array_equal(default,explicit);np.testing.assert_array_equal(xyz,other)
            self.assertEqual(report['algorithm'],'conventional');self.assertEqual(report['estimated_clean_work_items'],0)
            self.assertEqual(report['input_sha256'],other_report['input_sha256'])

    def test_clean_sweep_shares_csm_and_accounts_unique_frequency_work(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.h5';self.signal(path,[[-.4,.3,1.5],[.5,-.3,1.5]],[1.,.7])
            original=ac.PowerSpectra.calc_csm;calls=[]
            def count(instance):calls.append(instance);return original(instance)
            opts=self.options();opts.pop('frequency')
            with patch.object(ac.PowerSpectra,'calc_csm',count):
                powers,xyz,reports=offline.source_maps(path,self.geometry,method='clean-sc',frequencies=[4096.,4200.],**opts)
            self.assertEqual(len(calls),1)
            indices={i for report in reports for i in report['frequency_band']['fft_bin_indices']}
            self.assertEqual(reports[0]['unique_requested_fft_bins'],len(indices))
            scalar_work=[]
            for f,power in zip([4096.,4200.],powers):
                scalar,_,single=offline.source_map(path,self.geometry,method='clean-sc',frequency=f,**opts)
                np.testing.assert_array_equal(power,scalar)
                scalar_work.append(single['estimated_clean_work_items'])
            self.assertLess(reports[0]['estimated_clean_work_items'],sum(scalar_work))
            self.assertGreaterEqual(reports[0]['estimated_clean_work_items'],max(scalar_work))

    def test_invalid_method_and_clean_work_limits_refuse_before_spectral_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.h5';self.signal(path,[[-.4,.3,1.5],[.5,-.3,1.5]],[1.,.7])
            _,_,baseline=offline.source_map(path,self.geometry,**self.options())
            with patch('evaluation.offline.ac.PowerSpectra') as graph:
                with self.assertRaisesRegex(ValueError,'method'):offline.source_map(path,self.geometry,method='unknown',**self.options())
                with patch.object(offline,'_MAX_CLEAN_WORK_ITEMS',1):
                    with self.assertRaisesRegex(ValueError,'repeated beamforming/coherence'):offline.source_map(path,self.geometry,method='clean-sc',**self.options())
                bound=baseline['estimated_working_bytes']+baseline['retained_band_map_bytes']
                with patch.object(offline,'_MAX_WORKING_BYTES',bound):
                    with self.assertRaisesRegex(ValueError,'CLEAN-SC scratch'):offline.source_map(path,self.geometry,method='clean-sc',**self.options())
                graph.assert_not_called()

    def test_clean_failure_closes_samples_restores_policy_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.h5';self.signal(path,[[-.4,.3,1.5],[.5,-.3,1.5]],[1.,.7])
            policy=ac.config.global_caching;instances=[];original=ac.MaskedTimeSamples
            def samples(*args,**kwargs):
                result=original(*args,**kwargs);instances.append(result);return result
            try:
                ac.config.global_caching='readonly'
                with patch('evaluation.offline.ac.MaskedTimeSamples',side_effect=samples),patch.object(ac.BeamformerCleansc,'synthetic',side_effect=RuntimeError('injected CLEAN failure')):
                    with self.assertRaisesRegex(RuntimeError,'CLEAN failure'):offline.source_map(path,self.geometry,method='clean-sc',**self.options())
                self.assertEqual(ac.config.global_caching,'readonly')
                self.assertFalse(instances[0].h5f.isopen)
                power,_,_=offline.source_map(path,self.geometry,method='clean-sc',**self.options())
                self.assertGreater(power.max(),0)
            finally:ac.config.global_caching=policy

    def test_zero_signal_refused_before_deconvolution_without_runtime_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'zero.h5'
            with h5py.File(path,'w') as file:
                data=file.create_dataset('time_data',data=np.zeros((8192,16)));data.attrs['sample_freq']=16384.
            with patch('evaluation.offline.ac.BeamformerCleansc',wraps=ac.BeamformerCleansc) as algorithm,warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always',RuntimeWarning)
                with self.assertRaisesRegex(ValueError,'no positive finite'):
                    offline.source_map(path,self.geometry,method='clean-sc',**self.options())
                algorithm.assert_not_called()
                self.assertFalse(any(issubclass(w.category,RuntimeWarning) for w in observed))
