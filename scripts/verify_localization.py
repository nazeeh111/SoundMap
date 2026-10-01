"""Check evaluation coordinates with independent ideal spherical-wave signals.

Requires the existing kernel dependencies. No devices or Acoular source simulator
are used. The expected evaluation frame is (-XML x, +XML y, +XML z), in metres.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir', required=True, type=Path, help='Scratch destination for signals and caches')
    parser.add_argument('--output', required=True, type=Path, help='JSON receipt destination')
    args = parser.parse_args()
    source = args.source_dir.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    for name, value in {
        'NUMBA_CACHE_DIR': str(work / 'numba-cache'),
        'MPLCONFIGDIR': str(work / 'matplotlib-cache'),
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
    }.items():
        os.environ[name] = value
    import acoular as ac
    import h5py
    import numpy as np
    from tables.file import _open_files

    ac.config.global_caching = 'none'
    ac.config.cache_dir = str(work / 'acoular-cache')
    module_file = source / 'evaluation/beamforming_funcs.py'
    module_hash = sha256(module_file)
    spec = importlib.util.spec_from_file_location('evaluation_beamforming', module_file)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    geometry = Path(ac.__file__).parent / 'xml/minidsp_uma-16_mirrored.xml'
    # Read the actual geometry independently of Acoular's signal simulation.
    microphones = np.array([[float(p.attrib[axis]) for axis in ('x', 'y', 'z')]
                            for p in ET.parse(geometry).findall('.//pos')]).T
    assert microphones.shape == (3, 16)
    assert np.array_equal(microphones, ac.MicGeom(from_file=geometry).mpos_tot)
    frequency, sample_rate, samples, speed = 4096.0, 16384.0, 8192, 343.0
    cases = [('center', [0.0, 0.0, 1.5]), ('positive-x-positive-y', [0.5, 0.3, 1.5]),
             ('negative-x-positive-y', [-0.4, 0.6, 1.8]), ('positive-x-negative-y', [0.7, -0.5, 2.0])]
    fixtures = []
    for label, coordinates in cases:
        position = np.array(coordinates)
        distances = np.linalg.norm(microphones - position[:, None], axis=0)
        times = np.arange(samples) / sample_rate
        # Independent spherical travel delay and 1/r amplitude, no PointSource.
        data = 0.01 * np.sin(2 * np.pi * frequency * (times[:, None] - distances[None, :] / speed)) / distances[None, :]
        path = work / (label + '.h5')
        with h5py.File(path, 'w') as file:
            dataset = file.create_dataset('time_data', data=data)
            dataset.attrs['sample_freq'] = sample_rate
        fixtures.append({'case': label, 'file': str(path), 'file_sha256': sha256(path),
                         'time_data_sha256': hashlib.sha256(data.tobytes()).hexdigest(),
                         'physical_xml_frame_m': coordinates})

    definitions = [('single_source_beamforming_2D', 2, 0.05), ('single_source_cleansc_2D', 2, 0.1),
                   ('single_source_beamforming', 3, 0.05), ('single_source_cleansc', 3, 0.1)]
    observations = []
    for name, dimensions, increment in definitions:
        for fixture in (fixtures if dimensions == 2 else fixtures[1:3]):
            physical = np.array(fixture['physical_xml_frame_m'])
            expected = physical[:dimensions] * np.array([-1, 1] if dimensions == 2 else [-1, 1, 1])
            function = getattr(kernels, name)
            try:
                arguments = [fixture['file'], speed, frequency]
                if dimensions == 2:
                    arguments.append(float(physical[2]))
                actual = np.asarray(function(*arguments, block_size=256))
            finally:
                # Close only the disposable fixture handles this call opened.
                for handle in tuple(_open_files.handlers):
                    if Path(handle.filename).resolve() == Path(fixture['file']):
                        handle.close()
            residual = actual - expected
            passed = bool(actual.shape == expected.shape and np.isfinite(actual).all()
                          and np.max(np.abs(residual)) <= increment + 1e-9)
            row = {'kernel': name, 'case': fixture['case'], 'dimensions': dimensions,
                   'expected_evaluation_frame_m': expected.tolist(), 'returned_coordinates_m': actual.tolist(),
                   'residual_m': residual.tolist(), 'declared_tolerance_per_axis_m': increment, 'passed': passed}
            observations.append(row)
            print(json.dumps(row), flush=True)

    assert sha256(module_file) == module_hash, 'Kernel changed during verification'
    failures = sum(not row['passed'] for row in observations)
    report = {
        'observed_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'source_dir': str(source), 'module': str(module_file), 'module_sha256': module_hash,
        'verifier_sha256': sha256(Path(__file__)),
        'environment': {'python': sys.version, 'python_executable': sys.executable,
                        'packages': {name: importlib.metadata.version(name) for name in
                                     ('acoular', 'numpy', 'scipy', 'h5py', 'traits', 'tables')}},
        'geometry': {'file': str(geometry), 'sha256': sha256(geometry), 'microphone_positions_m': microphones.tolist()},
        'signal_model': {'equation': 'p_m(t)=0.01 sin(2*pi*f*(t-||r_m-s||/c))/||r_m-s||',
                         'frequency_hz': frequency, 'sample_rate_hz': sample_rate, 'samples': samples,
                         'speed_of_sound_m_s': speed, 'fft_block_size': 256, 'independent_numpy_generator': True},
        'expected_frame': '(-XML x, +XML y, +XML z) for evaluation kernels; 2D returns x,y with known z input.',
        'tolerance': 'One grid interval per axis: 0.05 m Base, 0.1 m CLEAN; declared before execution.',
        'fixtures': fixtures, 'observations': observations, 'passed': len(observations) - failures, 'failed': failures,
        'limits': ['Single ideal coherent tone, one source, grid-aligned points, no noise/reverberation/calibration errors.',
                   'No external measurement frame, hardware, camera overlay, trained model or full application validation.',
                   'Floating-point residuals do not establish sub-grid or general localization accuracy.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'failed': failures, 'receipt': str(args.output)}), flush=True)
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
