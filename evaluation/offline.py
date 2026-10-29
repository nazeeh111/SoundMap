"""Recorded-data, frequency-domain Acoular beamforming in the geometry XML frame.

No acquisition devices, camera, TensorFlow, model weights, or network access.
The scan plane depth is supplied, never inferred from a two-dimensional map.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from threading import RLock
import xml.etree.ElementTree as ET

import acoular as ac
import h5py
import numpy as np

_CACHE_LOCK = RLock()
_MAX_WORKING_BYTES = 512 * 1024 * 1024


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def source_map(signal, geometry, *, frequency, z, bounds=(-1.5, 1.5, -1.5, 1.5),
               increment=0.05, speed=343.0, block_size=1024, calibration=None, invalid_channels=(), start=0, stop=None):
    """Return a linear-power map and provenance; channel i matches XML position i.

    Input: Acoular HDF5 `time_data` [sample, channel], with dataset attribute
    `sample_freq` in Hz. Values must be finite real numbers. Calibration factors
    multiply the samples in XML order. No implicit coordinate mirrors are used.
    """
    signal, geometry = Path(signal).resolve(), Path(geometry).resolve()
    microphones = np.array([[float(p.attrib[a]) for a in ('x', 'y', 'z')]
                            for p in ET.parse(geometry).findall('.//pos')])
    if microphones.ndim != 2 or microphones.shape[1] != 3 or not 2 <= len(microphones) <= 256:
        raise ValueError('Geometry must contain 2..256 ordered microphone positions')
    if not np.isfinite(microphones).all() or len(np.unique(microphones, axis=0)) != len(microphones):
        raise ValueError('Microphone positions must be finite and distinct')
    xmin, xmax, ymin, ymax = map(float, bounds)
    parameters = [frequency, z, increment, speed, xmin, xmax, ymin, ymax]
    if not np.isfinite(parameters).all() or z == 0 or increment <= 0 or speed <= 0:
        raise ValueError('Finite frequency, bounds, nonzero z, positive increment and speed are required')
    if xmin >= xmax or ymin >= ymax:
        raise ValueError('Grid bounds must be increasing')
    if block_size not in (128, 256, 512, 1024, 2048, 4096):
        raise ValueError('FFT block size must be a power of two from 128 to 4096')
    grid = ac.RectGrid(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax, z=z, increment=increment)
    grid_points = int(grid.size)
    if grid_points > 250000:
        raise ValueError('Scan exceeds 250000 grid points; increase increment')
    with h5py.File(signal, 'r') as file:
        if 'time_data' not in file:
            raise ValueError('HDF5 requires a time_data dataset')
        data = file['time_data']
        if data.ndim != 2 or data.shape[1] != len(microphones):
            raise ValueError('Data channel count must equal ordered geometry positions')
        if not block_size <= data.shape[0] <= 2000000 or data.dtype.kind not in 'fiu':
            raise ValueError('Require finite real samples, at least one FFT block and at most 2000000 samples')
        sample_rate = float(data.attrs.get('sample_freq', float('nan')))
        if not np.isfinite(sample_rate) or not 0 < frequency < sample_rate / 2:
            raise ValueError('sample_freq must be positive Hz and frequency strictly below Nyquist')
        if frequency * 2**(1/6) >= sample_rate / 2:
            raise ValueError('Entire third-octave band must be below Nyquist')
        for offset in range(0, data.shape[0], 8192):
            if not np.isfinite(data[offset:offset+8192]).all():
                raise ValueError('Recording contains nonfinite samples')
        total_samples = int(data.shape[0])
        if not isinstance(start, int) or start < 0 or (stop is not None and (not isinstance(stop, int) or stop > total_samples)):
            raise ValueError('Sample start/stop must be integer indices inside the recording')
        end = total_samples if stop is None else stop
        if end-start < block_size:
            raise ValueError('Selected sample range must contain at least one FFT block')
        sample_count = end-start
    invalid_channels = list(invalid_channels)
    if len(set(invalid_channels)) != len(invalid_channels) or any(not isinstance(i,int) or not 0 <= i < len(microphones) for i in invalid_channels):
        raise ValueError('Invalid channels must be unique zero-based indices inside the recording')
    if len(microphones)-len(invalid_channels) < 2:
        raise ValueError('At least two valid microphone channels are required')
    valid_channels = len(microphones)-len(invalid_channels)
    # Acoular 25.3 calc_csm keeps upper, lower, combined and scaled complex128
    # CSM arrays. Bound those buffers before constructing the spectral engine;
    # the microphone-grid bound alone does not bound this quadratic allocation.
    spectral_work_bytes = 4 * (block_size//2+1) * valid_channels**2 * np.dtype(np.complex128).itemsize
    # BeamformerBase also allocates every frequency/grid result, even when
    # synthetic() requests only a single third-octave band.
    beamformer_result_bytes = (block_size//2+1) * grid_points * np.dtype(np.float64).itemsize
    estimated_working_bytes = spectral_work_bytes + beamformer_result_bytes
    if estimated_working_bytes > _MAX_WORKING_BYTES:
        raise ValueError('Estimated spectral working buffers and beamformer result exceed 512 MiB; reduce FFT block size, valid channels or grid density')
    if calibration:
        factors = [float(p.attrib['factor']) for p in ET.parse(calibration).findall('.//pos')]
        if len(factors) != len(microphones) or not np.isfinite(factors).all() or min(factors) <= 0:
            raise ValueError('Calibration requires one positive finite factor per ordered channel')
    if grid_points * valid_channels > 4000000:
        raise ValueError('Scan exceeds 4000000 microphone-grid pairs; increase increment')
    started = time.perf_counter()
    # Acoular's global policy overrides individual objects' cached=False flags.
    # Serialize our offline calls while disabling result reuse, then restore the
    # caller's policy even if graph construction, computation or cleanup fails.
    with _CACHE_LOCK:
        previous_cache_policy = ac.config.global_caching
        samples = None
        ac.config.global_caching = 'none'
        try:
            mics = ac.MicGeom(file=str(geometry), invalid_channels=invalid_channels)
            samples = ac.MaskedTimeSamples(file=str(signal), invalid_channels=invalid_channels, start=start, stop=end)
            calibrated = ac.Calib(source=samples, file=str(Path(calibration).resolve()), invalid_channels=invalid_channels) if calibration else samples
            spectra = ac.PowerSpectra(source=calibrated, window='Hanning', overlap='50%', block_size=block_size)
            steer = ac.SteeringVector(mics=mics, grid=grid, env=ac.Environment(c=speed))
            beamformer = ac.BeamformerBase(freq_data=spectra, steer=steer, r_diag=True)
            power = np.array(beamformer.synthetic(frequency, num=3), copy=True).reshape(grid.shape)
        finally:
            try:
                if samples is not None and samples.h5f is not None:
                    samples.h5f.close()
            finally:
                ac.config.global_caching = previous_cache_policy
    if not np.isfinite(power).all() or float(np.max(power)) <= 0:
        raise ValueError('Recording has no positive finite beamforming power in the requested band')
    coordinates = grid.gpos.reshape((3, *grid.shape))
    report = {
        'implementation_sha256': digest(__file__), 'schema_version': 1, 'method': 'Acoular BeamformerBase, diagonal removed, third-octave band',
        'coordinate_frame': 'Geometry XML x,y,z in metres; channel i is XML position i; no mirror',
        'depth': 'Supplied scan-plane z, not estimated depth', 'scan_depth_m': float(z),
        'frequency_hz': float(frequency), 'sample_rate_hz': sample_rate, 'samples': sample_count,
        'channels': len(microphones)-len(invalid_channels), 'input_channels':len(microphones),
        'excluded_zero_based_channels':invalid_channels, 'sample_start':start, 'sample_stop':end, 'speed_of_sound_m_s': float(speed), 'fft_block_size': block_size,
        'increment_m': increment, 'bounds_m': [xmin,xmax,ymin,ymax],
        'estimated_spectral_working_bytes': spectral_work_bytes,
        'estimated_beamformer_result_bytes': beamformer_result_bytes,
        'estimated_working_bytes': estimated_working_bytes,
        'working_budget_bytes': _MAX_WORKING_BYTES,
        'input_sha256': digest(signal), 'geometry_sha256': digest(geometry),
        'calibration_sha256': digest(calibration) if calibration else None,
        'input': str(signal), 'geometry': str(geometry), 'calibration': str(calibration) if calibration else None,
        'power_units': 'squared input units after optional calibration factors; absolute SPL requires pressure calibration',
        'compute_seconds': time.perf_counter()-started,
        'packages': {name: importlib.metadata.version(name) for name in ('acoular','numpy','scipy','h5py','traits','tables')},
        'limits': ['Peaks may include sidelobes/reflections, not a guaranteed source count.',
                   'Geometry, channel order, sound speed and supplied plane must match the recording.',
                   'No flow correction, camera registration, neural inference or physical accuracy validation.'],
    }
    return power, coordinates, report


def save_map(output, power, coordinates, report, *, peaks=3, separation=0.2):
    """Write a fresh result directory without replacing a previous result."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    if not 1 <= peaks <= 20 or not np.isfinite(separation) or separation <= 0:
        raise ValueError('Require 1..20 peaks and positive finite peak separation')
    remaining = power.copy()
    found = []
    for _ in range(peaks):
        index = np.unravel_index(np.argmax(remaining), remaining.shape)
        value = float(remaining[index])
        if value <= 0:
            break
        xyz = coordinates[:, index[0], index[1]]
        found.append({'position_m': xyz.tolist(), 'power': value,
                      'level_relative_to_map_peak_db': float(10*np.log10(value/power.max()))})
        distance = np.linalg.norm(coordinates[:2]-xyz[:2,None,None], axis=0)
        remaining[distance < separation] = -np.inf
    report = dict(report, peaks=found, peak_separation_m=separation)
    output.mkdir(parents=True)
    np.savez_compressed(output/'map.npz', power=power, coordinates_m=coordinates)
    (output/'measurements.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    from bokeh.embed import file_html
    from bokeh.models import ColorBar, LinearColorMapper
    from bokeh.palettes import Magma256
    from bokeh.plotting import figure
    from bokeh.resources import INLINE
    relative = 10*np.log10(np.maximum(power/power.max(), 1e-12))
    plot = figure(width=700, height=600, match_aspect=True,
        title=f"SoundMap | {report['frequency_hz']:g} Hz | supplied z={report['scan_depth_m']:g} m",
        x_axis_label='Geometry XML x (m)', y_axis_label='Geometry XML y (m)', output_backend='svg')
    mapper = LinearColorMapper(palette=Magma256, low=-20, high=0)
    step = report['increment_m']
    plot.image(image=[relative.T], x=float(coordinates[0].min())-step/2,
               y=float(coordinates[1].min())-step/2,
               dw=power.shape[0]*step, dh=power.shape[1]*step, color_mapper=mapper)
    plot.scatter(x=[p['position_m'][0] for p in found], y=[p['position_m'][1] for p in found],
                 marker='cross', size=16, line_color='cyan', line_width=2)
    plot.add_layout(ColorBar(color_mapper=mapper, title='Relative level (dB)'), 'right')
    (output/'map.html').write_text(file_html(plot, INLINE, 'SoundMap recorded-data source map'))
    return report
