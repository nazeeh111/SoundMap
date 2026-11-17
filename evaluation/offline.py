"""Recorded-data, frequency-domain Acoular beamforming in the geometry XML frame.

No acquisition devices, camera, TensorFlow, model weights, or network access.
The scan plane depth is supplied, never inferred from a two-dimensional map.
"""
import hashlib
from html import escape
import ctypes
import errno
import os
import shutil
import struct
import sys
import tempfile
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
_MAX_CLEAN_WORK_ITEMS = 200_000_000
_MAX_PCM_BYTES = 128 * 1024 * 1024


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _pcm_format(recording):
    """Validate RIFF bounds and return PCM metadata plus the first frame offset."""
    header = recording.read(12)
    if len(header) != 12 or header[:4] != b'RIFF' or header[8:12] != b'WAVE':
        raise ValueError('Expected a little-endian RIFF/WAVE recording')
    end = 8 + int.from_bytes(header[4:8], 'little')
    if end != os.fstat(recording.fileno()).st_size:
        raise ValueError('WAV RIFF size does not match the recording')
    sample_format = None
    data = None
    # Limit metadata work even if the recording is mostly empty chunks.
    for _ in range(4096):
        if recording.tell() == end:
            break
        header = recording.read(8)
        if len(header) != 8:
            raise ValueError('Incomplete WAV chunk header')
        kind, size = struct.unpack('<4sI', header)
        offset = recording.tell()
        next_chunk = offset + size + (size % 2)
        if next_chunk > end:
            raise ValueError('WAV chunk exceeds RIFF bounds')
        if kind == b'fmt ':
            if sample_format is not None or size < 16:
                raise ValueError('Expected one complete WAV format chunk')
            fields = recording.read(min(size, 40))
            tag, channels, rate, byte_rate, alignment, bits = struct.unpack_from('<HHIIHH', fields)
            if tag == 0xfffe:
                extension_size, valid_bits = struct.unpack_from('<HH', fields, 16) if size >= 40 else (0, 0)
                pcm_subtype = bytes.fromhex('0100000000001000800000aa00389b71')
                if not (22 <= extension_size <= size - 18 and 1 <= valid_bits <= bits
                        and fields[24:40] == pcm_subtype):
                    raise ValueError('Invalid WAVE_FORMAT_EXTENSIBLE integer PCM format')
            elif tag != 1:
                raise ValueError('Expected an uncompressed integer PCM WAV')
            if bits not in (8, 16, 24, 32):
                raise ValueError('Expected an 8-, 16-, 24- or 32-bit PCM container')
            width = bits // 8
            if not channels or not rate or alignment != channels * width or byte_rate != alignment * rate:
                raise ValueError('Inconsistent WAV sample rate or frame alignment')
            sample_format = width, channels, rate, alignment
        elif kind == b'data':
            if sample_format is None or data is not None:
                raise ValueError('Expected one WAV data chunk after the format chunk')
            data = offset, size
        recording.seek(next_chunk)
    if recording.tell() != end:
        raise ValueError('WAV exceeds the 4096 RIFF chunk limit')
    if data is None:
        raise ValueError('WAV requires a data chunk')
    width, channels, rate, alignment = sample_format
    offset, size = data
    if size % alignment:
        raise ValueError('WAV data is not a whole number of frames')
    return width, channels, size // alignment, float(rate), offset


def _read_pcm(recording, width, channels, data_offset, start, end):
    """Decode the selected frames as normalized full-scale amplitudes."""
    # Float64 preserves all 32 PCM bits; other signed PCM fits in float32.
    dtype = np.dtype(np.float64 if width == 4 else np.float32)
    if (end - start) * channels * dtype.itemsize > _MAX_PCM_BYTES:
        raise ValueError('Decoded WAV selection exceeds 128 MiB; narrow the sample range')
    frame_bytes = channels * width
    recording.seek(data_offset + start * frame_bytes)
    values = np.empty((end - start, channels), dtype=dtype)
    scale = 2.0 ** (8 * width - 1)
    for offset in range(0, len(values), 8192):
        batch = values[offset:offset + 8192]
        raw = recording.read(len(batch) * frame_bytes)
        if len(raw) != len(batch) * frame_bytes:
            raise ValueError('WAV data is shorter than the declared frame count')
        if width == 3:
            triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            words = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
            counts = (words ^ 0x800000) - 0x800000
        elif width == 1:
            counts = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        else:
            counts = np.frombuffer(raw, dtype=f'<i{width}')
        np.divide(counts.reshape(batch.shape), scale, out=batch)
    return values


def source_map(signal, geometry, *, frequency, z, bounds=(-1.5, 1.5, -1.5, 1.5),
               increment=0.05, speed=343.0, block_size=1024, calibration=None, invalid_channels=(), start=0, stop=None, method="conventional"):
    """Analyze one third-octave band; see source_maps for shared multi-band analysis."""
    powers, coordinates, reports = source_maps(signal, geometry, frequencies=[frequency], z=z,
        bounds=bounds, increment=increment, speed=speed, block_size=block_size,
        calibration=calibration, invalid_channels=invalid_channels, start=start, stop=stop, method=method)
    return powers[0], coordinates, reports[0]


def source_maps(signal, geometry, *, frequencies, z, bounds=(-1.5, 1.5, -1.5, 1.5),
                increment=0.05, speed=343.0, block_size=1024, calibration=None,
                invalid_channels=(), start=0, stop=None, method="conventional"):
    """Analyze owned snapshots; original paths are annotations, not evidence of current bytes.

    If an input changes during copying, the snapshot is the byte stream actually
    copied, not necessarily a point-in-time version of the original file.
    """
    if type(method) is not str or method not in ('conventional', 'clean-sc'):
        raise ValueError('method must be conventional or clean-sc')
    frequencies = tuple(float(f) for f in frequencies)
    if not 1 <= len(frequencies) <= 32 or not np.isfinite(frequencies).all() or len(set(frequencies)) != len(frequencies):
        raise ValueError('Require 1..32 distinct finite band centre frequencies')
    originals = {'input': Path(signal).resolve(), 'geometry': Path(geometry).resolve(),
                 'calibration': Path(calibration).resolve() if calibration else None}
    with tempfile.TemporaryDirectory(prefix='soundmap-inputs-') as directory:
        snapshots = {}
        for name, original in originals.items():
            if original is None:
                snapshots[name] = None
                continue
            suffix = '.wav' if name == 'input' and Path(signal).suffix.lower() == '.wav' else original.suffix
            snapshot = Path(directory) / (name + suffix)
            with original.open('rb') as source, snapshot.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            snapshot.chmod(0o400)
            snapshots[name] = snapshot
        powers, coordinates, reports = _source_maps_snapshot(
            snapshots['input'], snapshots['geometry'], calibration=snapshots['calibration'],
            frequencies=frequencies, z=z, bounds=bounds, increment=increment, speed=speed,
            block_size=block_size, invalid_channels=invalid_channels, start=start, stop=stop, method=method)
        for report in reports:
            for name, original in originals.items():
                report[name] = str(original) if original is not None else None
            report['input_provenance'] = ('Hashes identify private input snapshots used for validation and computation; '
                                      'original paths are annotations and may have changed during or after copying.')
        return powers, coordinates, reports


def _source_maps_snapshot(signal, geometry, *, frequencies, z, bounds=(-1.5, 1.5, -1.5, 1.5),
               increment=0.05, speed=343.0, block_size=1024, calibration=None, invalid_channels=(), start=0, stop=None, method="conventional"):
    """Return a linear-power map and provenance; channel i matches XML position i.

    Accept Acoular HDF5 or interleaved integer PCM WAV. Calibration factors
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
    parameters = [*frequencies, z, increment, speed, xmin, xmax, ymin, ymax]
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
    is_wav = signal.suffix.lower() == '.wav'
    pcm = None
    with (signal.open('rb') if is_wav else h5py.File(signal, 'r')) as recording:
        if is_wav:
            width, channels, total_samples, sample_rate, data_offset = _pcm_format(recording)
            pcm_bits = 8 * width
        else:
            if 'time_data' not in recording:
                raise ValueError('HDF5 requires a time_data dataset')
            data = recording['time_data']
            if data.ndim != 2:
                raise ValueError('Data channel count must equal ordered geometry positions')
            total_samples, channels = data.shape
            sample_rate = float(data.attrs.get('sample_freq', float('nan')))
            if data.dtype.kind not in 'fiu':
                raise ValueError('Require finite real samples, at least one FFT block and at most 2000000 samples')
        if channels != len(microphones):
            raise ValueError('Data channel count must equal ordered geometry positions')
        if not block_size <= total_samples <= 2000000:
            raise ValueError('Require finite real samples, at least one FFT block and at most 2000000 samples')
        if not np.isfinite(sample_rate) or sample_rate <= 0 or any(not 0 < f < sample_rate / 2 for f in frequencies):
            raise ValueError('sample_freq must be positive Hz and frequency strictly below Nyquist')
        if any(f * 2**(1/6) >= sample_rate / 2 for f in frequencies):
            raise ValueError('Entire third-octave band must be below Nyquist')
        fft_frequencies = np.fft.rfftfreq(block_size, 1 / sample_rate)
        bands = []
        for frequency in frequencies:
            lower, upper = frequency * 2**(-1/6), frequency * 2**(1/6)
            # Match Acoular synthetic's lower-inclusive, upper-exclusive slice.
            first, last = np.searchsorted(fft_frequencies, [lower, upper])
            if first == last:
                raise ValueError(f'Band at {frequency:g} Hz contains no FFT bins; increase block size')
            bands.append(dict(lower_hz=lower, upper_hz=upper,
                              fft_bin_indices=list(range(int(first), int(last))),
                              fft_frequencies_hz=fft_frequencies[first:last].tolist(),
                              fft_spacing_hz=sample_rate / block_size))
        if not isinstance(start, int) or start < 0 or (stop is not None and (not isinstance(stop, int) or stop > total_samples)):
            raise ValueError('Sample start/stop must be integer indices inside the recording')
        end = total_samples if stop is None else stop
        if end-start < block_size:
            raise ValueError('Selected sample range must contain at least one FFT block')
        sample_count = end-start
        if is_wav:
            pcm = _read_pcm(recording, width, channels, data_offset, start, end)
        else:
            for offset in range(0, total_samples, 8192):
                if not np.isfinite(data[offset:offset+8192]).all():
                    raise ValueError('Recording contains nonfinite samples')
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
    estimated_working_bytes = spectral_work_bytes + beamformer_result_bytes + (pcm.nbytes if is_wav else 0)
    if estimated_working_bytes > _MAX_WORKING_BYTES:
        raise ValueError('Estimated spectral working buffers and beamformer result exceed 512 MiB; reduce FFT block size, valid channels, grid density or WAV sample range')
    if calibration:
        factors = [float(p.attrib['factor']) for p in ET.parse(calibration).findall('.//pos')]
        if len(factors) != len(microphones) or not np.isfinite(factors).all() or min(factors) <= 0:
            raise ValueError('Calibration requires one positive finite factor per ordered channel')
    if grid_points * valid_channels > 4000000:
        raise ValueError('Scan exceeds 4000000 microphone-grid pairs; increase increment')
    # Retain at most 32 maps in addition to the already bounded Acoular result.
    retained_map_bytes = len(frequencies) * grid_points * np.dtype(np.float64).itemsize
    if estimated_working_bytes + retained_map_bytes > _MAX_WORKING_BYTES:
        raise ValueError('Estimated working buffers plus retained band maps exceed 512 MiB; reduce bands or grid density')
    unique_bins = len({index for band in bands for index in band['fft_bin_indices']})
    algorithm_parameters = dict(n_iter=0, damp=.6, stopn=3) if method == 'clean-sc' else {}
    clean_iterations = 2 * valid_channels if method == 'clean-sc' else 0
    # Additional CSM temporaries and map vectors in installed CLEAN-SC _calc;
    # this remains an allocation estimate, not whole-process peak memory.
    clean_working_bytes = (6 * valid_channels**2 * np.dtype(np.complex128).itemsize
                           + 4 * grid_points * np.dtype(np.float64).itemsize) if clean_iterations else 0
    if estimated_working_bytes + retained_map_bytes + clean_working_bytes > _MAX_WORKING_BYTES:
        raise ValueError('Estimated buffers including CLEAN-SC scratch exceed 512 MiB; reduce FFT size, channels or grid density')
    clean_work_items = (unique_bins * (grid_points * valid_channels**2
                        + clean_iterations * (grid_points * valid_channels + valid_channels**2 + 20 * valid_channels))
                        if clean_iterations else 0)
    if clean_work_items > _MAX_CLEAN_WORK_ITEMS:
        raise ValueError('CLEAN-SC repeated beamforming/coherence work exceeds 200000000; reduce bands, FFT size, channels or grid density')
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
            if is_wav:
                samples = ac.MaskedTimeSamples(data=pcm, sample_freq=sample_rate, invalid_channels=invalid_channels)
            else:
                samples = ac.MaskedTimeSamples(file=str(signal), invalid_channels=invalid_channels, start=start, stop=end)
            calibrated = ac.Calib(source=samples, file=str(Path(calibration).resolve()), invalid_channels=invalid_channels) if calibration else samples
            spectra = ac.PowerSpectra(source=calibrated, window='Hanning', overlap='50%', block_size=block_size)
            if method == 'clean-sc':
                csm = spectra.csm
                if any(not any(np.any(csm[index]) for index in band['fft_bin_indices']) for band in bands):
                    raise ValueError('Recording has no positive finite beamforming power in a requested band')
            steer = ac.SteeringVector(mics=mics, grid=grid, env=ac.Environment(c=speed))
            algorithm = ac.BeamformerBase if method == 'conventional' else ac.BeamformerCleansc
            beamformer = algorithm(freq_data=spectra, steer=steer, r_diag=True, **algorithm_parameters)
            powers = [np.array(beamformer.synthetic(frequency, num=3), copy=True).reshape(grid.shape)
                      for frequency in frequencies]
        finally:
            try:
                if samples is not None and samples.h5f is not None:
                    samples.h5f.close()
            finally:
                ac.config.global_caching = previous_cache_policy
    if any(not np.isfinite(power).all() or float(np.max(power)) <= 0 for power in powers):
        raise ValueError('Recording has no positive finite beamforming power in a requested band')
    coordinates = grid.gpos.reshape((3, *grid.shape))
    report = {
        'implementation_sha256': digest(__file__), 'schema_version': 1,
        'method': ('Acoular BeamformerBase' if method == 'conventional' else 'Acoular BeamformerCleansc') + ', diagonal removed, third-octave band',
        'algorithm': method,
        'algorithm_parameters': (dict(algorithm_parameters, max_iterations_per_frequency=clean_iterations)
                                 if method == 'clean-sc' else {}),
        'estimated_clean_working_bytes': clean_working_bytes,
        'estimated_clean_work_items': clean_work_items,
        'unique_requested_fft_bins': unique_bins,
        'clean_work_item_budget': _MAX_CLEAN_WORK_ITEMS if clean_iterations else None,
        'coordinate_frame': 'Geometry XML x,y,z in metres; channel i is XML position i; no mirror',
        'depth': 'Supplied scan-plane z, not estimated depth', 'scan_depth_m': float(z),
        'sample_rate_hz': sample_rate, 'samples': sample_count,
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
                   'No flow correction, camera registration, neural inference or physical accuracy validation.',
                   'CLEAN-SC resolves sparse coherent components, not a guarantee of distinct or closely spaced sources; compare with conventional maps.'],
    }
    if is_wav:
        report.update(input_format='wav-pcm', pcm_bits=pcm_bits, estimated_pcm_buffer_bytes=pcm.nbytes,
                      power_units='squared normalized full-scale amplitudes after optional calibration factors; '
                                  'absolute SPL requires pressure calibration')
    reports = [dict(report, frequency_hz=frequency, frequency_band=band,
                    sweep_band_count=len(frequencies), retained_band_map_bytes=retained_map_bytes,
                    compute_scope='Shared spectral graph and all requested bands; excludes snapshots and input validation')
               for frequency, band in zip(frequencies, bands)]
    return powers, coordinates, reports


def save_maps(output, powers, coordinates, reports, *, peaks=3, separation=0.2):
    """Publish every requested band or none; preserve previous output on any failure."""
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    if not 1 <= len(powers) <= 32 or len(powers) != len(reports):
        raise ValueError('Require one report per map, for 1..32 bands')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.soundmap-sweep-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'result'
        stage.mkdir()
        bands = []
        for index, (power, report) in enumerate(zip(powers, reports)):
            directory = f'band-{index + 1:02d}'
            saved = save_map(stage / directory, power, coordinates, report, peaks=peaks, separation=separation)
            bands.append(dict(directory=directory, frequency_hz=saved['frequency_hz'],
                              algorithm=saved.get('algorithm'), method=saved.get('method'), peaks=saved['peaks']))
        summary = dict(schema_version=1, bands=bands,
                       compute_seconds=reports[0]['compute_seconds'],
                       compute_scope='Shared spectral graph and all requested bands; excludes snapshots and input validation')
        (stage / 'sweep.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        _publish_directory(stage, output)
    return summary


def _publish_directory(source, destination):
    """Atomically publish a directory only if the destination is absent."""
    if os.name == 'nt':
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith('linux'):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, 'Atomic no-replace directory publication is unavailable')
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def save_map(output, power, coordinates, report, *, peaks=3, separation=0.2):
    """Write a fresh result directory without replacing a previous result."""
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    if not 1 <= peaks <= 20 or not np.isfinite(separation) or separation <= 0:
        raise ValueError('Require 1..20 peaks and positive finite peak separation')
    if power.ndim != 2 or not np.isfinite(power).all() or power.size == 0 or float(np.max(power)) <= 0:
        raise ValueError('Map must contain positive finite beamforming power')
    if coordinates.shape != (3, *power.shape) or not np.isfinite(coordinates).all():
        raise ValueError('Map coordinates must be finite and match the power grid')
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.soundmap-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'result'
        stage.mkdir()
        np.savez_compressed(stage/'map.npz', power=power, coordinates_m=coordinates)
        (stage/'measurements.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        from bokeh.embed import file_html
        from bokeh.models import ColorBar, Div, LinearColorMapper, SaveTool, Title
        from bokeh.layouts import column
        from bokeh.palettes import Magma256
        from bokeh.plotting import figure
        from bokeh.resources import INLINE
        relative = 10*np.log10(np.maximum(power/power.max(), 1e-12))
        method_label = {'conventional': 'Conventional', 'clean-sc': 'CLEAN-SC'}.get(report.get('algorithm'), 'Method unreported')
        plot = figure(width=700, height=600, sizing_mode="scale_width", match_aspect=True,
            title=f"SoundMap | {method_label}",
            x_axis_label='Geometry XML x (m)', y_axis_label='Geometry XML y (m)', output_backend='svg')
        plot.add_layout(Title(text=f"{report['frequency_hz']:g} Hz | supplied z={report['scan_depth_m']:g} m", text_font_size="10pt"), "above")
        plot.select_one(SaveTool).filename = "soundmap-map"
        mapper = LinearColorMapper(palette=Magma256, low=-20, high=0)
        step = report['increment_m']
        plot.image(image=[relative.T], x=float(coordinates[0].min())-step/2,
                   y=float(coordinates[1].min())-step/2,
                   dw=power.shape[0]*step, dh=power.shape[1]*step, color_mapper=mapper)
        peak_x = [p['position_m'][0] for p in found]
        peak_y = [p['position_m'][1] for p in found]
        plot.scatter(x=peak_x, y=peak_y, marker='cross', size=16, line_color='black', line_width=5)
        plot.scatter(x=peak_x, y=peak_y, marker='cross', size=16, line_color='white', line_width=2)
        plot.add_layout(ColorBar(color_mapper=mapper, title='Relative level (dB)'), 'right')
        strongest = found[0]['position_m']
        context = Div(text=(
            f"<h2>SoundMap</h2><p><strong>Method:</strong> {escape(str(report.get('method', 'Method not supplied')))}<br>"
            f"<strong>Frequency:</strong> {report['frequency_hz']:g} Hz · "
            f"<strong>Supplied plane z:</strong> {report['scan_depth_m']:g} m<br>"
            f"<strong>Strongest grid point (x, y, z):</strong> "
            f"({strongest[0]:g}, {strongest[1]:g}, {strongest[2]:g}) m</p>"
            "<p>Levels are relative to the map peak. Crosses mark separated peaks, "
            "which can include sidelobes or reflections; they are not a source count. "
            "The scan-plane depth is supplied.</p>"), sizing_mode="stretch_width")
        (stage/'map.html').write_text(file_html(column(context, plot, sizing_mode="stretch_width", max_width=700), INLINE, 'SoundMap recorded-data source map'))
        _publish_directory(stage, output)
    return report
