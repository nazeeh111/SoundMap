# Recorded-data source maps

The offline command executes Acoular's conventional frequency-domain beamformer.
It does not need TensorFlow, camera packages, devices, or weights. Use Python 3.12:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-offline.txt
python scripts/fetch_example.py --output .verification/airfoil-data
python soundmap.py offline \
  --input .verification/airfoil-data/example_data.h5 \
  --geometry .verification/airfoil-data/array_56.xml \
  --calibration .verification/airfoil-data/example_calib.xml \
  --invalid-channels 1 7 --start 0 --stop 16000 \
  --frequency 4000 --z=-0.68 --speed 346.04 \
  --bounds -0.6 0 -0.3 0.3 --increment 0.05 --block-size 128 \
  --output .verification/airfoil-4000
```

Only the explicit fetch command uses the network. It downloads 5.82 MB total,
checks fixed lengths/SHA256 digests, and refuses to replace mismatched files.
The map command writes `measurements.json`, `map.npz` (linear power and XYZ grid),
a self-contained `map.html` plot (inline Bokeh resources, SVG backend) into a **new** directory. Existing output is rejected.
The plot toolbar Save action downloads `soundmap-map.svg` by default.
Change `--frequency 5000` and use a new `--output` to compute another result.
On Linux, classical dashboard dependencies also require the system PortAudio
library (for Debian/Ubuntu: `sudo apt-get install libportaudio2`). The offline
requirements do not include sounddevice or require this acquisition library.

First import can compile Acoular's numerical kernels; this is separate from the
recorded `compute_seconds`, which measures the map's numerical pipeline.

Data attribution: Ennes Sarradj (2021), *Microphone array data for basic examples
in Acoular*, [DOI 10.5281/zenodo.5809069](https://doi.org/10.5281/zenodo.5809069),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `example_data.h5` is a
wind-tunnel recording from 56 microphones (25,600 samples at 51,200 Hz,
0.5 s in the stored file; the landing-page description says one second), unlike the synthetic
`three_sources.h5` in the same record. The raw recording is downloaded unchanged.
The exclusions, scan plane, speed, sample slice and FFT block above follow the
[Acoular airfoil frequency-domain example](https://acoular.org/acoular/v26.01/auto_examples/wind_tunnel_examples/example_airfoil_in_open_jet_freq_domain_methods.html).

## Analyze a multichannel WAV export

Pass an interleaved integer PCM `.wav` file to the same `--input` option and
supply the geometry XML in **exported channel order**. Export one channel per
microphone from the synchronized recording. Channel 1 in the editor becomes
channel index 0 in SoundMap. Keep the channels separate and use one common
sample rate and time origin. A stereo mixdown of a larger array discards the
channel measurements needed for localization.

The command accepts 8-, 16-, packed 24-, and 32-bit little-endian integer PCM
containers, including WAVE_FORMAT_EXTENSIBLE with a PCM subtype and a valid
precision from one bit up to the container width. Extensible integer samples use
the standard left-aligned representation. The RIFF size, chunk boundaries and
padding, frame alignment and PCM data length must agree. The reader admits up
to 4,096 RIFF chunks. Compressed or floating-point WAV, RF64, and separate
per-microphone files are outside this input contract. `--start` and `--stop` select frames as with HDF5. If the decoded
selection exceeds 128 MiB, choose a shorter frame range. The 2,000,000-frame
recording limit also applies to WAV.

Integer amplitudes are divided by the signed full-scale magnitude after
centering unsigned 8-bit samples at 128. JSON identifies the WAV input, PCM
container width and units, and hashes the actual WAV bytes. Apply calibration
factors in normalized amplitude units. Keep the original digital gain when
using pressure calibration. Calibration derived for raw integer counts must be
converted before use. The command uses channel order from the file, without
inferring a microphone layout from WAV speaker labels.

## Input and coordinate contract

- HDF5 dataset `time_data`: real finite `[sample, channel]` data, with dataset
  attribute `sample_freq` in Hz. No channel remapping is inferred.
- Geometry XML `pos` entries: `x`, `y`, `z` in metres, in recorded channel order.
  Optional calibration XML has one positive `factor` per original channel, in
  the same order. Excluded channels are removed from all three together.
- Grid and returned peaks use the **geometry XML frame with no mirror**.
  This differs from the older evaluation helpers' explicitly mirrored x frame.
  The signed `--z` is supplied from measurement setup, not estimated depth.
- Hanning FFT, 50% overlap, third-octave band, conventional beamforming with
  diagonal removal. JSON includes full input digests, versions and parameters.
- Plot levels are relative to the map maximum. Absolute pressure/SPL requires
  independently known physical input units and calibration. Map maxima can be
  sidelobes or reflections, so `--peaks` is not a source-count guarantee.

The airfoil run demonstrates genuine recorded-data processing. It provides no
independent ground-truth localization error, live UMA16 acquisition validation,
flow correction, camera registration or trained neural inference. The separate
`python scripts/verify_localization.py ...` and `test_offline_map` check generated
ideal signals, channel/frame signs and changed-input localization at grid scale.
They do not measure hardware accuracy.

## Resource and caller-state limits

The command admits at most 256 ordered microphones, 2,000,000 samples, 250,000
scan points and 4,000,000 microphone/grid pairs. Dominant work allocations
share a 512 MiB budget: the decoded WAV selection, if present, plus four
complex128 cross-spectral arrays of `(FFT size / 2 + 1) × valid channels × valid
channels`, plus the float64 all-frequency result of `(FFT size / 2 + 1) × actual
scan points`. Even a single-band request allocates that complete result. A valid small spatial
grid alone does not prevent a multi-gigabyte spectral allocation. Reduce
`--block-size`, increase `--increment`, or exclude known invalid channels when
that separate bound fails.
This estimates those spectral/result buffers, not the entire process's peak memory.

Offline calls are serialized while Acoular result caching is disabled, with the
caller's prior global policy restored on success and failure. The recording
handle is also closed when constructing or computing the numerical graph fails.
These guarantees cover this offline helper; they do not make arbitrary external
Acoular calls thread-safe.

### Installed offline command and publication

For the recorded-data command only, install the local project into an isolated
Python environment with `python -m pip install .`, then run `soundmap-offline`
with the same flags as `python scripts/offline_map.py`. It does not install the
camera or neural requirements. The existing pinned Python 3.12 offline
requirements remain available in `requirements-offline.txt`.

The output directory must be new. All files are generated in a temporary
sibling directory and published together with an atomic no-replace operation;
rendering failure removes the temporary files and permits retry. Existing paths,
including symlinks, are refused. The exclusive publication implementation supports
macOS, Linux with `renameat2`, and Windows. Other platforms fail closed. This is
not a claim of power-loss durability or protection from hostile path substitution.
A recording with no positive finite beamforming power is rejected, not rendered
as a misleading relative-level map.

Each run copies the recording, geometry and optional calibration to a private
read-only temporary snapshot before validation and analysis. Receipt hashes
identify those analyzed copies; the displayed original paths are annotations
and may no longer contain the same bytes. If a file changes while being copied,
the snapshot is the byte stream that was copied, not necessarily a single
point-in-time version of the source. Temporary input copies are removed when the
run completes or fails and are not included in the published result. Sufficient
local temporary disk space is required for the copied recording.

## Synthetic report preview

[The secondary README preview](synthetic-report-preview.svg) is the actual SVG
saved from a locally generated 16-channel synthetic fixture at 4096 Hz, with
scan plane z = 1.5 m supplied and bounds x = 0.2–0.8 m, y = 0–0.6 m at
0.05 m spacing. The strongest grid point is (0.5, 0.3, 1.5) m; other markers
may be sidelobes, not additional known sources. This is a software demonstration,
not a recording or a measurement of physical localization accuracy. Its SVG
SHA-256 is `84ca991689252a7cc92f9a95364ab1c300f43d720e606a37ff2225155eb7716f`.
The separate wind-tunnel image in the README retains its recorded-data attribution.

## Frequency sweeps from the same captured inputs

Use `--frequencies 3072 5120` instead of `--frequency 3072` to analyze up to 32
distinct third-octave bands together. The installed `soundmap-offline` and source
CLI accept the same option. Input snapshots, validation, CSM and beamforming
graph are shared; the existing one-band command and output layout stay available.
A sweep publishes a new directory containing `sweep.json` and `band-01/`,
`band-02/`, etc., each with the usual measurements, NPZ and offline HTML report.
Every band succeeds before the containing directory is published. A later-band
compute or render failure leaves no published partial sweep and permits retry.

Each band's `frequency_band` receipt gives the nominal lower/upper edges, actual
FFT bin indices/frequencies and spacing. Integration includes the lower edge and
excludes the upper edge, matching the installed Acoular implementation. Bands
containing no FFT bins are rejected before spectral construction; increase
`--block-size` to improve frequency resolution. A nonempty band does not establish
adequate resolution or physical accuracy. Shared `compute_seconds` measures the
entire sweep's spectral graph and maps, excluding snapshots and input validation;
it is not a separate per-band duration. Retained map arrays count toward the
512 MiB work admission in addition to the spectral and all-frequency buffers.

The Python `source_maps(..., frequencies=[...])` returns `(powers, coordinates,
reports)`, preserving requested order. `save_maps(...)` publishes these together.
`source_map(...)` remains the scalar API and uses the same implementation.

## Optional CLEAN-SC deconvolution

Add `--method clean-sc` to either a scalar map or a frequency sweep. The default
remains `--method conventional` and its numerical map is unchanged. CLEAN-SC uses
Acoular's existing `BeamformerCleansc`, not a new SoundMap algorithm. It can isolate
sparse coherent components that conventional peak picking confuses with the broad
main lobe or sidelobes. HTML, JSON and sweep entries identify the selected method;
keep that identity when comparing maps.

For example, add `--method clean-sc` to the airfoil command above and choose a new
output directory. The same recorded channels, calibration, third-octave FFT bins,
signed plane and coordinates are used. No TensorFlow, weights, camera or devices
are required. Algorithm controls are explicit in each receipt: automatic maximum
`2 * valid channels` iterations per frequency, damping 0.6 and stopn 3. These are
Acoular defaults, not retuned experimental parameters. The receipt records the
maximum, not an observed iteration count or a convergence certificate.

CLEAN-SC's repeated beamforming/coherence work is admitted separately, at most
200 million work items. The proxy counts the initial grid/CSM products plus the
automatic maximum iterations of grid/channel, CSM and 20 coherence-refinement
vector products per unique requested FFT bin. Overlapping bands share bins.
This is a dominant-work proxy, not a FLOP count or runtime guarantee. Additional
CSM/map scratch estimates join the existing 512 MiB buffer admission; the estimate
is still not whole-process peak memory. Reduce FFT size, band count, channels or
grid density when a bound is exceeded.

CLEAN-SC produces grid-point component powers; its point-sparse output is not a
source-count guarantee. Closely spaced, coherent or distributed sources, wrong
geometry/calibration and reflections can remain unresolved or misleading. The
independently generated two-source software fixtures include a close-source case
that **neither method resolves** at the chosen grid/band. No hardware localization
accuracy, broad statistical improvement or absolute SPL calibration is established.

References: [Acoular CLEAN-SC API](https://acoular.org/acoular/api_ref/generated/acoular.fbeamform.html),
Pieter Sijtsma (2007), *CLEAN based on spatial source coherence*, listed on the
[author's publications page](https://www.psa3.nl/publications-2/). Existing Acoular
license and provenance remain in `third_party/Acoular-LICENSE`.
