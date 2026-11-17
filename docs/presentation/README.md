# One recording. Two frequency bands.

This is an actual current-main conventional Acoular run, not a synthetic map or a live acquisition. Source commit `452eb6870a5601fa34d90bf81676015c185f5d82`, actual tree `5d340dcc71bf5ad9ea656fe7576fca72846e7c54`. The offline default and original workflows are unchanged.

## Run without devices

Use the existing Python 3.12 offline route from the repository root. The first two commands install the recorded offline dependency set; the explicit fetch is the only input download (about 5.82 MB):

```sh
python3.12 -m venv .venv-recorded
. .venv-recorded/bin/activate
python -m pip install -r requirements-kernels.txt -r requirements-offline.txt
python scripts/fetch_example.py --output inputs-airfoil
python soundmap.py offline \
  --input inputs-airfoil/example_data.h5 \
  --geometry inputs-airfoil/array_56.xml \
  --calibration inputs-airfoil/example_calib.xml \
  --invalid-channels 1 7 --start 0 --stop 16000 \
  --frequency 4000 --z=-0.68 --speed 346.04 \
  --bounds -0.6 0 -0.3 0.3 --increment 0.05 --block-size 128 \
  --output map-4000
```

Repeat the final command with `--frequency 5000` and a fresh `--output map-5000`. These are **two independent single-band calls**, not a frequency-sweep API. Output directories must be new. The launcher repeated the 4000 Hz call in a fresh directory and reproduced the exact numerical power/coordinate arrays; an existing-directory retry was refused with exit 2. HTML element IDs and measured timings can change, so artifact hashes identify the recorded exports rather than promising fresh byte identity. No microphone/camera device, model weight or network service is used by the map computation. First import may compile Acoular's numerical kernels.

## Read the input and result

The public recording contains 25600 samples from 56 microphones at 51200 Hz (0.5 s stored; the dataset landing page describes 1 second). This run analyzes sample indices `[0,16000)`, a 0.3125 s slice. Geometry position i matches data/calibration channel i; excluding zero-based 1 and 7 removes all three consistently, leaving 54 channels. Provided factors are applied, but absolute SPL needs independently known pressure units and calibration.

| Parameter | Actual setting |
| --- | --- |
| Array coordinates | [Unmodified XML](array_56.xml), metres; microphones at z=0 |
| Channel factors | [Unmodified XML](example_calib.xml), ordered per original channel |
| Scan coordinates | Geometry XML x/y/z, no mirror; separate from camera overlay frame |
| Supplied plane | z=-0.68 m; depth is not estimated |
| Sound speed | 346.04 m/s; no flow correction |
| Grid | x=-0.6…0 m, y=-0.3…0.3 m; 0.05 m spacing, 13×13 points |
| Spectrum | Hanning window, 50% overlap, FFT 128; spacing 400 Hz |
| Beamformer | Acoular 25.3 BeamformerBase, diagonal removal, third-octave integration |

The nominal third-octave edges are `f × 2^(±1/6)`. Acoular selects FFT bins at or above the lower edge and below the upper edge; it does not invent spectral samples between bins.

| Nominal band | Nominal edges / Hz | Actual bin centres / Hz | Strongest grid point (x,y,z) / m |
| --- | --- | --- | --- |
| 4000 Hz | 3563.595…4489.848 | 3600,4000,4400 | (-0.25,-0.10,-0.68) |
| 5000 Hz | 4454.494…5612.310 | 4800,5200,5600 | (-0.25,+0.05,-0.68) |

These are 169 sampled points per map, not a continuous spatial reconstruction. The figure keeps every cell, with no interpolation or smoothing, in the correct x/y orientation. The marker is the map maximum, not a known ground-truth source or a tracked trajectory. Other peaks can be sidelobes/reflections. The supplied z, array geometry and sound speed must match the physical setup; execution does not independently validate them. The coarse frequency/grid resolution does not establish physical localization accuracy.

Color is `10 log10(max(power / power.max(),1e-12))`, independently for each map. The display saturates values below −20 dB to the low color, with maximum 0 dB. Raw linear-power arrays retain their values in the NPZ files. Identical colors in different bands do not imply equal absolute pressure or power; a separate comparison needs a common physical reference and justified calibration.

## Download, export and reopen

[report-exports.zip](report-exports.zip) includes the actual two producer output directories: `measurements.json`, `map.npz` and self-contained `map.html` for each band. It also includes the original geometry/calibration, attribution, path-scope notes and a SHA256SUMS list. Extract into a fresh directory, verify the included hashes, and open either HTML file in your browser. Bokeh resources are inline, so the report is usable offline. Pan/zoom, reset the view, and use its toolbar **Save** action to export `soundmap-map.svg`. The JSON/NPZ preserve settings, hashes, grid coordinates and linear power. No JSON/NPZ import control or live dashboard acquisition is implied by opening an exported report.

The recording itself is explicitly external and fetched by the pinned script above; it is not a missing archive member. Receipt original paths annotate the producer's temporary input locations. Geometry/calibration copies are in `inputs/` inside the archive; transient private read-only input snapshots were intentionally removed. Source hashes refer to the pinned full repository. Every archive verification path is included; the archive README distinguishes these external/regenerated inputs. Fresh extraction hashes and arrays were actually checked, both HTML reports reopened, and a toolbar SVG export reopened. [Actual report capture](report-current.png).

Raw public figure data: [4000 Hz receipt](band-4000.json), [NPZ](band-4000.npz), [5000 Hz receipt](band-5000.json), [NPZ](band-5000.npz), [provenance and exact geometry](provenance.json). `python docs/presentation/render.py` regenerates desktop/phone SVG/PNG figures from these exact arrays using NumPy and Matplotlib, checks the producing source hashes, and makes no beamformer/device/provider call.

## Attribution and scope

Ennes Sarradj (2021), [Microphone array data for basic examples in Acoular](https://doi.org/10.5281/zenodo.5809069), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The fetched recording, microphone geometry and channel-factor files are unchanged. Derived maps, bin selection and peak-relative normalization are disclosed above. The source recipe follows the project's existing Acoular airfoil example; Acoular's own BSD license remains in [third_party](../../third_party/Acoular-LICENSE).

This single recording is a source/coordinate/export demonstration. It establishes no hardware accuracy, calibrated SPL, source-count guarantee, neural result, camera registration, flow correction, 3D depth recovery or general algorithm-performance advantage. The separate [ideal coordinate check](../../README.md#try-the-offline-coordinate-check), [complete offline contract](../OFFLINE.md) and [verification record](../VERIFICATION.md) retain their different scopes.
