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
