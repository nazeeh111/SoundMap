# Verification

Executed on 2026-09-22. Python 3.12.13, Acoular 25.3, NumPy 1.26.4, SciPy 1.14.0, h5py 3.11.0, Traits 6.4.3 on macOS ARM64. This verifies the offline kernel only, not the full application’s stated Python 3.11 environment.

## Passed

- 32 tracked numerical-source, firmware, configuration, and media files remained byte-identical to the baseline; 29 original Python files passed syntax parsing. [File hashes](source-verification.json).
- The new command dispatch test passed, checking argument preservation, paths containing spaces, expected working directories, and child exit-code propagation for every exposed command. Dispatch was mocked to avoid launching hardware routes.
- Real Acoular 2D beamforming completed twice on a generated 4096×16-channel noise recording and produced identical source-coordinate arrays. No mocks replaced the beamforming calculation. The fixture has no intended source position, so this establishes parity, not localization accuracy. [Kernel evidence](kernel-verification.json).
- New command help works without importing optional hardware or model dependencies.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/verify_sources.py --baseline-dir /path/to/previous-checkout
python scripts/verify_beamforming.py --baseline-dir /path/to/previous-checkout
```

Parity commands take an explicit separate prior checkout; they do not depend on unpublished historical Git objects. Kernel dependencies are separated from the full application requirements. Source syntax checks do not prove that optional dependencies or hardware work.

## Not executed

The full Python 3.11 application stack, live microphone array, camera, Bokeh/Flask UI, TensorFlow model route, trained weights, and measurement-dataset evaluations were not run. Source scripts named test_processing.py and test_devices.py were not treated as safe automated tests because they can use devices.

These are bounded source and synthetic-runtime checks, not universal correctness or research-replication claims. Original numerical code, calibration, and recorded scientific images were preserved.
