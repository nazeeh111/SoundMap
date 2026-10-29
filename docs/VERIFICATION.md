# Verification

## Recorded-data and classical readiness, 2026-10-29

The default classical dashboard no longer selects a neural checkpoint or imports
TensorFlow. Explicit missing/malformed model requests fail closed; missing UMA16
acquisition stays disabled. Real Bokeh document construction (including classical
controls) passes without device discovery, TensorFlow or OpenCV. This is app
construction readiness, not a live audio test or HTTP readiness guarantee.

All 30 repository tests pass on Python 3.12.13/macOS ARM64, including the 16
existing launcher-process cases. The new production Acoular offline path locates
two independent generated spherical-wave inputs at the expected coordinates
within one 0.05 m grid interval, rejects malformed channel/rate/nonfinite data,
and refuses to overwrite existing results. These are ideal synthetic checks.

The [recorded-data command](OFFLINE.md) additionally processed the unchanged
CC BY 4.0 wind-tunnel recording with its geometry/calibration, excluding channels
1 and 7 consistently. Its strongest 4 kHz peak was (-0.25,-0.10,-0.68) m; changing
the band to 5 kHz gave (-0.25,0.05,-0.68) m. These are map peaks on a supplied
plane, not ground-truth localization errors. Current-source Bokeh HTML was
rendered and visually inspected; the axes, relative dB scale and peak markers
match the numerical map.

An independent NumPy Welch/steering calculation differed from the 4 kHz map by
9.35e-12 squared input units, 2.04e-7 relative to its peak (3.08e-7 at 5 kHz).
Acoular 25.3 explicitly rounds phase arguments to float32 inside its optimized
kernel even for float64 results. An explicit float32-phase reference reduced
peak-normalized discrepancies to 4.52e-9 and 2.99e-9, but strict pointwise
`allclose(rtol=1e-8, atol=1e-14)` still failed near small map values. No exact
double-precision equivalence is claimed; failed trials were retained rather
than relaxing that check. The passing coordinate checks are a separate gate.

No microphone/camera capture, real neural inference, known-target measurement,
flow correction, calibrated depth estimation or hardware accuracy is established.

The older verification records below describe their then-current source.


## Launcher process checks

The public command now observes Bokeh and Flask termination, preserves nonzero child status and cleans up its owned children. Previously, Bokeh-only failure became exit 0, the combined route ignored child exits, and unread Flask output pipes could block the child. Both modes now own their Bokeh process. Shutdown requests reach the application launcher through the public dispatcher; SIGINT produces exit 130 and SIGTERM produces exit 143. A child terminated by a signal uses the same `128 + signal` convention.

Sixteen process cases exercise the real public dispatcher and application launcher with controlled executable children. They cover early and later failures, orderly exit, simultaneous zero/nonzero exits, both model argument variants, arguments containing spaces, working directories, output exceeding pipe capacity, forwarded signals and a stubborn child that requires kill followed by reaping. Every process case asserts that no controlled child remains alive. Browser opening is checked with a substituted function; a failed startup must not call it. The existing mocked dispatch test also checks restoration of signal handlers.

The first two Bokeh-only failure cases reproduce child exit 7 becoming public exit 0. Four further cases reproduce hangs in the combined and signal-shutdown paths before supervision. All 17 command tests pass after the correction.

Optional imports, server executables and browser opening are substituted to avoid devices, browsers and servers. These checks establish launcher-process behavior, not Bokeh/Flask application behavior, TensorFlow models, microphone acquisition, camera integration or the full Python 3.11 stack. Process survival after the startup delays is not HTTP readiness. The numerical processing, calibration, data and model implementations are unchanged.

## Evaluation coordinates, 2026-10-01 UTC

Python 3.12.13, Acoular 25.3, NumPy 1.26.4, SciPy 1.14.0, h5py 3.11.0 and Traits 6.4.3 on macOS ARM64. This checks the offline evaluation kernels, not the full application's stated Python 3.11 environment.

All four evaluation kernels now return the same frame: `(-x_raw, +y_raw, +z_raw)` in metres, where raw coordinates use Acoular's packaged `minidsp_uma-16_mirrored.xml` geometry and grid. The 2D variants return x and y with known z supplied. This matches the active evaluation/model x-only mirror. The camera image overlay has a separate frame and was not changed.

Independent NumPy arithmetic generated spherical travel delays and `1/r` amplitudes from the actual 16 microphone positions. Acoular's source simulator was not used as the oracle. The real kernels processed a single 4096 Hz tone with 8192 samples at 16384 Hz. The expected frame and tolerance of one grid interval per axis were declared before execution: 0.05 m for Base and 0.1 m for CLEAN-SC.

- Before correction, the verifier passed 10 of 12 cases and failed both off-axis CLEAN-SC 3D cases: y errors were 0.6 m and 1.2 m against the evaluation frame.
- After changing only the CLEAN-SC 3D returned y sign, all 12 cases passed: four positions for each 2D kernel and two asymmetric positions for each 3D kernel. [Dated acceptance receipt](localization-verification.json).
- The existing command-dispatch test passed. It mocks dispatch to preserve arguments, working directories and child exit codes without starting devices.

These points are ideal and grid-aligned. Small floating-point residuals do not establish sub-grid accuracy. The 2D checks do not establish depth inference. No noise, reverberation, calibration error, measured dataset, external measurement frame, trained model, camera overlay or hardware was validated.

### Reproduce current checks

Use an existing environment with the kernel dependencies; the verifier does not install packages or start devices.

```bash
python -m unittest discover -s tests -v
python scripts/verify_localization.py --work-dir .verification/localization --output .verification/localization-result.json
```

The verifier exits 0 when all declared checks pass and 1 when any coordinate check fails. Its receipt records the executed kernel/verifier hashes, geometry/input hashes, dependency versions, actual and expected coordinates, residuals and limits. `--source-dir /path/to/previous-checkout` can demonstrate the original CLEAN-SC 3D discrepancy with the same verifier.

The [offline-kernel workflow](../.github/workflows/offline-kernels.yml) configures these same two checks on Python 3.12 with the observed versions pinned in `requirements-kernels.txt`, separate from the full application dependencies. It uses read-only repository permission and does not retain checkout credentials. A configured job is not evidence of a successful hosted run.

## Historical packaging checks, 2026-09-22

The following results describe the source before the evaluation-coordinate correction. Python 3.12.13, Acoular 25.3, NumPy 1.26.4, SciPy 1.14.0, h5py 3.11.0 and Traits 6.4.3 on macOS ARM64.

- 32 tracked numerical-source, firmware, configuration, and media files remained byte-identical to the baseline; 29 original Python files passed syntax parsing. [File hashes](source-verification.json).
- The new command dispatch test passed, checking argument preservation, paths containing spaces, expected working directories, and child exit-code propagation for every exposed command. Dispatch was mocked to avoid launching hardware routes.
- Real Acoular 2D beamforming completed twice on a generated 4096×16-channel noise recording and produced identical source-coordinate arrays. No mocks replaced the beamforming calculation. The fixture has no intended source position, so this establishes parity, not localization accuracy. [Kernel evidence](kernel-verification.json).
- New command help works without importing optional hardware or model dependencies.

### Reproduce historical parity

Run these commands from a separate historical packaging checkout **before** the evaluation-coordinate correction, against the original source-only baseline used for that comparison:

```bash
cd /path/to/historical-packaging-checkout
python -m unittest discover -s tests -v
python scripts/verify_sources.py --baseline-dir /path/to/original-source-only-baseline
python scripts/verify_beamforming.py --baseline-dir /path/to/original-source-only-baseline
```

The historical checkout and baseline are separate inputs; supplying an old baseline to the current checkout does not reproduce the old 32-file claim. The current strict source checker intentionally rejects the one changed evaluation kernel, `evaluation/beamforming_funcs.py`. It does not exempt that correction or claim all current numerical-source bytes remain identical. These commands do not depend on unpublished historical Git objects. Kernel dependencies are separated from the full application requirements. Source syntax checks do not prove that optional dependencies or hardware work.

## Not executed

The full Python 3.11 application stack, live microphone array, camera, Bokeh/Flask UI, TensorFlow model route, trained weights, and measurement-dataset evaluations were not run. Source scripts named test_processing.py and test_devices.py were not treated as safe automated tests because they can use devices.

These are bounded source and synthetic-runtime checks, not universal correctness or research-replication claims. The later change affects only the CLEAN-SC 3D evaluation return frame; original camera/streaming code, calibration, model implementation and recorded scientific images were preserved.
