# SoundMap

**Development history:** Developed locally using Git before publication. These projects were published to GitHub together, so similar upload dates do not indicate when development began.

**Microphone-array measurements into sound-source maps.**

The application combines Acoular beamforming with a Bokeh dashboard and optional Flask camera overlay. Signal processing, geometry, configuration, and model interfaces remain intact.

## Set up

The original full application targets Python 3.11. Create an isolated environment before installing its pinned dependencies.

```bash
python -m pip install -r requirements.txt
python soundmap.py --help
# Requires the configured microphone array:
python soundmap.py run --no-flask
```

The command delegates to the existing application from its required working directory. A `.keras` model may be supplied with `--model`; the model route requires separate weights and TensorFlow. See the [original setup guide](docs/REFERENCE.md).

## Processing layout

![SoundMap processing architecture](overview.png)

This remains a research/test application with the original implementation's limitations. A real microphone array, device configuration, optional camera, and any chosen model are needed for live use. No microphone, camera, or device-capture route was started during packaging.

## Verification

See [the verification record](docs/VERIFICATION.md) for executed checks and unavailable hardware/model checks. Successful computational behavior is preserved; renamed commands add a presentation layer.

Maintained by [nazeeh111](https://github.com/nazeeh111).
