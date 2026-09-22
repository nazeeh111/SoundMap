"""Compare real Acoular beamforming on a generated multichannel signal file."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import acoular as ac
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline-dir", required=True, type=Path)
args = parser.parse_args()
ac.config.global_caching = "none"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load(args.baseline_dir / "evaluation/beamforming_funcs.py", "original_beamforming")
branded = load(ROOT / "evaluation/beamforming_funcs.py", "branded_beamforming")
work = ROOT / ".verification"
work.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=work) as directory:
    signal = Path(directory) / "synthetic.h5"
    data = np.random.default_rng(42).normal(0, 0.01, (4096, 16))
    with h5py.File(signal, "w") as file:
        dataset = file.create_dataset("time_data", data=data)
        dataset.attrs["sample_freq"] = 16000.0
    expected = original.single_source_beamforming_2D(str(signal), 343, 1000, 1.5, block_size=256)
    actual = branded.single_source_beamforming_2D(str(signal), 343, 1000, 1.5, block_size=256)
    assert np.array_equal(actual, expected)
    assert np.isfinite(actual).all()
    # The retained kernel opens PyTables handles; close this disposable fixture.
    from tables.file import _open_files
    _open_files.close_all()
    print(json.dumps({"exact_beamforming_parity": True, "signal_shape": list(data.shape), "sample_rate": 16000, "coordinates": actual.tolist(), "scope": "Synthetic seeded noise, real Acoular kernel; no microphone/camera/model execution and no localization accuracy claim"}, indent=2))
