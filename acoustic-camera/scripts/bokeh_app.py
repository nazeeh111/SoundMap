"""Classical dashboard by default; explicit --model enables optional neural mode."""
import argparse
from pathlib import Path

import acoular as ac
from bokeh.plotting import curdoc
from ui import Dashboard
from data_processing import Processor
from config import ConfigManager, get_uma16_index, calculate_alphas
from config.runtime import resolve_checkpoint


def build_document(doc, *, model=None, no_device=False):
    config = ConfigManager('config/config.json')
    checkpoint = resolve_checkpoint(model)
    # Never fall back to an arbitrary default microphone with unknown channels.
    device = None if no_device else get_uma16_index()
    geometry = Path(ac.__file__).parent / config.get('acoular.micgeom_file.base_path') / config.get('acoular.micgeom_file.file_name')
    Path('results').mkdir(exist_ok=True)
    processor = Processor(config, device, geometry, 'results', checkpoint, checkpoint is not None,
                          config.get('app_default_settings.z'))
    alphas = calculate_alphas(dx=config.get('app_settings.dx'), dz=config.get('app_settings.dz'))
    dashboard = Dashboard(config, processor, checkpoint is not None, alphas)
    doc.title = 'SoundMap'
    doc.add_root(dashboard.get_layout())
    return dashboard


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Optional model directory containing ckpt/best_ckpt/<integer>-*.keras')
parser.add_argument('--no-device', action='store_true', help='Build a read-only dashboard without device discovery')
args, _ = parser.parse_known_args()
ac.config.global_caching = 'none'
build_document(curdoc(), model=args.model, no_device=args.no_device)
