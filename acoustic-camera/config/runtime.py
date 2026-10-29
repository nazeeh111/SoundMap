"""Optional model selection without importing neural libraries or touching devices."""
from pathlib import Path
import importlib.util


def resolve_checkpoint(model_directory):
    if model_directory is None:
        return None
    directory = Path(model_directory).expanduser().resolve() / 'ckpt/best_ckpt'
    candidates = []
    for path in directory.glob('*.keras'):
        prefix = path.stem.split('-', 1)[0]
        if prefix.isdigit() and '-' in path.stem:
            candidates.append((int(prefix), path.name, path))
    if not candidates:
        raise ValueError(f'No numbered .keras checkpoint in {directory}; expected <integer>-<name>.keras')
    if importlib.util.find_spec('tensorflow') is None:
        raise ValueError('Neural mode requires requirements-neural.txt and a compatible trained checkpoint')
    return max(candidates)[2]
