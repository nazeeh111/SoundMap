"""Build the device-free view from the repository's recorded-data evidence."""

from base64 import b64encode
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


ASSETS = Path(__file__).resolve().parents[2] / 'docs' / 'presentation'
TEMPLATES = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parents[1] / 'templates'),
    autoescape=select_autoescape(),
)


def _data_url(filename, media_type):
    try:
        data = (ASSETS / filename).read_bytes()
    except OSError:
        return None
    return f'data:{media_type};base64,{b64encode(data).decode("ascii")}' if data else None


def recorded_view(message):
    bands = [
        {
            'frequency': frequency,
            'bins': bins,
            'peak_y': peak_y,
            'files': [
                (f'band-{frequency}.{extension}', _data_url(f'band-{frequency}.{extension}', media_type))
                for extension, media_type in [('json', 'application/json'), ('npz', 'application/octet-stream')]
            ],
        }
        for frequency, bins, peak_y in [
            (4000, '3600, 4000, 4400', '−0.10'),
            (5000, '4800, 5200, 5600', '+0.05'),
        ]
    ]
    return TEMPLATES.get_template('recordings.html').render(
        message=message,
        wide_map=_data_url('recorded-bands.png', 'image/png'),
        narrow_map=_data_url('recorded-bands-mobile.png', 'image/png'),
        has_reports=(ASSETS / 'report-exports.zip').is_file(),
        bands=bands,
    )
