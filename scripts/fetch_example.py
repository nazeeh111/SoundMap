"""Explicit bounded download of the CC BY 4.0 Acoular wind-tunnel example.

Ennes Sarradj (2021), Microphone array data for basic examples in Acoular.
https://doi.org/10.5281/zenodo.5809069 . No network is used by the map command.
"""
import argparse
import hashlib
from pathlib import Path
import urllib.request

FILES = {
    'example_data.h5': (5807392, '6fe0e3c13ce45e98859509094debea0b5dacb96dc18df22be435b2ff066ec690'),
    'array_56.xml': (3428, 'a9ef0201c44fefedcc2e93e7f75f5728707c84c6fd1b92543c5ccfd6a380e083'),
    'example_calib.xml': (2586, '83a257f6e91840228c9c96fff5f575ea5b211b6b5673a4c312958326e4fc8847'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (size, expected) in FILES.items():
        target = args.output/name
        if target.exists():
            if target.stat().st_size != size or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                parser.exit(2, f'Refusing to replace existing mismatched file: {target}\n')
        else:
            url = f'https://zenodo.org/records/5809069/files/{name}?download=1'
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(size+1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
                parser.exit(2, f'Download size/digest mismatch for {name}; nothing written\n')
            with target.open('xb') as file:
                file.write(data)
        print(f'{name} {size} bytes sha256:{expected}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
