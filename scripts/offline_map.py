"""Map a recorded multichannel HDF5 file without microphones, camera, or weights."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--geometry', required=True, type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--invalid-channels', type=int, nargs='*', default=[], help='Zero-based bad channels, removed from data, geometry and calibration together')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--frequency', required=True, type=float)
    parser.add_argument('--z', required=True, type=float, help='Known scan-plane depth in metres, not estimated')
    parser.add_argument('--bounds', nargs=4, type=float, default=(-1.5,1.5,-1.5,1.5), metavar=('XMIN','XMAX','YMIN','YMAX'))
    parser.add_argument('--increment', type=float, default=0.05)
    parser.add_argument('--speed', type=float, default=343.0)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--peaks', type=int, default=3)
    parser.add_argument('--peak-separation', type=float, default=0.2)
    parser.add_argument('--output', required=True, type=Path, help='New result directory; existing paths are rejected')
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f'Output already exists: {args.output}')
        from evaluation.offline import source_map, save_map
        power, coordinates, report = source_map(args.input, args.geometry, calibration=args.calibration,
            frequency=args.frequency, z=args.z, bounds=args.bounds, increment=args.increment,
            speed=args.speed, block_size=args.block_size, invalid_channels=args.invalid_channels, start=args.start, stop=args.stop)
        report = save_map(args.output, power, coordinates, report, peaks=args.peaks, separation=args.peak_separation)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'SoundMap: {error}\n')
    print(json.dumps({'output':str(args.output), 'peaks':report['peaks'], 'compute_seconds':report['compute_seconds']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
