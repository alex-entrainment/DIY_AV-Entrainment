import argparse
from pathlib import Path

from . import realtime_backend


def _backend(args: argparse.Namespace) -> None:
    """Handle the `backend` subcommand."""
    track_path = Path(args.path)
    if not track_path.is_file():
        raise SystemExit(f"Track file not found: {track_path}")

    if args.generate:
        output_path = track_path.with_suffix('.wav')
        with track_path.open('r', encoding='utf-8') as f:
            track_json = f.read()
        realtime_backend.write_wav(track_json, str(output_path))
        print(f"WAV written to {output_path}")
    else:
        realtime_backend.play_track_file(str(track_path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='realtime', description='Realtime backend utilities')
    subparsers = parser.add_subparsers(dest='command', required=True)

    backend_parser = subparsers.add_parser('backend', help='Play or render a track JSON file')
    backend_parser.add_argument('--path', required=True, help='Path to track JSON file')
    backend_parser.add_argument('--generate', action='store_true', help='Render full WAV instead of streaming')
    backend_parser.set_defaults(func=_backend)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':  # pragma: no cover
    main()
