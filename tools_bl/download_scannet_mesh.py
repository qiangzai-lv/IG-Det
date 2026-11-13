#!/usr/bin/env python3
"""Download selected ScanNet v2 meshes using only the Python standard library.

Examples (after obtaining ScanNet access and accepting its terms):
    python download_scannet_mesh.py --scene-ids scene0015_00 --with-metadata
    python download_scannet_mesh.py --scene-list scenes.txt --accept-terms
    python download_scannet_mesh.py --scene-ids scene0015_00 --dry-run

Official data organization: https://github.com/ScanNet/ScanNet#scannet-data
"""
from __future__ import annotations

import argparse
import http.client
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request


BASE_URL = 'https://kaldir.vc.cit.tum.de/scannet/'
TERMS_URL = BASE_URL + 'ScanNet_TOS.pdf'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'data' / 'scannet_val'
DEFAULT_VALIDATION_SPLIT = Path(__file__).resolve().parent / 'scannet_val_scenes.txt'
SCENE_PATTERN = re.compile(r'scene\d{4}_\d{2}')


def validate_file(path: Path, suffix: str) -> None:
    """Catch empty files, HTML error pages and missing mesh headers."""
    with path.open('rb') as source:
        head = source.read(1024 * 1024)
    if not head or b'<html' in head[:512].lower() or b'<!doctype html' in head[:512].lower():
        raise ValueError(f'Empty file or HTML response: {path}')
    if suffix.endswith('.ply'):
        if head.splitlines()[0].strip() != b'ply' or b'end_header' not in head:
            raise ValueError(f'Invalid PLY header: {path}')
        if not re.search(rb'element\s+vertex\s+[1-9]\d*', head):
            raise ValueError(f'PLY contains no vertices: {path}')


def download_file(url: str, destination: Path, suffix: str, *,
                  retries: int, timeout: float, overwrite: bool) -> str:
    if destination.exists() and not overwrite:
        validate_file(destination, suffix)
        print(f'SKIP {destination}', flush=True)
        return 'skipped'
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'IGDet-ScanNet-Mesh/1.0'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected = response.headers.get('Content-Length')
                expected = int(expected) if expected is not None else None
                count = 0
                last_progress = time.monotonic()
                with partial.open('wb') as target:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        count += len(chunk)
                        if time.monotonic() - last_progress >= 2:
                            total = f' / {expected / 1024**2:.1f} MiB' if expected is not None else ' MiB'
                            print(f'  {destination.name}: {count / 1024**2:.1f}{total}', flush=True)
                            last_progress = time.monotonic()
                if expected is not None and count != expected:
                    raise ValueError(f'Incomplete download: expected {expected} bytes, received {count}')
            validate_file(partial, suffix)
            partial.replace(destination)
            print(f'OK   {destination} ({count / 1024**2:.2f} MiB)', flush=True)
            return 'downloaded'
        except (OSError, ValueError, http.client.HTTPException) as error:
            partial.unlink(missing_ok=True)
            # Wrong scene ids or access errors cannot be resolved by retrying.
            permanent = isinstance(error, urllib.error.HTTPError) and error.code in (400, 401, 403, 404)
            if permanent or attempt == retries:
                raise
            print(f'RETRY {destination.name}: {error} ({attempt + 1}/{retries})', file=sys.stderr)
            time.sleep(min(2**attempt, 8))
        except KeyboardInterrupt:
            partial.unlink(missing_ok=True)
            raise
    raise AssertionError('unreachable')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scenes = parser.add_mutually_exclusive_group()
    scenes.add_argument('--scene-ids', nargs='+', help='One or more scene IDs, e.g. scene0015_00')
    scenes.add_argument('--scene-list', type=Path, help='UTF-8 text file, one scene ID per line; # comments allowed')
    parser.add_argument('-o', '--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--type', choices=['_vh_clean_2.ply', '_vh_clean.ply'], default='_vh_clean_2.ply',
                        help='Mesh file suffix (default: cleaned/decimated mesh)')
    parser.add_argument('--with-metadata', action='store_true', help='Also download each scene\'s .txt metadata')
    parser.add_argument('--split', choices=['scans', 'scans_test'], default='scans',
                        help='scans contains train/validation; scans_test is the official test set')
    parser.add_argument('--retries', type=int, default=2, help='Retries after the first attempt (default: 2)')
    parser.add_argument('--timeout', type=float, default=60, help='Socket timeout in seconds (default: 60)')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing files after a successful download')
    parser.add_argument('--accept-terms', action='store_true',
                        help='Confirm you already have ScanNet access approval and accept the official terms')
    parser.add_argument('--dry-run', action='store_true', help='Print URLs and output paths without accessing the network')
    args = parser.parse_args()
    if args.retries < 0 or args.timeout <= 0:
        parser.error('--retries must be nonnegative and --timeout must be positive')
    try:
        identifiers = args.scene_ids
        if args.scene_list is None and identifiers is None:
            args.scene_list = DEFAULT_VALIDATION_SPLIT
        if args.scene_list is not None:
            identifiers = [line.split('#', 1)[0].strip()
                           for line in args.scene_list.read_text(encoding='utf-8-sig').splitlines()]
        args.scenes = list(dict.fromkeys(scene for scene in identifiers if scene))
        if not args.scenes or any(SCENE_PATTERN.fullmatch(scene) is None for scene in args.scenes):
            parser.error('Expected scene IDs such as scene0015_00; scene lists must have one ID per line')
    except OSError as error:
        parser.error(str(error))
    return args


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.accept_terms:
        print(f'ScanNet access application: https://github.com/ScanNet/ScanNet#scannet-data\nTerms: {TERMS_URL}')
        if not sys.stdin.isatty():
            print('After approval and accepting the terms, rerun with --accept-terms.', file=sys.stderr)
            return 2
        answer = input('Have you obtained ScanNet access approval and accepted the terms? [y/N] ')
        if answer.strip().lower() not in ('y', 'yes'):
            return 2
    counts = {'downloaded': 0, 'skipped': 0, 'failed': 0}
    suffixes = [args.type] + (['.txt'] if args.with_metadata else [])
    for scene in args.scenes:
        for suffix in suffixes:
            filename = scene + suffix
            url = f'{BASE_URL}v2/{args.split}/{scene}/{filename}'
            destination = args.output_dir.expanduser() / args.split / scene / filename
            if args.dry_run:
                print(f'{url}\n  -> {destination}')
                continue
            try:
                status = download_file(url, destination, suffix, retries=args.retries,
                                       timeout=args.timeout, overwrite=args.overwrite)
                counts[status] += 1
            except (OSError, ValueError, http.client.HTTPException) as error:
                counts['failed'] += 1
                print(f'FAIL {url}: {error}', file=sys.stderr)
    if not args.dry_run:
        print('Finished: ' + ', '.join(f'{key}={value}' for key, value in counts.items()))
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        sys.exit(130)
