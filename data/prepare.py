"""Restore the 18 HAR70+ CSV files and check their original byte-level hashes."""
import argparse
import csv
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
SOURCE_URL = "https://archive.ics.uci.edu/static/public/780/har70.zip"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory, manifest):
    for row in manifest:
        path = directory / row["filename"]
        if not path.is_file():
            raise ValueError(f"Missing participant CSV: {path}")
        if path.stat().st_size != int(row["bytes"]) or checksum(path) != row["sha256"]:
            raise ValueError(f"Participant CSV differs from the research archive: {path}")


def extract(archive, staging, manifest):
    with zipfile.ZipFile(archive) as dataset:
        names = dataset.namelist()
        for row in manifest:
            filename = row["filename"]
            paths = {f"har70plus/{filename}", f"har70dataset/{filename}"}
            matches = [name for name in names if name in paths]
            if len(matches) != 1:
                raise ValueError(f"Expected one copy of {filename} in the archive, found {len(matches)}")
            info = dataset.getinfo(matches[0])
            if info.file_size != int(row["bytes"]):
                raise ValueError(f"Unexpected file size for {filename}")
            with dataset.open(info) as source, (staging / filename).open("wb") as dest:
                shutil.copyfileobj(source, dest)
    verify(staging, manifest)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="use the UCI ZIP or the original research ZIP offline")
    source.add_argument("--verify", action="store_true", help="check an existing directory only")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "raw", help="participant CSV directory")
    args = parser.parse_args()
    output = args.output.resolve()
    with (DATA_DIR / "manifest.csv").open(newline="", encoding="utf-8") as table:
        manifest = list(csv.DictReader(table))

    if args.verify or output.exists():
        verify(output, manifest)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".har70-", dir=output.parent) as tmp:
            staging = Path(tmp) / "raw"
            staging.mkdir()
            archive = args.archive
            if archive is None:
                archive = Path(tmp) / "har70.zip"
                print("Downloading the HAR70+ archive from UCI...", flush=True)
                with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, archive.open("wb") as dest:
                    shutil.copyfileobj(response, dest)
            extract(archive, staging, manifest)
            try:
                staging.rename(output)
            except OSError:
                if not output.is_dir():
                    raise
                verify(output, manifest)
    rows = sum(int(row["samples"]) for row in manifest)
    print(f"Verified {len(manifest)} participant CSVs ({rows:,} samples) in {output}")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        sys.exit(f"Data preparation failed: {error}")
