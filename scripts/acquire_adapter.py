"""Download the intentionally published adapter, verifying its frozen SHA256."""

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from adaptlm.artifacts.manifest import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/main-adapter-214"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing bundle; choose a new output directory")
    manifest = json.loads(Path("configs/adapter-release.json").read_text())
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "adapter.zip"
        with (
            urllib.request.urlopen(manifest["url"], timeout=120) as response,
            archive.open("wb") as file,
        ):
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > manifest["bytes"]:
                    raise ValueError("download exceeds frozen artifact size")
                file.write(chunk)
        if archive.stat().st_size != manifest["bytes"] or digest(archive) != manifest["sha256"]:
            raise ValueError("adapter download integrity mismatch")
        staged = Path(temporary) / "extracted"
        staged.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            if sum(p.file_size for p in zipped.infolist()) > 50_000_000:
                raise ValueError("uncompressed archive exceeds limit")
            for info in zipped.infolist():
                name = Path(info.filename)
                if (
                    name.name != info.filename
                    or info.is_dir()
                    or (info.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("archive must contain only flat regular files")
                (staged / name).write_bytes(zipped.read(info))
        bundle = json.loads((staged / "bundle.json").read_text())
        if bundle["bundle_id"] != manifest["bundle_id"]:
            raise ValueError("unexpected bundle identity")
        for name, expected in bundle["artifact_hashes"].items():
            if Path(name).name != name or digest(staged / name) != expected:
                raise ValueError("extracted adapter artifact mismatch")
        if digest(staged / "run.json") != bundle["run_manifest_hash"]:
            raise ValueError("training manifest mismatch")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, args.output)
    print(f"Verified {manifest['bundle_id']} at {args.output}")


if __name__ == "__main__":
    main()
