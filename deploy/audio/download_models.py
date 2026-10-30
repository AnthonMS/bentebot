"""Download only named, checksum-pinned CPU models; no external Python execution."""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="/app/data/audio-models")
    parser.add_argument("--manifest", default="deploy/audio/model-manifest.json")
    args = parser.parse_args()
    target = Path(args.directory)
    target.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(args.manifest).read_text())
    for item in items:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Model names must be simple filenames")
        path = target / item["name"]
        if path.is_file():
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() == item["sha256"]:
                    print("Verified " + item["name"])
                    continue
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(item["url"], timeout=90) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > item["bytes"]:
                        raise ValueError("Model download exceeded pinned size")
                    digest.update(chunk)
                    output.write(chunk)
            if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError("Model download does not match pinned checksum")
            os.replace(temporary, path)
            print("Downloaded and verified " + item["name"])
        finally:
            temporary.unlink(missing_ok=True)
    (target / "manifest.json").write_text(json.dumps(items, indent=2) + "\n")


if __name__ == "__main__":
    main()
