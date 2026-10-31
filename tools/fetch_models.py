"""Run on a connected preparation machine; strictly offline mode is --verify."""
import argparse
import json
import os
import urllib.request
from pathlib import Path
from service.models import digest, paths
from service.settings import model_dir

SOURCES = Path(__file__).resolve().parents[1] / "models" / "SOURCES.json"

def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="Online preparation only")
    group.add_argument("--verify", action="store_true", help="No network access; check manifest if present")
    group.add_argument("--register", action="store_true", help="No network; record checksum manifest after manual model delivery")
    opts = p.parse_args()
    source = json.loads(SOURCES.read_text(encoding="utf-8"))
    root = model_dir()
    root.mkdir(parents=True, exist_ok=True)
    if opts.download:
        for item in source["artifacts"]:
            target = root / item["filename"]
            if target.exists() and (not item["sha256"] or digest(target) == item["sha256"]):
                print("present:", target.name)
                continue
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                with urllib.request.urlopen(item["url"], timeout=90) as resp, tmp.open("wb") as out:
                    while chunk := resp.read(1024 * 1024):
                        out.write(chunk)
                if item["sha256"] and digest(tmp) != item["sha256"]:
                    raise ValueError(f"SHA256 mismatch for {item['filename']}")
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
            print("downloaded:", target.name)
    present = paths(check=True, include_cv=True)
    manifest = {key: {"filename": obj.name, "sha256": digest(obj),
                      "bytes": obj.stat().st_size} for key, obj in present.items()}
    if opts.download or opts.register:
        (root / "manifest.local.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        snap = root / "manifest.local.json"
        if snap.exists():
            recorded = json.loads(snap.read_text(encoding="utf-8"))
            if recorded != manifest:
                raise ValueError("Model manifest does not match local files")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
