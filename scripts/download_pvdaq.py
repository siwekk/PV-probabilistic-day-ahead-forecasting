"""Download a fixed, geographically diverse PVDAQ validation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree

import requests


BUCKET = "https://oedi-data-lake.s3.amazonaws.com"
SYSTEMS = [10020, 10, 10000, 11740]
YEAR = 2023
ROOT = Path(__file__).resolve().parents[1]


def list_keys(prefix: str) -> list[tuple[str, str, int]]:
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = []
    token = None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={quote(prefix, safe='/=')}"
        if token:
            url += f"&continuation-token={quote(token, safe='')}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        for item in root.findall("s3:Contents", namespace):
            keys.append(
                (
                    item.findtext("s3:Key", namespaces=namespace),
                    item.findtext("s3:ETag", namespaces=namespace).strip('"'),
                    int(item.findtext("s3:Size", namespaces=namespace)),
                )
            )
        if root.findtext("s3:IsTruncated", namespaces=namespace) != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=namespace)
    return keys


def download(key: str, target: Path, expected_etag: str) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    response = requests.get(f"{BUCKET}/{key}", stream=True, timeout=120)
    response.raise_for_status()
    with target.open("wb") as stream:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                stream.write(chunk)
                digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_etag:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {key}: {actual} != {expected_etag}")
    return {"key": key, "path": str(target), "bytes": target.stat().st_size, "md5": actual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=YEAR)
    args = parser.parse_args()

    raw = ROOT / "data" / "raw" / "pvdaq"
    catalogue = raw / "systems_20250729.csv"
    if not catalogue.exists():
        raise FileNotFoundError(f"Missing catalogue: {catalogue}")

    manifest: list[dict[str, object]] = []
    for system_id in SYSTEMS:
        prefix = f"pvdaq/csv/pvdata/system_id={system_id}/year={args.year}/"
        keys = list_keys(prefix)
        if not keys:
            raise RuntimeError(f"No PVDAQ files found for system {system_id}, year {args.year}")
        for key, etag, expected_size in keys:
            target = raw / "selected" / key.removeprefix("pvdaq/csv/")
            if target.exists() and target.stat().st_size == expected_size:
                actual = hashlib.md5(target.read_bytes()).hexdigest()
                if actual == etag:
                    manifest.append({"key": key, "path": str(target), "bytes": expected_size, "md5": actual})
                    continue
            manifest.append(download(key, target, etag))

    output = raw / "selected" / "download_manifest.json"
    output.write_text(json.dumps({"year": args.year, "systems": SYSTEMS, "files": manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"systems": SYSTEMS, "files": len(manifest), "bytes": sum(x["bytes"] for x in manifest)}, indent=2))


if __name__ == "__main__":
    main()
