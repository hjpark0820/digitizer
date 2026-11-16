"""Download only the authors' public neural legend-matching checkpoints."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
SPECS = {
    "ckpt_30.t7": ("1N49INpPKXWbAk7DCk8dRJX9MUnUa184d", 165916917),
    "mlp_ckpt.t7": ("1FtA-gJoU6Nf1nUTkMX2g_909l1rn96N0", 246529),
}


def download(item):
    name, (file_id, expected_size) = item
    path = ROOT / "weights" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    if not path.exists() or path.stat().st_size != expected_size:
        partial = path.with_suffix(".download")
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", ""):
                raise RuntimeError(f"{name}: Drive returned HTML instead of model data")
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f"{name}: size does not match official public folder metadata")
        partial.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"name": name, "path": str(path), "url": url, "file_id": file_id,
              "size": path.stat().st_size, "sha256": digest,
              "official_folder_url": "https://drive.google.com/drive/folders/15bKREf2EBORHBZx_xz53eVJzs9e4cZ7Q"}
    print(json.dumps(result), flush=True)
    return result


def main():
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(download, SPECS.items()))
    (ROOT / "weights" / "matching_provenance.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
