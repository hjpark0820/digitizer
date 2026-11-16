"""Download the authors' public checkpoint; never execute downloaded content."""
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parent
FILE_ID = "1QyPykDBusW9LW0YejuTJfgfPr4qICtq8"
EXPECTED_SIZE = 487265684
URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"


def main():
    folder = ROOT / "weights"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "ckpt_L+D.t7"
    if not path.exists() or path.stat().st_size != EXPECTED_SIZE:
        partial = path.with_suffix(".download")
        with requests.get(URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", ""):
                raise RuntimeError("Drive returned HTML, not a checkpoint")
            total = 0
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
                    total += len(chunk)
                    if total // (50 * 1024 * 1024) != (total - len(chunk)) // (50 * 1024 * 1024):
                        print(f"Downloaded {total:,} / {EXPECTED_SIZE:,} bytes", flush=True)
        if partial.stat().st_size != EXPECTED_SIZE:
            raise RuntimeError(f"Unexpected size {partial.stat().st_size}")
        partial.replace(path)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    metadata = {"path": str(path), "url": URL, "official_folder_url": "https://drive.google.com/drive/folders/15bKREf2EBORHBZx_xz53eVJzs9e4cZ7Q", "size": path.stat().st_size, "sha256": digest, "expected_size_source": "official Google Drive public folder metadata"}
    (folder / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
