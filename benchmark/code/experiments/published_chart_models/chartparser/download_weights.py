"""Download the checkpoint linked by the official ChartParser README, without loading it."""
from pathlib import Path
import hashlib
import json
import re
import requests

ROOT = Path(__file__).resolve().parent
URL = 'https://drive.google.com/uc?export=download&id=1P8i5cR8pr9ncKjaIzksrnViRZje6zf6C'

def main():
    target = ROOT / 'weights' / 'epoch-19.pth'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    session = requests.Session()
    response = session.get(URL, timeout=60, stream=True)
    response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type', ''):
        page = response.text
        assert 'Google Drive' in page and 'epoch-19.pth' in page
        fields = dict(re.findall(r'name="([^"]+)" value="([^"]+)"', page))
        assert fields['id'] == '1P8i5cR8pr9ncKjaIzksrnViRZje6zf6C'
        assert fields['confirm'] == 't'
        response = session.get('https://drive.usercontent.google.com/download', params=fields,
                               timeout=120, stream=True)
        response.raise_for_status()
    assert 'text/html' not in response.headers.get('Content-Type', '')
    digest = hashlib.sha256()
    size = 0
    partial = target.with_suffix('.pth.part')
    with partial.open('xb') as handle:
        for block in response.iter_content(1024 * 1024):
            handle.write(block)
            digest.update(block)
            size += len(block)
            if size % (64 * 1024 * 1024) == 0:
                print(f'Downloaded {size / 2**20:.0f} MiB', flush=True)
    assert size > 100_000_000
    partial.rename(target)
    manifest = {'official_source': URL, 'download_url': response.url,
                'filename': target.name, 'size_bytes': size, 'sha256': digest.hexdigest(),
                'load_policy': 'torch.load(weights_only=True); never unrestricted pickle'}
    target.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__':
    main()
