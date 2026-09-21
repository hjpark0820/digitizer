"""Start the packaged web application independently of the working directory."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='Start the local Plot Digitizer web app.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--ai-ocr', action='store_true', help='Let the operating AI answer OCR snippets if Tesseract fails')
    args = parser.parse_args()
    if args.ai_ocr:
        os.environ['CHARTOCODE_OCR_MODE'] = 'auto-agent'
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    # Child detection/correction processes inherit consistent Windows encoding.
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    import uvicorn
    uvicorn.run('unified_server:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
