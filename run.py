"""Start the packaged web application independently of the working directory."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='Start the local Plot Digitizer web app.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--region-mode', choices=('manual', 'agent'),
                        help='Manual GUI regions (default), or agent regions with agent vision OCR')
    parser.add_argument('--ocr-engine', choices=('tesseract', 'agent', 'auto-agent'),
                        help='Manual-mode OCR choice; agent regions always require agent OCR')
    parser.add_argument('--agent-workflow', action='store_true',
                        help='Legacy alias for --region-mode agent')
    parser.add_argument('--ai-ocr', action='store_true',
                        help='Legacy manual-mode alias for --ocr-engine auto-agent')
    args = parser.parse_args()
    if args.ai_ocr and args.ocr_engine not in (None, 'auto-agent'):
        parser.error('--ai-ocr conflicts with --ocr-engine ' + args.ocr_engine)
    if args.agent_workflow and args.region_mode == 'manual':
        parser.error('--agent-workflow conflicts with --region-mode manual')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    region_mode = args.region_mode or ('agent' if args.agent_workflow else 'manual')
    ocr_mode = args.ocr_engine or ('auto-agent' if args.ai_ocr else None)
    if region_mode == 'agent' and ocr_mode not in (None, 'agent'):
        parser.error('Agent regions require agent OCR; remove the Tesseract/fallback flag')
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    import unified_server
    import uvicorn
    # The server resolves omitted OCR to agent for AI regions, otherwise Tesseract.
    unified_server.configure_startup(ocr_mode=ocr_mode, region_mode=region_mode)
    uvicorn.run(unified_server.app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
