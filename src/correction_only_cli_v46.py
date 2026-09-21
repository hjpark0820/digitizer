"""Run saved evidence only: never imports or calls the detection runners."""
import argparse
from correction_session_v46 import correct_saved

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--iterations',type=int,default=5)
    parser.add_argument('--stop-policy',choices=('manual','auto'),default='manual')
    args = parser.parse_args()
    if not 1 <= args.iterations <= 50:
        parser.error('iterations must be 1–50')
    correct_saved(args.out_dir,args.iterations,stop_policy=args.stop_policy)
