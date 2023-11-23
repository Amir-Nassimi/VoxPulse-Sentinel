''' With the help of this module, you can do pitch shift augmentation '''
import os
import sys
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.utils import pitch_shift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='3sec audio file paths')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory to write augmented files')
    parser.add_argument('--step', type=int, required=True, help='pitch shift step from -5 to 10')
    parser.add_argument('--sr', type=int, default=16000, help='desired sample rate')
    args = parser.parse_args()

    audio_files = os.listdir(args.input_dir)
    for audio in audio_files:
        pitch_shift(f'{args.input_dir}/{audio}', f'{args.output_dir}/{audio}', args.step, args.sr)

if __name__=="__main__":
    main()
