''' With the help of this module, you can split audio files into audio files of desired length '''
import os
import sys
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.utils import split_audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='input audio files directory')
    parser.add_argument('--output_dir', type=str, required=True, help='output audio files directory')
    parser.add_argument('--len', type=int, required=True, help='cutting length of audio files in seconds')
    parser.add_argument('--sr', type=int, default=16000, help='desired sample rate')
    args = parser.parse_args()

    audio_files = os.listdir(args.input_dir)
    for audio in audio_files:
        split_audio(f'{args.input_dir}/{audio}', args.len, args.output_dir, args.sr)

if __name__=="__main__":
    main()
