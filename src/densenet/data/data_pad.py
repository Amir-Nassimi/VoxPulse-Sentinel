''' With the help of this module, you can pad audio files into desired length '''
import os
import sys
import argparse
from pathlib import Path
import soundfile as sf
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.utils import pad_audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='input audio files directory')
    parser.add_argument('--output_dir', type=str, required=True, help='output audio files directory')
    parser.add_argument('--len', type=int, required=True, help='final audio file len')
    parser.add_argument('--sr', type=int, default=16000, help='desired sample rate')
    args = parser.parse_args()

    audio_files = os.listdir(args.input_dir)
    for audio in audio_files:
        name = os.path.splitext(os.path.basename(audio))[0]
        converted_audio = f'{args.output_dir}/convert_{name}.wav'
        os.system(f'ffmpeg -i {args.input_dir}/{audio} -ar {args.sr} -ac 1 {converted_audio}')
        data = pad_audio(converted_audio, args.sr, args.len)
        sf.write(f'{args.output_dir}/{name}.wav', data, args.sr)
        os.remove(converted_audio)

if __name__=="__main__":
    main()
