''' With the help of this module, you can add noise to audio data '''
import os
import sys
import random
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.utils import make_noisy

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
    parser.add_argument('--input_dir', type=str, required=True, help='3sec audio file paths')
    parser.add_argument('--noise_dir', type=str, required=True, help='3sec noise file paths')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory to write noisy files')
    parser.add_argument('--sr', type=int, default=16000, help='desired sample rate')
    args = parser.parse_args()

    audio_files = os.listdir(args.input_dir)
    noise_files = os.listdir(args.noise_dir)

    for audio in audio_files:
        name = os.path.splitext(os.path.basename(audio))[0]
        converted_audio = f'{args.output_dir}/convert_{name}.wav'
        os.system(f'ffmpeg -y -i {args.input_dir}/{audio} -ar {args.sr} -ac 1 {converted_audio}')

        numb = random.randint(0,len(noise_files)-1)
        noise_name =  os.path.splitext(os.path.basename(noise_files[numb]))[0]
        converted_noise = f'{args.output_dir}/convert_{noise_name}.wav'
        os.system(f'ffmpeg -y -i {args.noise_dir}/{noise_files[numb]} -ar {args.sr} -ac 1 {converted_noise}')

        make_noisy(converted_audio, converted_noise, f'{args.output_dir}/{audio}', args.sr)
        os.remove(converted_audio)
        os.remove(converted_noise)

if __name__=="__main__":
    main()
