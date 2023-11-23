''' Offline inference from hotword detection model '''
import os
import argparse
import sys
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.densenet.codes.main import Transcribe

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=f'{Path(__file__).resolve().parents[3]}/src/densenet/test/sample/sample.wav', help="input audio file")
    parser.add_argument("--chime", default=f'{Path(__file__).resolve().parents[3]}/src/densenet/test/sample/chime.wav', help="path to chime.wav")
    parser.add_argument("--output_dir", default=f'{Path(__file__).resolve().parents[3]}/src/densenet/test', help="output directory")
    parser.add_argument("--sr", default=16000, help="model sample rate")
    args = parser.parse_args()
    name = os.path.split(args.input)[-1][:-4]
    convert_name = f'{args.output_dir}/convert_{name}.wav'
    os.system(f"ffmpeg -y -i {args.input} -ar {args.sr} {convert_name}")
    transcribe = Transcribe()
    result = transcribe.offline_inference(convert_name, args.chime, args.output_dir)
    print(result)
    os.remove(convert_name)

if __name__=="__main__":
    main()
