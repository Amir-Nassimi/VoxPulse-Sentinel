''' evaluate hotword detection model '''
import os
import argparse
import sys
from time import time
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.densenet.codes.main import Transcribe

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="input audio files directory")
    parser.add_argument("--temp_dir", default=f'{Path(__file__).resolve().parents[3]}/src/densenet/evaluation/temp', help="temp directory for writing resampled audio files")
    parser.add_argument("--sr", default=16000, help="model sample rate")
    args = parser.parse_args()
    audio_files = os.listdir(args.input_dir)
    true_positive, false_positive, false_negative, true_negative = 0,0,0,0
    rtf_sum=0
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    for audio in audio_files:
        audio_path = f'{args.input_dir}/{audio}'
        name = os.path.split(audio)[-1][:-4]
        convert_audio = f'{args.temp_dir}/convert_{name}.wav'
        os.system(f"ffmpeg -y -i {audio_path} -ar {args.sr} {convert_audio}")
        transcribe = Transcribe()
        t1 = time()
        predict_label, _= transcribe.predict_one_frame(convert_audio)
        t2 = time()
        rtf_sum+=((t2-t1)/3)
        os.remove(convert_audio)
        if predict_label=='heyholoo' and 'heyholoo' in audio_path:
            true_positive+=1
        elif predict_label=='heyholoo' and 'heyholoo' not in audio_path:
            false_positive+=1
        elif predict_label!='heyholoo' and 'heyholoo' in audio_path:
            false_negative+=1
        elif predict_label!='heyholoo' and 'heyholoo' not in audio_path:
            true_negative+=1

    print(true_positive)
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    rtf = rtf_sum/len(audio_files)

    print('precision:', precision)
    print('recall:', recall)
    print('rtf:', rtf)

if __name__=="__main__":
    main()
