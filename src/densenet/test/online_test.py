''' Online inference from hotword detection model '''
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.densenet.codes.main import Transcribe

def main():
    transcribe = Transcribe()
    transcribe.online_inference()

if __name__=="__main__":
    main()
