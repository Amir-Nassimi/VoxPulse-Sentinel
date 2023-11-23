import os
import sys
import argparse
import pathlib
import librosa
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.utils import pad_audio, extract_spec, split_dataset

def extract_features(audio_files, save_numpy_dir, save_pickle_path, data_types, sample_rate=16000):
    '''
    Extracting and storing the spectrogram of each data
    '''
    x_list, y_list = [], []
    for audio_file in tqdm(audio_files):
        name = os.path.splitext(os.path.basename(audio_file))[0] +'.npy'
        for data_type in data_types:
            if data_type in audio_file:
                audio_data_type=data_type
                break
        data = pad_audio(audio_file, sample_rate, 3)
        spect = extract_spec(data, sample_rate)
        if not os.path.exists(f"{save_numpy_dir}/{audio_data_type}"):
            pathlib.Path(f"{save_numpy_dir}/{audio_data_type}").mkdir(parents=True, exist_ok=True)
        np_path = os.path.join(save_numpy_dir,audio_data_type,name)
        np.save(np_path, spect)
        x_list.append(np_path)
        y_list.append(audio_data_type)

        data_dict = {}
        data_dict['x'] = x_list
        data_dict['y'] = y_list
        data_frame = pd.DataFrame(data_dict)
        data_frame.to_pickle(save_pickle_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,help='dataset directory path')
    parser.add_argument('--train_np',type=str, required=True,help='path to save train numpy files')
    parser.add_argument('--valid_np', type=str, required=True, help='path to save valid numpy files')
    parser.add_argument('--train_csv', type=str, required=True, help='path to save train csv file')
    parser.add_argument('--valid_csv', type=str, required=True, help='path to save valid csv file')
    args = parser.parse_args()

    #Data types is a list that contains the names of each data class. The data directory for each class must have the same name as that class
    data_types = ['heyholoo', 'noise', 'same']
    train_audio_files, valid_audio_files = split_dataset(args.dataset, data_types)
    extract_features(train_audio_files, args.train_np, args.train_csv, data_types)
    extract_features(valid_audio_files, args.valid_np, args.valid_csv, data_types)


if __name__=="__main__":
    main()
