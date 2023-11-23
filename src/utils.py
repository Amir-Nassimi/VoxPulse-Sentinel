import os
import random
import librosa
import numpy as np
import soundfile as sf
from pedalboard import *
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, Compressor, Gain, PitchShift,Mix

def make_noisy(audio_file, noise_file, output_file, sample_rate):
    '''
    add noise from noise_file to audio_file
    inputs:
        audio_file: input audio file
        noise_file: noise file
        output_file: path to save noisy data
        sample_rate: desired sample_rate
    '''

    data, sr =librosa.load(audio_file, sr=sample_rate)
    ndata, sr = librosa.load(noise_file, sr=sample_rate)
    wdata = data + ndata
    sf.write(output_file, wdata, sample_rate)


def pitch_shift(audio_file, output_file, step, sample_rate):
    '''
    inputs:
        audio_file: input audio file
        step: pitch shift step from -5 to 10
        sample_rate: desired sample rate
    '''
    pitch_shift = Pedalboard([PitchShift(semitones=step), Gain(gain_db=2)])
    board = Pedalboard([Compressor(), Mix([pitch_shift])])
    with AudioFile(audio_file).resampled_to(sample_rate) as f:
        audio = f.read(f.frames)
    effected = board(audio, sample_rate)
    final_waveform = effected[0].tolist()
    sf.write(output_file, final_waveform, sample_rate)


def split_audio(audio_file, split_len, output_dir, sample_rate):
    '''
    Split an audio file into a number of audio files of a certain length
    '''
    name = os.path.splitext(os.path.basename(audio_file))[0]
    converted_audio = f'{output_dir}/convert_{name}.wav'
    os.system(f'ffmpeg -y -i {audio_file} -ar {sample_rate} -ac 1 {converted_audio}')
    data, _ = librosa.load(converted_audio, sr=sample_rate)
    samples = split_len*sample_rate
    count = int(len(data)/samples)
    for i in range(count):
        st = int(i*samples)
        end = int(st+samples)
        wdata = data[st:end]
        sf.write(f'{output_dir}/{name}-{i}.wav', wdata, sample_rate)
    os.remove(converted_audio)

def pad_audio(path, s_r, final_len):
    '''
    The audio file is loaded and padded to the desired length
    '''
    data, _ = librosa.load(path, sr=s_r)
    data = librosa.util.pad_center(data, size=final_len*s_r, axis=0)
    return data

def extract_spec(loaded_data , s_r):
    '''
    extract spectrogram from audio file
    '''
    spec = librosa.feature.melspectrogram(y=loaded_data, sr=s_r,n_mels=100, hop_length=160, win_length=400, n_fft=512)
    spec_db_2 = librosa.amplitude_to_db(spec)**2
    rgb_spec_db_2 = np.repeat(spec_db_2[..., np.newaxis], 3, -1)
    return rgb_spec_db_2

def split_dataset(dataset_dir, data_types):
    audio_files = []
    train_audio_files = []
    valid_audio_files = []

    for data_type in data_types:
        for (root , _, files) in os.walk(f'{dataset_dir}/{data_type}', topdown = 'True'):
            for file in files:
                audio_files.append(os.path.join(root, file))

    splt = int(0.9*len(audio_files))
    random.seed(43)
    random.shuffle(audio_files)
    for audio in audio_files[:splt]:
        train_audio_files.append(audio)
    for audio in audio_files[splt:]:
        valid_audio_files.append(audio)

    return train_audio_files, valid_audio_files
