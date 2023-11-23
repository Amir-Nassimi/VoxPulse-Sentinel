import os
import sys
import wave
import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import pyaudio as pa
sys.path.append(os.path.abspath(Path(__file__).resolve().parents[3]))
from src.utils import pad_audio, extract_spec
from src.densenet.codes.offline import FrameASR

class Transcribe:
    '''
    Examining each frame of data and detecting the presence or absence of hotword in each frame
    Arg:
        model_path: path to detection model
        step: infer every step seconds
        window_size: length of audio to be sent to NN
        sample_rate: Acceptable sampling rate for the model
    '''
    def __init__(self, model_path = f'{Path(__file__).resolve().parents[3]}/models/VoxPulse_Sentinel_Weight.h5', step=0.5, window_size = 3.0, sample_rate=16000):
        self.model = tf.keras.models.load_model(model_path)
        self.mbn = FrameASR(model=self.model,frame_len=step, frame_overlap = (window_size-step)/2,offset=0)
        self.chunk_size = int(step*sample_rate)
        self.sample_rate = sample_rate
        self.labelsource = {0:'heyholoo',
                            1:'noise',
                            2:'same',
                           }

    def offline_inference(self, wave_file, chime_file, outdir):
        """
        Detect each audio frame offline
        """

        i=0
        mbn_history = [0,0,0,0]
        chimes = [wave_file]
        wave_name = os.path.split(wave_file)[-1]

        wf = wave.open(wave_file, 'rb')
        data = wf.readframes(self.chunk_size)
        detection_info = []
        while len(data) > 0:
            data = wf.readframes(self.chunk_size)
            signal = np.frombuffer(data, dtype=np.int16)
            signal = signal.astype(np.float32)/32768.
            mbn_result = self.mbn.transcribe(signal)
            mbn_history[:-1]=mbn_history[1:]
            if mbn_result[0]==['heyholoo']:
                mbn_history[-1]=1
            else:
                mbn_history[-1]=0

            if mbn_history[1]==1 and mbn_history[2:]==[0,0]:
                pos = (i-2)*0.5
                detection_info.append({'command':'heyholoo', 'detect_time':pos})

                audio_clip = AudioSegment.from_wav(chimes[-1])
                chime = AudioSegment.from_wav(chime_file)
                audio_clip = audio_clip.overlay(chime, position = pos*1000)
                audio_clip.export(f"{outdir}/{wave_name[:-4]}_output.wav", format='wav')
                chimes.append(f"{outdir}/{wave_name[:-4]}_output.wav")

        self.mbn.reset()
        return detection_info


    def predict_one_frame(self, wave_file, final_len=3):
        data = pad_audio(wave_file, s_r=self.sample_rate, final_len=final_len)
        rgb_spec_db_2 = extract_spec(data, s_r=self.sample_rate)
        rgb_spec_db_2 = np.expand_dims(rgb_spec_db_2, axis=0)
        label = self.model.predict(rgb_spec_db_2, batch_size=1)
        label_num = np.argmax(label)
        label_prob = label[0][label_num]
        label_name = self.labelsource[label_num]
        return label_name, label_prob



    def online_inference(self, channels=1):

        self.mbn.reset()

        # Setup input device
        p = pa.PyAudio()
        print('Available audio input devices:')
        input_devices = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get('maxInputChannels'):
                input_devices.append(i)
                print(i, dev.get('name'))

        if len(input_devices):
            mbn_history = [0,0,0,0]
            dev_idx = -2
            while dev_idx not in input_devices:
                print('Please type input device ID:')
                dev_idx = int(input())

            # streaming
            stream = p.open(format=pa.paInt16,
                            channels=channels,
                            rate=self.sample_rate,
                            input=True,
                            input_device_index=dev_idx,
                            frames_per_buffer=self.chunk_size)


            while True:
                data = stream.read(self.chunk_size)
                signal = np.frombuffer(data, dtype=np.int16)
                signal = signal.astype(np.float32)/32768.
                mbn_result, data = self.mbn.transcribe(signal)
                mbn_history[:-1] = mbn_history[1:]
                if mbn_result==['heyholoo']:
                    mbn_history[-1]=1
                else:
                    mbn_history[-1]=0

                if mbn_history[1:-1]==[0,0] and mbn_history[-1]==1:
                    result = {'command':'heyholoo', 'detect_time': datetime.datetime.now().strftime('%H:%M:%S')}
                    print(result)

            print('Listening...')
            stream.start_stream()

            # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
            try:
                while stream.is_active():
                    sleep(0.1)

            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
                print()
                print("PyAudio stopped")

        else:
            print('ERROR: No audio input device found.')
