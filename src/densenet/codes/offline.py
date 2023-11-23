import numpy as np
import librosa

def infer_signal(model, signal):
    '''
    It converts the input data into a spectrogram and then predicts the data class using the model
    Inputs:
        model: loaded model
        signal: specified length of input audio loaded
    '''
    spec = librosa.feature.melspectrogram(y=signal, sr=16000,n_mels=100, hop_length=160, win_length=400, n_fft=512)
    spec_db_2 = librosa.amplitude_to_db(spec)**2
    rgb_spec_db_2 = np.repeat(spec_db_2[..., np.newaxis], 3, -1)
    rgb_spec_db_2 = np.expand_dims(rgb_spec_db_2, axis=0)
    logits = model.predict(rgb_spec_db_2, batch_size=1)
    return logits

class FrameASR:
    def __init__(self, model,
                 frame_len, frame_overlap,
                 offset=0):
        '''
        Args:
            model : loadedmodel
            frame_len: amount of shift in each stage (seconds)
            frame_overlap: Half the difference between the length of the window and the length of the frame
        '''


        self.full_data = []
        self.model = model
        self.labelsource = {0:'heyholoo',
                            1:'noise',
                            2:'same',
                           }
        self.sr = 16000
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()

    def _decode(self, frame, offset=0):
        assert len(frame)==self.n_frame_len
        self.buffer[:24000] = 24000*[0]
        self.buffer[24000:-self.n_frame_len] = self.buffer[self.n_frame_len+24000:]
        self.buffer[-self.n_frame_len:] = frame

        self.full_data.extend(frame)
        logits = infer_signal(self.model, self.buffer)
        decoded = self._mbn_greedy_decoder(logits, self.labelsource)
        return decoded[:len(decoded)], self.full_data

    def transcribe(self, frame, merge=False):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged , data = self._decode(frame,self.offset)
        return unmerged, data

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.mbn_s = []

    @staticmethod
    def _mbn_greedy_decoder(logits, labelsource):
        mbn_s = []
        if logits.shape[0]:
            label_num = np.argmax(logits)
            class_label = labelsource[label_num]
            mbn_s.append(class_label)
        return mbn_s
