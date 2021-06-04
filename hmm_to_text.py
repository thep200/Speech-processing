import numpy as np
import sounddevice as sd
import hmm_speech_models as hsm
from scipy.io.wavfile import write

class speech_to_text:

    def __init__(self):
        self.duration = 1
        self.samplerate = 44100
        self.hsm = hsm.Gaussian_hmm()
        self.labels_audio_names = self.hsm.labels_audio_names
        self.model = self.hsm.load_model_words()
        self.audio_record_path = 'audio_record_save/'

    def set_durations(self, duration):
        self.duration = duration

    def set_samplerate(self, samplerate):
        self.samplerate = samplerate

    def audio_record(self):
        print(f'Starting recording sudio.....')
        myrecording = sd.rec(int(self.duration * self.samplerate), samplerate=44100, channels=1)
        # Wait until recording is finished
        sd.wait()  
        # Save as WAV file 
        write(f'{self.audio_record_path}output.wav', self.samplerate, myrecording) 

    def check_speech_word(self):
        record_mfcc = self.hsm.get_mfcc(f'{self.audio_record_path}output.wav')
        scores = [self.model[cname].score(record_mfcc) for cname in self.labels_audio_names]
        predict_word = self.labels_audio_names[np.argmax(scores)]
        # os.remove(f'{self.audio_record_path}output.wav')
        return predict_word
        