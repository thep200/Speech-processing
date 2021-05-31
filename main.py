import clean_modules
from python_speech_features import mfcc
from python_speech_features import logfbank

if __name__ == '__main__':
    cp = clean_modules.clean_speech()
    cp.set_your_list_word(['cho', 'mua', 'theo', 'trong'])
    cp.extract_audio_segment()

    # (rate, sig) = wav.read('data_export/01_925408.wav')
    # mfcc_feat = mfcc(sig,rate)
    # fbank_feat = logfbank(sig,rate)
    # print(fbank_feat[1:3, :])