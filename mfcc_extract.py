import os
import numpy as np
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided

class mfcc:
    def __init__(self):
        self.fpaths = []
        self.labels = []
        self.spoken = []
        self.input_path = 'extracted_data/'

    def __str__(self):
        return f'make a mfcc class'

    def set_input_path(self, input_path):
        print(f'Setting input path ...')
        self.input_path = input_path
        print(f'Done!')

    def show_data_file(self):
        print(f'Loading file data...')
        for f in os.listdir(self.input_path):
            for w in os.listdir(self.input_path + f):
                self.fpaths.append(self.input_path + f + '/' + w)
                self.labels.append(f)
                if f not in self.spoken:
                    self.spoken.append(f)
        print(f'Done!')
        return self.spoken

    def set_lables(self):
        print(f'Setting lables...')
        data = np.zeros((len(self.fpaths), 32000))
        maxsize = -1
        for n,file in enumerate(self.fpaths):
            _, d = wavfile.read(file)
            data[n, :d.shape[0]] = d
            if d.shape[0] > maxsize:
                maxsize = d.shape[0]
        data = data[:, :maxsize]

        all_labels = np.zeros(data.shape[0])
        for n, l in enumerate(set(self.labels)):
            all_labels[np.array([i for i, _ in enumerate(self.labels) if _ == l])] = n
        print(f'Done!')
        return [data, all_labels]

    # Short Time Fourier Transform (STFT)
    def stft(self, x, fftsize=64, overlap_pct=.5):   
        hop = int(fftsize * (1 - overlap_pct))
        w = np.hanning(fftsize + 1)[:-1]    
        raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
        return raw[:, :(fftsize // 2)]

    # Fast Fourier Transform (FFT)
    def peakfind(self, x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
        win_size = l_size + r_size + c_size
        shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
        strides = x.strides + (x.strides[-1],)
        xs = as_strided(x, shape=shape, strides=strides)
        def is_peak(x):
            centered = (np.argmax(x) == l_size + int(c_size/2))
            l = x[:l_size]
            c = x[l_size:l_size + c_size]
            r = x[-r_size:]
            passes = np.max(c) > np.max([f(l), f(r)])
            if centered and passes:
                return np.max(c)
            else:
                return -1
        r = np.apply_along_axis(is_peak, 1, xs)
        top = np.argsort(r, None)[::-1]
        heights = r[top[:n_peaks]]
        top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
        return heights, top[:n_peaks]

    # mfcc
    def get_mfcc(self):
        print(f'starting extract mfcc features...')
        all_obs = []
        data = self.set_lables()[0]
        for i in range(data.shape[0]):
            d = np.abs(self.stft(data[i, :]))
            n_dim = 6
            obs = np.zeros((n_dim, d.shape[0]))
            for r in range(d.shape[0]):
                _, t = self.peakfind(d[r, :], n_peaks=n_dim)
                obs[:, r] = t.copy()
            all_obs.append(obs)
        print(f'Done!')
        return np.atleast_3d(all_obs)
