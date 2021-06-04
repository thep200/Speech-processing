import os
import math
import pickle
import librosa
import numpy as np
import hmmlearn.hmm as hmm
from scipy.signal import savgol_filter
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class Gaussian_hmm():

    def __init__(self):
        self.labels_audio_names = ['ba', 'bảy', 'bốn', 'chín', 'có', 'hai', 'không', 'một', 'năm', 'sáu', 'tám', 'ngày', 'tháng', 'lớp', '_con', 'hàng', 'học', 'nhà', 'sinh', 'tuyển']
        self.states = [9, 12, 12, 12, 12, 9, 9, 12, 9, 12, 12, 12, 12, 12, 9, 12, 12, 12, 9, 12]
        self.input_path = 'data/test_code_data/'
        self.models_save_path = 'models_save/'
        self.load_models = {}
        self.test_size = 0.2
        self.dic = dict(zip([_ for _ in range(0, len(self.labels_audio_names))], self.labels_audio_names))

    def __str__(self):
        return f'this for havefun :)'

    def set_labels_audio(self, list_labels_name):
        self.labels_audio_names = list_labels_name

    def set_input_path(self, input_path):
        self.input_path = input_path

    def set_test_size(self, test_size):
        self.test_size = test_size

    def set_states(self, list_states):
        self.states = list_states

    def show_file_data(self):
        length = 0
        for name in self.labels_audio_names:
            # print(f'{os.listdir(self.input_path + name)} /n')
            length += len(os.listdir(self.input_path + name))
        print(length)

    # extract feature
    def get_mfcc(self, file_path):
        y, sr = librosa.load(file_path)

        y = savgol_filter(y, 5, 2, mode='nearest')

        # 10ms hop
        hop_length = math.floor(sr*0.010) 
        
        # 25ms frame
        win_length = math.floor(sr*0.025)
        
        # mfcc is 12 x T matrix
        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=12, n_fft=1024, hop_length=hop_length, win_length=win_length)
        
        # mfcc = librosa.segment.recurrence_matrix(mfcc, mode='connectivity')

        # substract mean from mfcc --> normalize mfcc
        mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
        
        # delta feature 1st order and 2nd order
        delta1 = librosa.feature.delta(mfcc, order=1, mode='nearest')
        delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
        
        # X is 36 x T
        X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
        
        # return T x 36 (transpose of X)
        # hmmlearn use T x N matrix
        return X.T 

    def build_data(self):
        all_data = {}
        all_labels = {}
        X = {'train': {}, 'test': {}}
        y = {'train': {}, 'test': {}}
        for cname in self.labels_audio_names:
            file_paths = [os.path.join(self.input_path, cname, i) for i in os.listdir(os.path.join(self.input_path, cname)) if i.endswith('.wav')]
            data = [self.get_mfcc(file_path) for file_path in file_paths]
            all_data[cname] = data
            all_labels[cname] = [self.labels_audio_names.index(cname) for i in range(len(file_paths))]
        
        for cname in self.labels_audio_names:
            x_train, x_test, _, y_test = train_test_split(all_data[cname], all_labels[cname], test_size=self.test_size, random_state=42)
            X['train'][cname] = x_train
            X['test'][cname] = x_test
            y['test'][cname] = y_test
        return X, y

    def hmm_training(self, X):
        model = {}
        for idx, cname in enumerate(self.labels_audio_names):
            start_prob = np.full(self.states[idx], 0.0)
            start_prob[0] = 1.0
            trans_matrix = np.full((self.states[idx], self.states[idx]), 0.0)
            p = 0.5
            np.fill_diagonal(trans_matrix, p)
            np.fill_diagonal(trans_matrix[0:, 1:], 1 - p)
            trans_matrix[-1, -1] = 1.0

            model[cname] = hmm.GaussianHMM(n_components=self.states[idx], verbose=True, n_iter=300, startprob_prior=start_prob, transmat_prior=trans_matrix, params='stmc',init_params='mc',random_state=42)
            model[cname].fit(X=np.vstack(X['train'][cname]), lengths=[x.shape[0] for x in X['train'][cname]])
        return model

    # save model
    def save_model(self, model):
        for cname in self.labels_audio_names:
            name = f'{self.models_save_path}model_{cname}.model'
            with open(name, 'wb') as file: 
                pickle.dump(model[cname], file)

    # test models
    def model_test(self, X, y, model):
        y_true = []
        y_pred = []
        for cname in self.labels_audio_names:
            for mfcc, target in zip(X['test'][cname], y['test'][cname]):
                scores = [model[cname].score(mfcc) for cname in self.labels_audio_names]
                pred = np.argmax(scores)
                y_pred.append(pred)
                y_true.append(target)
        report = classification_report(y_true, y_pred, target_names=self.labels_audio_names)
        
        print(f'y_true        y_pred')
        for y_t, y_p in zip(y_true, y_pred):
            print(f'{self.dic[y_t]}        {self.dic[y_p]}')
        return report

    # load models
    def load_model_words(self):
        for key in self.labels_audio_names:
            name = f'{self.models_save_path}model_{key}.model'
            with open(name, 'rb') as file:
                self.load_models[key] = pickle.load(file)
        return self.load_models


    
