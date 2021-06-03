# import clean_modules
import mfcc_extract
import gmmhmm_model
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # cp = clean_modules.extract_segments()
     # cp.set_your_list_word(['em'])
    # cp.extract_audio_segment()

    mfcc = mfcc_extract.mfcc()
    mfcc.set_input_path('code example/fruits/data_fruits/')
    mfcc.show_data_file()
    all_obs = mfcc.get_mfcc()
    all_labels = mfcc.set_lables()[1]

    for n,i in enumerate(all_obs):
        all_obs[n] /= all_obs[n].sum(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(all_obs, all_labels, test_size=0.1, random_state=0)

    print('Size of training matrix:', X_train.shape)
    print('Size of testing matrix:', X_test.shape)

    ys = set(all_labels)

    ms = [gmmhmm_model.gmmhmm() for y in ys]

    _ = [m.fit(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]

    ps = [m.transform(X_test) for m in ms]

    res = np.vstack(ps)

    predicted_labels = np.argmax(res, axis=0)

    missed = (predicted_labels != y_test)

    print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))
