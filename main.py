# import clean_modules
import hmm_speech_models as hsm
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # cp = clean_modules.extract_segments()
    # cp.set_your_list_word(['con'])
    # cp.extract_audio_segment()
   
    hsm = hsm.Gaussian_hmm()
    
    # path = 'code example/canandcannot/data/'
    # list_labels_name = ['_con', 'hàng', 'học', 'nhà', 'sinh', 'tuyển']

    # hsm.set_input_path(path)
    # hsm.set_labels_audio(list_labels_name)
    hsm.set_test_size(0.3)

    # print(hsm.__str__())

    hsm.show_file_data()                

    # extract feature gán labels và chia tập train test
    X, y = hsm.build_data()

    # extract model
    model = hsm.hmm_training(X)

    # lưu model
    hsm.save_model(model)

    hsm.load_model_words()

    print(hsm.model_test(X, y, model))