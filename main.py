import clean_modules
import hmm_to_text
import hmm_speech_models as hsm
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # ---------------------------- extrac segment ----------------------------
    # cp = clean_modules.extract_segments()
    # cp.set_your_list_word(['bảy', 'tám'])
    # cp.extract_audio_segment()
   
    # --------------------------- ttraining data ------------------------------
    hsm = hsm.Gaussian_hmm()
    # path = 'code example/canandcannot/data/'
    # list_labels_name = ['_con', 'hàng', 'học', 'nhà', 'sinh', 'tuyển']

    # hsm.set_input_path('extracted_data/')
    # hsm.set_labels_audio(list_labels_name)
    hsm.set_test_size(0.2)

    # check số kích thước dữ liệu và thống kê các file là tên của các word
    hsm.show_file_data()                

    # extract feature gán labels và chia tập train test
    X, y = hsm.build_data()

    # extract model
    model = hsm.hmm_training(X)

    # lưu model
    hsm.save_model(model)

    # chạy thử tập test
    print(hsm.model_test(X, y, model))

    # thử load model lên
    print(hsm.load_model_words())

    # ------------------speech to text ------------------------------
    # stt = hmm_to_text.speech_to_text()
    # stt.audio_record()
    # print(f'This text is : {stt.check_speech_word()}')