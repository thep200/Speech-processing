# Speech-processing INT3311 20 Kỳ 2 2021

## Thành viên nhóm
- Hồ Văn Thép     - 18021206
- Nguyễn Phi Long - 18020792

## Đề tài lựa chọn
Speech to text với bộ dữ liệu ở bài tập nhỏ lần 1 của lớp

## Mô hình sử dụng
- HMM 

## Phương pháp thực hiện
- từ rawdata của lớp extract ra các đoạn audio ứng với các từ muốn sử dụng
- dùng hàm `librosa.feature.mfcc` để lấy extract các fextures của các đoạn audio đó
- build dataset, gán labels và chia tập training và test
- sử dụng thư viện `hmmlearn` để training dữ liệu ở trên
- theo dõi report test sau training và xuất ra models của các từ
- implements models vào app
    - âm thanh đầu vào được ghi trực tiếp từ app và chuyển đến module nhận diện.
    - module predict words được build sử dụng models ở phần trên để dự đoán
    - âm thanh chuyển vào module predict và xuất ra text tương ứng.
    - render đoạn text tương ứng lên màn hình

## Tài liệu tham khảo
- [Filter banks, Mel-Frequency Cepstral Coefficients](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)

- [API Reference](https://hmmlearn.readthedocs.io/en/latest/api.html?fbclid=IwAR2kPKglFcpcRx7wAwPR-Z4-4Q8_fL8A0oFScnE6KHrtLf_qoMevIDB7W1k#hmmlearn.hmm.GaussianHMM)

- [librosa.feature.mfcc](https://librosa.org/doc/main/generated/librosa.feature.mfcc.html?fbclid=IwAR0yygIVKcgi0yOEBBKwq70s9fPQB7uoprh8cvbZI8e6aCJCQEmw2vtola8)

- [librosa.feature.delta](http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/generated/librosa.feature.delta.html?fbclid=IwAR22exjc2QvPAH-oztKJfPkAJCRIU7PhaXKtnUQLq4BYfRABH1J_jRR-DoA)

- [Window width and frame stride when calculating MFCC](https://github.com/librosa/librosa/issues/584?fbclid=IwAR2uj9dKYpuVYsu6tC8LCtN5b8-OteQa4LW4H0bg2vdVglV_iT_5hDkXFMg)

- [Split speech audio file on words in python](https://stackoverflow.com/questions/36458214/split-speech-audio-file-on-words-in-python?fbclid=IwAR3BrL4TslicaTB-rNvm03YBuGpCF1Wj_Rthzpk9DgCMw3GaeFV8CgbOxjs)


