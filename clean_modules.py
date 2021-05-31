
import os
from pydub import AudioSegment

class clean_speech:
    
    def __init__(self):
        self.output_path = 'data_export/'
        self.input_path = 'datan/'
        self.list_folder_data = os.listdir(self.input_path)
        self.list_name_data = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        self.list_words = []

    def set_output_input_path(self, output_path, input_path):
        self.output_path = output_path
        self.input_path = input_path
        self.list_folder_data = os.listdir(self.input_path)

    def set_your_list_word(self, list_words): # set your list of words that you want to extract from the audio dataset
        self.list_words = list_words

    def __str__(self):
        return 'This code for clean and extract audio data'

    def convert_to_time(self, time_start, time_end): # convert time normal to milisecond time
        h1, m1, s1 = time_start.split(':')
        h2, m2, s2 = time_end.split(':')
        return [int((float(h1)*3600 + float(m1)*60 + float(s1))*1000), int((float(h2)*3600 + float(m2)*60 + float(s2))*1000)]

    def get_time_segment(self, path, word):
        time_segment = []
        with open(path, "r", encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            line = str(line).lower()
            if '-->' in line: 
                # clean string in srt file
                array_line = line[0:len(line) - 1].split(' --> ')
                array_line[0] = array_line[0].replace(',', '.')
                array_line[1] = array_line[1].replace(',', '.')

                if lines[lines.index(line) + 1][0:-1] == word:
                    time_segment.append(self.convert_to_time(array_line[0], array_line[1]))
        return time_segment


    def extract_audio_segment(self, count = 0):
        for word in self.list_words:
            if not self.list_words:
                break
            else:
                data_save_path = self.output_path + word + '/'
                # create a new folder
                if not os.path.exists(data_save_path):
                    os.makedirs(data_save_path)

                for name_fol in self.list_folder_data:
                    print(f'processing {name_fol} .....')
                    for number in self.list_name_data:
                        # get time that responsible segment
                        list_segment = self.get_time_segment(self.input_path + name_fol + '/' + number + '.srt', word)
                        for _ in list_segment:
                            count = count + 1
                            Audio = AudioSegment.from_wav(self.input_path + name_fol + '/' + number + '.wav')
                            subaudio = Audio[_[0]:_[1]]
                            subaudio.export(data_save_path + name_fol + '_' + number + '_' + str(count) + '.wav', format="wav")
