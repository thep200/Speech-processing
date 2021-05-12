import re
import numpy as np
from datetime import time
import scipy.io as wavfile

path = "datan/01.srt"

# Chuyển thời gian trong file srt sang milisecond
def convert_to_time(time_start, time_end):
    h1, m1, s1 = time_start.split(':')
    h2, m2, s2 = time_end.split(':')
    return [int((float(h1)*3600 + float(m1)*60 + float(s1))*1000), int((float(h2)*3600 + float(m2)*60 + float(s2))*1000)]
    

# Tách thời gian trong file srt và đoạn text tương ứng 
def get_time_segment():
    time_segment = []

    with open(path, "r", encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        if '-->' in line: 
            # clean string để có thể lấy được time text trong file srt
            array_line = line[0:len(line) - 1].split(' --> ')
            array_line[0] = array_line[0].replace(',', '.')
            array_line[1] = array_line[1].replace(',', '.')

            # i là thời gian tương ứng i + 1 là đoạn text tương ứng với segment đó
            time_segment.append(convert_to_time(array_line[0], array_line[1])) 
            time_segment.append(lines[lines.index(line) + 1][0:-1])

    return time_segment


if __name__ == '__main__':
    my_segments = get_time_segment()
    for i in range(0, len(my_segments), 2):
        print(my_segments[i],'    ', my_segments[i + 1], '\n')

