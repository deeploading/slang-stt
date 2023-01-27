import argparse
import pydub
import warnings
warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument('--data_path', required=True)
config=p.parse_args()


# json에서 text 불러오기 - 대화
# array 이용해서 file path, text, file name 정보 저장
# 화자를 기준으로 파일과 텍스트를 자르기 위한 start time, end time 정보를 array에 저장 (타임스탬프 개념)

import json, os
path = config.data_path + "/라벨링데이터/대화"
text, file, file_path, start_time, end_time = [], [], [], [], []
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        with open(dirpath + "/" + filename, 'r', encoding='utf-8') as f:
            try:
                title = dirpath + filename
                if title.endswith(".json"):
                    data = json.load(f)
                    file_path.append(path[:-9]+ "원천데이터/" + data['MediaUrl'])
                    file += [data['MediaUrl'][-30:]]

                    temp_text = []
                    for i in range(len(data['Dialogs'])):
                        temp_text += [data['Dialogs'][i]['SpeakerText']]
                    text.append(temp_text)

                    temp_st = []
                    for i in range(len(data['Dialogs'])):
                      temp_st += [data['Dialogs'][i]['StartTime']]
                    start_time.append(temp_st)

                    temp_et = []
                    for i in range(len(data['Dialogs'])):
                      temp_et += [data['Dialogs'][i]['EndTime']]
                    end_time.append(temp_et)
            except json.JSONDecodeError:
                continue

import os
path = config.data_path + '/원천데이터/분할대화'
if os.path.isdir(path) is not True:
    os.mkdir(path)

    # starttime, endtime 기준으로 대화 audio 분할하기
    from pydub import AudioSegment

    for fileno in range(len(file)):
        audio_file = file_path[fileno]
        audio = AudioSegment.from_wav(audio_file)

        for i in range(len(start_time[fileno])):
            start = start_time[fileno][i] * 1000 #pydub works in millisec
            end =  end_time[fileno][i] * 1000 #pydub works in millisec
            # print("split at [ {}:{}] ms".format(start, end))
            audio_chunk=audio[start:end]
            audio_chunk.export(config.data_path + "/원천데이터/분할대화/{}_part{}.wav".format(file[fileno][:-4], i), format="wav")

split_file, split_text, split_filepath = [], [], []

# 화자를 기준으로 텍스트 분할하여 저장

for fileno in range(0, len(file)):
    for i in range(0, len(text[fileno])):
        split_file.append(file[fileno][:-4] + "_part{}.wav".format(i))
        split_filepath.append(config.data_path + "/원천데이터/분할대화/" + file_path[fileno][-30:-4] + "_part{}.wav".format(i))
        split_text.append(text[fileno][i])

# json에서 text 불러오기 - 문장
# array 이용해서 file path, text, file name 정보 저장

import json, os
path = config.data_path + "/라벨링데이터/문장"
sen_text, sen_file, sen_path = [], [], []
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'r', encoding = 'utf-8') as f:
            try:
                data = json.load(f)
                sen_path.append(path[:-9] + "원천데이터/" + data['MediaUrl'])
                sen_text += [data['Dialogs'][0]['SpeakerText']]
                sen_file += [data['MediaUrl'][13:35]]
            except json.JSONDecodeError:
                continue


# json에서 text 불러오기 - 단어
# array 이용해서 file path, text, file name 정보 저장

import json, os
path = config.data_path + "/라벨링데이터/단어"
word_text, word_file, word_path = [], [], []
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'r', encoding = 'utf-8') as f:
            try:
                data = json.load(f)
                word_path.append(path[:-9] + "원천데이터/" + data['MediaUrl'])
                word_text += [data['Dialogs'][0]['SpeakerText']]
                word_file += [data['MediaUrl'][13:35]]
            except json.JSONDecodeError:
                continue

semi_text, semi_path = [], []

# 분할한 대화, 문장,단어의 텍스트와 file path를 각각 하나의 배열에 저장
semi_text = split_text + sen_text + word_text
semi_path = split_filepath+ sen_path + word_path

# 괄호가 중괄호로 되어 있는 경우가 있어 괄호로 바꾸기 
for i in range(len(semi_text)):
    if "}" in semi_text[i]:
        tx = semi_text[i].replace("}", ")")
        semi_text[i] = tx
        break

# 이중전사에서 한국어 삭제하기 
import re

for i in range(len(semi_text)):
    slang_sent = semi_text[i]
    if re.search('[(]', slang_sent):
        for j in range(2):
            if re.search('[(]', slang_sent):
                start = slang_sent.index("(")
                try: 
                    end = slang_sent.index(")")
                except ValueError:
                    continue
                raw_word = slang_sent[start:end + 2]
                try: 
                    start = slang_sent.index("(")
                except ValueError:
                    continue
                try:
                    end = slang_sent.index(")")
                except ValueError:
                    continue
                kor_word = slang_sent[start:end + 1] + "/"
                slang_sent = slang_sent.replace(kor_word, "")
                end = slang_sent.index(")")
                slang_sent = slang_sent[:start] + raw_word[1:-2] + slang_sent[end + 1:]
        semi_text[i] = slang_sent

# 이중전사를 마친 텍스트, 파일 경로 배열을 데이터프레임으로 만듦
import pandas as pd
data = pd.DataFrame(zip(semi_text, semi_path), columns = ['df_text', 'df_path'])

# NAN 데이터 제거
from tqdm import tqdm

for i in tqdm(range(len(data))):
    if type(data['df_text'][i]) is float:
        data = data.drop(i)

data=data.sample(frac=1).reset_index(drop=True) # shuffle 후 index reset

data.to_csv('./data.csv', index = False) # 전처리를 마친 데이터프레임을 csv 파일로 저장
print("Preprocessing Complete")