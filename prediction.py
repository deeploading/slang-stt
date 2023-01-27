import argparse 

p = argparse.ArgumentParser() 
p.add_argument('--data_path', required=True)  # 평가용 데이터 위치
p.add_argument('--slang_model', type=str, default="DeepLoading/slang-stt") # 모델 위치. 로컬 모델을 지정할 수 있음. 지정하지 않을 시에는 huggingface에 업로드된 모델을 사용.
config=p.parse_args() 


"""대화"""

import pandas as pd
import json, os, tqdm
import pydub
import warnings
warnings.filterwarnings("ignore")

conv_text, conv_file, conv_path = [], [], []

path = config.data_path + '/라벨링데이터/대화'
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        with open(dirpath + "/" + filename, 'r', encoding='utf-8') as f:
            try:
                title = dirpath + filename
                if title.endswith(".json"):
                    data = json.load(f)
                    conv_path.append(path[:-10] + "/원천데이터/" + data['MediaUrl'])
                    conv_file += [data['MediaUrl'][-30:-4]]

                    temp_text = []
                    for i in range(len(data['Dialogs'])):
                        temp_text += [data['Dialogs'][i]['SpeakerText']]
                    conv_text.append(temp_text)
            except json.JSONDecodeError:
                continue


split_file, split_text, split_filepath, save_filepath = [], [], [], []

path = config.data_path + '/원천데이터/분할대화/'
for fileno in range(len(conv_file)):
    for i in range(len(conv_text[fileno])):
        split_file.append(conv_file[fileno])
        split_filepath.append(path + conv_path[fileno][-30:-4] + "_part{}.wav".format(i))
        save_filepath.append(conv_path[fileno])
        split_text.append(conv_text[fileno][i])

"""문장"""

sen_text, sen_file, sen_path = [], [], []

path = config.data_path + '/라벨링데이터/문장'
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'r', encoding = 'utf-8') as f:
            try:
                data = json.load(f)
                sen_path.append(path[:-9] + "원천데이터/" + data['MediaUrl'])
                sen_text += [data['Dialogs'][0]['SpeakerText']]
                sen_file += [data['MediaUrl'][-26:-4]]
            except json.JSONDecodeError:
                continue

"""단어"""

word_text, word_file, word_path = [], [], []

path = config.data_path + '/라벨링데이터/단어'
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'r', encoding = 'utf-8') as f:
            try:
                data = json.load(f)
                word_path.append(path[:-9] + "원천데이터/" + data['MediaUrl'])
                word_text += [data['Dialogs'][0]['SpeakerText']]
                word_file += [data['MediaUrl'][-26:-4]]
            except json.JSONDecodeError:
                continue

"""Preprocessing"""

import re

semi_text, semi_path, semi_file, semi_save_path = [], [], [], []
semi_text = split_text + sen_text + word_text
semi_path = split_filepath + sen_path + word_path
semi_file = split_file + sen_file + word_file
semi_save_path = save_filepath + sen_path + word_path

# 괄호가 중괄호로 되어 있는 경우가 있어 괄호로 바꾸기 
for i in range(len(semi_text)):
    if "}" in semi_text[i]:
        tx = semi_text[i].replace("}", ")")
        semi_text[i] = tx
        break

# 이중전사에서 한국어 삭제하기 
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

import pandas as pd
eval_data = pd.DataFrame(zip(semi_file, semi_text, semi_path, semi_save_path), columns = ['df_key', 'df_text', 'df_path', 'df_save_path'])

"""Prediction"""

# Import necessary library

# For managing audio file
import librosa

#Importing Pytorch
import torch

#Importing Wav2Vec tokenizer
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

__all__ = ["split_syllable_char", "split_syllables",
           "join_jamos", "join_jamos_char",
           "CHAR_INITIALS", "CHAR_MEDIALS", "CHAR_FINALS"]

import itertools

INITIAL = 0x001
MEDIAL = 0x010
FINAL = 0x100
CHAR_LISTS = {
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])),
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])),
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ]))
}
CHAR_INITIALS = CHAR_LISTS[INITIAL]
CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
CHAR_FINALS = CHAR_LISTS[FINAL]
CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                for k, v in CHAR_LISTS.items()}


def is_hangul_syllable(c):
    return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


def is_hangul_jamo(c):
    return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


def is_hangul_compat_jamo(c):
    return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


def is_hangul_jamo_exta(c):
    return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


def is_hangul_jamo_extb(c):
    return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


def is_hangul(c):
    return (is_hangul_syllable(c) or
            is_hangul_jamo(c) or
            is_hangul_compat_jamo(c) or
            is_hangul_jamo_exta(c) or
            is_hangul_jamo_extb(c))


def is_supported_hangul(c):
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")


def get_jamo_type(c):
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)


def split_syllable_char(c):
    """
    Splits a given korean syllable into its components. Each component is
    represented by Unicode in 'Hangul Compatibility Jamo' range.

    Arguments:
        c: A Korean character.

    Returns:
        A triple (initial, medial, final) of Hangul Compatibility Jamos.
        If no jamo corresponds to a position, `None` is returned there.

    Example:
        >>> split_syllable_char("안")
        ("ㅇ", "ㅏ", "ㄴ")
        >>> split_syllable_char("고")
        ("ㄱ", "ㅗ", None)
        >>> split_syllable_char("ㅗ")
        (None, "ㅗ", None)
        >>> split_syllable_char("ㅇ")
        ("ㅇ", None, None)
    """
    check_hangul(c)
    if len(c) != 1:
        raise ValueError("Input string must have exactly one character.")

    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
        init, med, final = x // 21, x % 21, offset % 28
        if not final:
            final = None
        else:
            final -= 1
    else:
        pos = get_jamo_type(c)
        if pos & INITIAL == INITIAL:
            pos = INITIAL
        elif pos & MEDIAL == MEDIAL:
            pos = MEDIAL
        elif pos & FINAL == FINAL:
            pos = FINAL
        idx = CHAR_INDICES[pos][c]
        if pos == INITIAL:
            init = idx
        elif pos == MEDIAL:
            med = idx
        else:
            final = idx
    return tuple(CHAR_LISTS[pos][idx] if idx is not None else None
                 for pos, idx in
                 zip([INITIAL, MEDIAL, FINAL], [init, med, final]))


def split_syllables(s, ignore_err=True, pad=None):
    """
    Performs syllable-split on a string.

    Arguments:
        s (str): A string (possibly mixed with non-Hangul characters).
        ignore_err (bool): If set False, it ensures that all characters in
            the string are Hangul-splittable and throws a ValueError otherwise.
            (default: True)
        pad (str): Pad empty jamo positions (initial, medial, or final) with
            `pad` character. This is useful for cases where fixed-length
            strings are needed. (default: None)

    Returns:
        Hangul-split string

    Example:
        >>> split_syllables("안녕하세요")
        "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ"
        >>> split_syllables("안녕하세요~~", ignore_err=False)
        ValueError: encountered an unsupported character: ~ (0x7e)
        >>> split_syllables("안녕하세요ㅛ", pad="x")
        'ㅇㅏㄴㄴㅕㅇㅎㅏxㅅㅔxㅇㅛxxㅛx'
    """

    def try_split(c):
        try:
            return split_syllable_char(c)
        except ValueError:
            if ignore_err:
                return (c,)
            raise ValueError(f"encountered an unsupported character: "
                             f"{c} (0x{ord(c):x})")

    s = map(try_split, s)
    if pad is not None:
        tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
    else:
        tuples = map(lambda x: filter(None, x), s)
    return "".join(itertools.chain(*tuples))


def join_jamos_char(init, med, final=None):
    """
    Combines jamos into a single syllable.

    Arguments:
        init (str): Initial jao.
        med (str): Medial jamo.
        final (str): Final jamo. If not supplied, the final syllable is made
            without the final. (default: None)

    Returns:
        A Korean syllable.
    """
    chars = (init, med, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)

    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx
    # final index must be shifted once as
    # final index with 0 points to syllables without final
    final_idx = 0 if final_idx is None else final_idx + 1
    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)


def join_jamos(s, ignore_err=True):
    """
    Combines a sequence of jamos to produce a sequence of syllables.

    Arguments:
        s (str): A string (possible mixed with non-jamo characters).
        ignore_err (bool): If set False, it will ensure that all characters
            will be consumed for the making of syllables. It will throw a
            ValueError when it fails to do so. (default: True)

    Returns:
        A string

    Example:
        >>> join_jamos("ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안녕하세요"
        >>> join_jamos("ㅇㅏㄴㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안ㄴ녕하세요"
        >>> join_jamos()
    """
    last_t = 0
    queue = []
    new_string = ""

    def flush(n=0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"invalid jamo character: {new_queue[0]}")
            result = new_queue[0]
        elif len(new_queue) >= 2:
            try:
                result = join_jamos_char(*new_queue)
            except (ValueError, KeyError):
                # Invalid jamo combination
                if not ignore_err:
                    raise ValueError(f"invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result

    for c in s:
        if c not in CHARSET:
            if queue:
                new_c = flush() + c
            else:
                new_c = c
            last_t = 0
        else:
            t = get_jamo_type(c)
            new_c = None
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL == INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, c)
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string

# Importing Wav2Vec pretrained model from local

tokenizer = Wav2Vec2Tokenizer.from_pretrained(config.slang_model)
model = Wav2Vec2ForCTC.from_pretrained(config.slang_model)

eval_text, eval_files, eval_filename, eval_save_path = [], [], [], []
for i in range(len(eval_data)):
    eval_text.append(eval_data['df_text'][i])
    eval_files.append(eval_data['df_path'][i])
    eval_filename.append(eval_data['df_key'][i])
    eval_save_path.append(eval_data['df_save_path'][i])

# speech-to-text 함수

def get_transcription(wav_file):
    file_name = wav_file
    
    # Loading the audio file
    audio, rate = librosa.load(file_name, sr = 16000)
    
    # Taking an input value
    input_values = tokenizer(audio, return_tensors = "pt").input_values
    
    # Storing logits (non-normalized prediction values)
    logits = model(input_values).logits
    
    # Storing predicted id's
    prediction = torch.argmax(logits, dim = -1)
    
    # Passing the prediction to the tokenzer decode to get the transcription
    transcription = tokenizer.batch_decode(prediction)[0]
    
    return join_jamos(transcription)

# 오디오 파일 리스트를 입력으로 하고 stt 결과를 리스트로 리턴하는 함수

from tqdm import tqdm
def get_transcriptions_list(data):
    result = []
    for d in tqdm(data):
        res = get_transcription(d) 
        result.append(res)
    return result

transcriptions = get_transcriptions_list(eval_files)

# CER 측정
import nlptutti as metrics

CER_sum = []
for i in range(len(transcriptions)):
    prediction = transcriptions[i]  # 예측한 것 
    reference = eval_text[i]  # 답 
    result = metrics.get_cer(reference, prediction)  # CER 측정
    CER_sum.append(result['cer']*100)

avg = sum(CER_sum, 0.0) / len(CER_sum)  # 평균 CER 계산 
print("CER : ", avg, "%")

predict_df = pd.DataFrame(zip(eval_filename, eval_save_path, transcriptions, CER_sum), columns = ['title', 'path', 'prediction', 'CER'])
predict_df
predict_df.to_csv('./stt_result.csv', index = False)

print("Prediction Complete")