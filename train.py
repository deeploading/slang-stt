import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import argparse
p = argparse.ArgumentParser() 
p.add_argument('--slang_model', required=True) 
p.add_argument('--num_train_epochs', type=int, default=3) 
p.add_argument('--batch_size', type=int, default=4) 
p.add_argument('--weight_decay', type=float, default=0.005) 
p.add_argument('--learning_rate', type=float, default=1e-4)
p.add_argument('--step_size', type=int, default=1000)
config=p.parse_args() 


# read data csv to DataFrame
data = pd.read_csv('./data.csv', header=0)
train = data

test = train.sample(frac=0.1, random_state=21) # test datast을 만들기 위한 데이터프레임. trian dataset size의 10%

train_text = train['df_text'].tolist() # train 데이터프레임의 텍스트 부분을 리스트로 바꿈
train_path = train['df_path'].tolist() # train 데이터프레임의 파일 경로 부분을 리스트로 바꿈

test_text = test['df_text'].tolist() # test 데이터프레임의 텍스트 부분을 리스트로 바꿈
test_path = test['df_path'].tolist() # test 데이터프레임의 파일 경로 부분을 리스트로 바꿈

import datasets
from datasets import load_dataset, Dataset, Audio

# train 파일 경로, 텍스트 리스트를 train 딕셔너리로 만듦
train_dataset = Dataset.from_dict({"audio": train_path, "text": train_text}).cast_column("audio", Audio())

# test 파일 경로, 텍스트 리스트를 test 딕셔너리로 만듦
test_dataset = Dataset.from_dict({"audio": test_path, "text": test_text}).cast_column("audio", Audio())

import datasets
from datasets import load_dataset
from datasets import Dataset

sample_data = datasets.DatasetDict({"train":train_dataset, "test":test_dataset})

from datasets import load_dataset, load_metric, Audio

#resampling training data from 44100Hz to 16000Hz
sample_data['train'] = sample_data['train'].cast_column("audio", Audio(sampling_rate=16_000))
sample_data['test'] = sample_data['test'].cast_column("audio", Audio(sampling_rate=16_000))

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

import re

chars_to_ignore_regex = '[-=+,#/:^$@*\"※~&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》\n\{\}\t]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]) + " " # 특수문자 제거
    batch['text'] = split_syllables(batch['text']) # 자음, 모음 분리
    return batch

sample_data = sample_data.map(remove_special_characters) # 특수문자 제거하고 자음, 모음 분리 후 매핑

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocabs = sample_data.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=sample_data.column_names["train"])

vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

import json
with open("vocab.json", 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer

# json 파일을 이용해 Wav2Vec2CTCTokenizer 초기화
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

slang_model = config.slang_model

from transformers import Wav2Vec2FeatureExtractor

"""
Wav2Vec2FeatureExtractor: SequenceFeatureExtractor가 Superclass
Args:
    - feature_size (int, default to 0): feature vector 크기. 모델이 raw speech signal에서 훈련되었기 때문에 feature_size는 1
    - sampling_rate (int, defaults to 16000): wa2vec 모델의 input을 위한 16kHz
    - padding_value (float, default to 0.0): 패딩된 요소의 값. 짧은 input일수록 구체적인 값으로 패딩되어야 함
    - do_normalize (bool, optional, defaults to False): Whether or not to zero-mean unit-variance normalize the input.
    - return_attention_mask (bool, optional, defaults to False): 패딩된 토큰을 mask하기 위한 attention_mask를 사용해야 하는지. Wav2vec은 true, 다른 음성 모델은 false 권장.

"""

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

import gc
gc.collect()

def prepare_dataset(batch):
    audio = batch["audio"] # 오디오 데이터 로드 & resample

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0] # Wav2Vec2Processor가 데이터를 정규화
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids # 텍스트를 label로 변환(인코딩)
    return batch

sample_data = sample_data.map(prepare_dataset)

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

import evaluate
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-1b",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    # gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.gradient_checkpointing_enable()

model.freeze_feature_extractor()

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=config.slang_model, # 모델을 저장할 위치
  group_by_length=True,          # training 데이터셋의 샘플들을 비슷한 길이끼리 grouping할 것인지 (패딩 최소화, 보다 효율적, 동적 패딩 적용시 유용)
  per_device_train_batch_size=config.batch_size, # training할 때의 batch size
  per_device_eval_batch_size=config.batch_size,  # evaluation할 때의 batch size
  evaluation_strategy="epoch",   # 훈련 시 적용할 evaluation strategy
  num_train_epochs=config.num_train_epochs,            # training 에포크 수
  gradient_checkpointing=True,   # 역전파 시 메모리를 절약하기 위해 gradient checkpointing 사용
  save_steps=config.step_size,    # 지정 step마다 체크포인트 저장
  logging_steps=config.step_size, # 지정 step마다 log 출력
  learning_rate=config.learning_rate, # 학습률
  weight_decay=config.weight_decay, # 가중치 감소
  warmup_steps=config.step_size,  # Number of steps used for a linear warmup from 0 to learning_rate. Overrides any effect of warmup_ratio.
  save_total_limit=2, # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
  # push_to_hub=True,   # 모델을 save할 때마다 Hub에 모델을 push 할지말지
  optim="adafactor",  # 사용할 optimizer
)

from transformers import Trainer

trainer = Trainer(
    model=model, # 훈련할 모델
    data_collator=data_collator,           # train_datset 또는 eval_dataset의 element로부터 batch를 생성하기 위한 함수
    args=training_args,
    compute_metrics=compute_metrics,       # evaluation에 사용할 metric을 계산하는 함수. (CER)
    train_dataset=sample_data["train"],    # train에 사용되는 데이터셋
    eval_dataset=sample_data["test"],      # test에 사용되는 데이터셋
    tokenizer=processor.feature_extractor, # 데이터를 전처리할 때 사용하는 토크나이저
)

torch.cuda.empty_cache()

trainer.train()

# 토크나이저 저장
import json
 
# Data to be written
dictionary = {"bos_token": "<s>", 
              "eos_token": "</s>", 
              "unk_token": "[UNK]", 
              "pad_token": "[PAD]"}
 
# Serializing json
json_object = json.dumps(dictionary, indent=4)
 
# Writing to sample.json
with open(slang_model+"/special_tokens_map.json", "w") as outfile:
    outfile.write(json_object)

false=False
dictionary = {"unk_token": "[UNK]", 
              "bos_token": "<s>", 
              "eos_token": "</s>", 
              "pad_token": "[PAD]", 
              "do_lower_case": false, 
              "word_delimiter_token": "|", 
              "replace_word_delimiter_char": " ", 
              "tokenizer_class": "Wav2Vec2CTCTokenizer"}
json_object = json.dumps(dictionary, indent=4)
with open(slang_model+"/tokenizer_config.json", "w") as outfile:
    outfile.write(json_object)

dictionary = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "[UNK]", "pad_token": "[PAD]"}
json_object = json.dumps(dictionary, indent=4)
with open(slang_model+"/special_tokens_map.json", "w") as outfile:
    outfile.write(json_object)

import shutil

original = r'vocab.json'
target = r''+ slang_model + '/vocab.json'

shutil.copyfile(original, target)

trainer.save_model(slang_model) # 모델 저장

print("Training Complete")
