## Natural Language Processing (NLP)

**Natural Language Processing (NLP)** 분야를 학습하기 위해 필요한 **Library**에 대해 알아보자.

먼저 **Huggingface**부터 살펴보자.

- **transformers**
- **AutoTokenizer**
- **AutoModelForSeq2SeqLM**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model 선택
model_name = "facebook/blenderbot-400M-distill"

# Model_name에 맞는 Seq2Seq Model을 찾아 반환한다.
# 이후, model.generate()로 응답을 생성할 수 있다.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Model에 맞는 Tokenizer를 찾아 반환
tokenizer = AutoTokenizer.from_pretrained(model_name)

### Tokenizer와 model을 이용하여 답변을 생성하는 과정은 아래와 같다.
input_text = input()
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
```

## Tokenization: 문장을 작은 조각으로 나누는 과정

- Tokenoizer는 Tokenization을 위한 Method이다.

Tokenization 방식에는 여러 방법이 있다.

1. **한 단어를 한 Token에 Mapping**

   - 모델이 의미를 정확히 이해할 수 있지만, Vocabulary의 크기가 커진다.

2. **한 글자를 한 Token에 Mapping**

    - Vocabulary의 크기가 작아지지만 모델이 의미를 정확히 이해하기 어렵다.
    - 여러 입력이 들어가야 하기 때문에 Input dimension이 커진다.
    - 비슷한 의미의 단어임에도 불구하고 (ex. 단수/복수) 완전히 다른 Token으로 매핑할 수 있다.
  
3. **Subword-based tokenization**

     - 자주 사용되는 단어는 한 Token으로, 그렇지 않은 단어는 단어의 일부가 하나의 Token이 된다.
  
**Tokenization and indexing in PyTorch**

- `build_vacab_from_iterator` 함수를 이용해 Vocabulary를 형성한다.
- 각 Token마다 정수 Index를 부여한다.

```python
# Word 단위의 Tokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# dataset는 array of tuples라고 가정하자.

tokenizer = get_tokenizer("basic_english")
tokenizer(dataset[0][1])

def yield_tokens(data_iter):
  for _, text in data_iter:
    yield tokenizer(text)

my_iterator = yield_tokens(dataset)

next(my_iterator)

# build_vocab_from_iterator: Token과 정수 ID를 매핑시킴.
# specials=['<unk>']: <unk>도 Vocabulary에 포함시킨다.
vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
# unk token: Vocab에 없어 모르는 단어가 들어왔을 때, Default로 unk에 매핑되는 정수 ID를 사용하도록 함 
vocab.set_default_index(vocab["<unk>"])
# Key: 문자열, Value: 정수 ID인 Dictionary로 만듦.
vocab.get_stoi()
```

각 문장마다 **Special token**을 추가할 수 있다.

- **<bos>**: Begin Of Sequence로 문장의 시작을 의미하는 Token.
- **<eos>**: End Of Sequence로 문장의 시작을 의미하는 Token.
- **<pad>**: 문장의 길이를 일정하게 맞춰주기 위해 추가하는 Token

```python
tokenized_line = ['<bos>'] + tokenized_line + ['<eos>']

max_length = max(max_length, len(tokenized_line))

for i in range(len(tokens)):
  tokens[i] = tokens[i] + ['<pad>'] * (max_length) - len(tokens[i]))
```

**Tokenization**을 지원하는 여러 Library가 존재한다.

먼저 **word-based tokenizer algorithm**에 대해 알아보자.

**nltk**: Natural Language Toolkit

```python
from nltk.tokenize import word_tokenize

text = "This is a sample sentence for word tokenization."
tokens = word_tokenize(text)
```

**spaCy**: Open Source of for natural language processing

```python
import spacy

text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
# Language Model을 load
nlp = spacy.load("en_core_web_sm")
# doc object: tokenizer, tag 등을 담는 객체 생성
doc = nlp(text)

# doc 내의 token을 다루는 방법
token_list = [token.text for token in doc]
```

**BERT and XLNET**

```python
from transformers import BertTokenizer
from transformers import XLNetTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.tokenize("IBM taught me tokenization.")

tokenizer = XLNetTokenizer.from_pretrained("bert-base-uncased")
tokenizer.tokenize("IBM taught me tokenization.")
```

___

## Dataloader

Data를 전처리하고 미리 준비할 수 있도록 도와주는 모듈

-  Batching and Shuffle data
-  Data augmetation
-  Memory utilization

**NLP** 분야에선 Data를 **변형하거나 전처리**하는데 많이 사용된다.

나만의 데이터를 사용하기 위해서 **CustomDataset**을 구현할 수도 있다.

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 반드시 아래 구조를 따라야 한다. 
class CustomDataest(Dataset):
    # Data를 load하고 읽음
    def __init__(self, data):
        self.data = data

    # Data 길이 반환
    def __len__(self):
        return len(self.data)

    # 실제로 Item을 반환
    def __getitem__(self, idx):
        return self.data[idx]

custom_dataset = CustomDataset()

batch_size = 2
# Batching과 Shuffle이 쉽도록 해준다.
# 데이터를 무작위로 섞고 하나의 Batch 안에 Batch_size만큼의 데이터가 존재하도록 한다.
# Dataloader는 Iterator로 반환되며, 하나의 반환값으로 한 Batch가 반환된다. 
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    print(batch)
```

Input data가 Text라고 가정하면, 한 문장마다 길이가 다를 수 있다.

- 각 문장의 길이를 동일하게 만드는 것이 **학습 속도와 병렬화 측면**에서 좋다.

`DataLoader`를 통해 가져온 Data에 **padding**을 하여 같은 길이의 문장으로 만드는 것이 좋다.

```python
from torch.nn.utils.rnn import pad_sequence

# Batch 내에서 가장 긴 문장을 기준으로 0 값을 채워넣음
# batch_first=True: batch-dimension이 항상 첫 번째 차원이 되도록 강
padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
```

**DataLoader**는 data를 batch size만큼 묶어서 `collate_fn` 형태로 넘긴다.

- DataLoader에 collate_fn 함수를 따로 넘겨주지 않으면 그대로 Batch가 생성되게 된다.

데이터가 Batch size로 묶임에 따라 **추가적인 처리**가 필요할 경우, **collate_fn을 따로 정의**하여 `collate_fn=` Argument에 넘겨주는 것이 좋다.

- 특히, input이 sequence와 같이 **길이가 정해져 있지 않은 경우**에, 길이를 맞춰주는 과정은 collate_fn을 통해 하는 것이 좋다.
- 이미지 크기가 다르거나, 토큰화/마스크/라벨 가공이 필요한 경우에 요구된다.

정확한 과정은 아래와 같다.

- `Dataset.__getitem__`은 한 개 샘플을 만든다.

- `DataLoader`는 이런 샘플 B개를 모아 **collate_fn(samples: list)에** 넘긴다.

- `collate_fn`은 이 리스트를 **스택/패딩/마스킹/형변환** 등으로 정리해서 배치 딕셔너리/튜플을 리턴한다.

```python
def collate_fn(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)
```

## 데이터 품질과 다양성

Training을 통해 좋은 결과를 내기 위해선 여러 가지 작업이 필요하다.

- **노이즈 감소**: 관련이 없거나 반복적인 데이터를 제거
- **일관성 검사**: 상충되거나 오래된 정보를 제거하기 위해 주기적으로 체크
- **라벨링 품질**: 정확한 라벨링 요구

추가적으로 **문화적, 지역적 등으로 다양한 데이터**를 통해 모델의 포용성을 향상시킬 수 있어야 한다.
