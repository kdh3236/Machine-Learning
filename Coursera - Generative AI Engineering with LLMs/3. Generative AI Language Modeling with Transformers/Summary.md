# Positional Encoding

**문장을 구성하는 단어의 순서**는 문장의 의미에 중요한 영향을 끼친다.

- Transformer 같은 모델에서 **입력 문장의 순서 정보도 보존**하기 위해서 **Positonal Vector를** 사용한다.

Learnable Parameter가 아니라, **고정된 Positional Vector**로는 **Sin, Cos**을 사용한다.

**Sin, Cos**의 식은 아래와 같다.

- $PE[pos,2i]$ = $sin(\frac{pos​}{10000^{\frac{2i}{d_{model​}}​}})$
  
- $PE[pos,2i+1]$ = $cos(\frac{pos​}{10000^{\frac{2i}{d_{model​}}​}})$

- $pos$ = 각 단어의 위치 (0, 1,..., N)

- $i$ = Encoding Dimension이 m이라고 하면 **0 ~ m까지의 Dimension index**

- $d_{model}$ = Word Embedding Dimension

- **0~m-1** 까지의 index를 같는 Array의 각 Element를 $PE$ 식에 맞춰서 채워넣는다.

**Leanable Parameter**로 Neural Network가 **Positional Encoding Vector 자체를 학습**하도록 구현할 수도 있다.

두 방법 중 하나로 구한 **Positional Vector**를 **문장을 Tokenizaion한 결과에서의 각 단어와 해당 위치의 Positional vector를 더하여** Input으로 사용한다.

# Segment Embedding

특정 단어가 **어떤 문맥, 화자, 타입** 등에 속하는 지를 나타낸 Vector

- **BERT** 등에서 주로 사용된다.

- Segment Embedding까지 사용하면 **Token + Positional Vector + Segment Vector**로 구성된다.

# Attention Mechanism

> 문장 중 **특정 한 부분에 집중**하는 방식으로 문장을 해석하는 방법

**Query, Key, Value**로 이루어져 있다.

- **Query**: 지금 원하는 정보
- **Key**: 검색에 도움이 되는 정보
- **Value**: 실제 찾고자 하는 정보

중요한 점은 **Key와 Value**에서 **같은 행은 같은 Token (Data)에 대한 정보**를 나타내야 한다는 점이다.

- Self-Attention이던 Cross-Attention이던 **Key, Value는 같은 X로부터 얻는 이유**이다.

- Query가 달라도, Key를 통해 Value에서 연관된 정보를 얻을 수 있도록 하기 위함이다.

**Query와 Key의 Inner Product**를 통해 원하는 정보를 탐색한다.

이후, **Softmax**를 사용하는데, 이는 **값의 범위를 제한**하기 위함이다.

Word Embedding 관점에서 기존에 학습하지 못 하던, **문맥적인 단어 간의 관계도 학습**할 수 있다.

Sequence Data를 처리할 때, 한 Token씩 따로 처리하지 않고, **하나의 matrix로 묶어서 처리**한다.

- Query, Key, Value 모두 동일한 방법을 사용한다.

## Self-Attention Mechanism

**Scaled dot-product 방식으로 설명한다.**

Input X를 **Tokenization + Embedding**한 이후에, **Query, Key, Value Matrix**와 곱하여 **Key, Query, Value**를 얻는다.

- **Query, Key, Value Matrix**는 Learnable Parameter이며, Bias term도 사용하기도 한다.

이후, **Query와 Key를 Inner Product**하고 **첫 번째 Dimension의 제곱근**으로 나눈다.

- 나누는 이유는 Variance가 너무 커져 Score 값이 너무 커지는 것을 방지하기 위함이다.

- 결과로 나오는 행렬에 **Softmax Function**을 적용하고 이 행렬을 $A$라고 한다.

$A$를 Value와 Matrix Multiplication한 결과로 나오는 Matrix가 **Context Matrix**이다.

- Attention Score ($A$)를 봤을 때, Value에서 **특정 단어마다 집중할 위치가 표시**되기 때문이다.

이후, Output을 생성하기 위한 **Output Matrix**와 곱하고 **평균**내어 **MLP**에 넘겨주어 Classiciation이나 Prediction 등을 수행할 수도 있다.

## Multi-head Attention

각 **Head**는 **독립적인 Parameter를 사용하는 Self-Attention Module로** 이루어져 있고, **같은 Input(X)를** 사용하여 병렬적으로 처리하는 방법

결과로 나온 Head 개수만큼의 $H'$ Matrix를 전부 **Concatenate**하고, **Output Matrix를 통해 Output을 생성**한다. 

**Multi-head Attention**은 하나의 문장을 **병렬적으로 여러 관점에서 처리**할 수 있도록 하여 **모델의 표현력을 높인다.**

# Transformer

> **Self-Attnetion Module을 기본으로 사용하며, Encoder와 Decoder로 구성된다.**

먼저, **Encoder**부터 살펴보자.

1. Embedding된 Input에 Positional Vector를 더한다.
2. **Multi-head Self-Attention Module**에 (1)의 결과를 넘겨준다.

    - 일반적으로, 여기선 Masking을 사용하지 않는다.
    
3. Add and Normalization
4. FC Layer
5. Add and Normalization

**Decoder**는 아래와 같이 구성된다.

1. Decoder Input을 Embedding하고 Positional Vector를 더한다.
2. **Masked Multi-haed Self-Attention module**을 사용한다.

      - Decoder에서는 특정 단어 기준 미래 단어를 미리 알 수 있으면 안 된다.
  
3. Add and Normalization
4. Encoder의 Output을 Query로 이용하고, (2)에서의 출력을 Key, Value로 사용한 **Multi-head Cross-Attention Module**
5. Add and Normalization
6. FC Layer
7. Add and Normalization
8. Linear Layer
9. Softmax -> Output Probabilities

**Translation**의 경우, Encoder Input: Source Langauge / Decoder Input: Target Language로 구성된다.

일반적으로 **LLM**에서는 Encoder 입력 = 질문(소스) + (선택) 정보 컨텍스트 / Decoder 입력 = 정답의 앞부분(BOS/이전 생성 토큰) 으로 구성된다.

**Transformer Block**을 **여러 Layer로 구성**해서 더 복잡한 Sequence Data를 처리할 수도 있다.

- 이 경우 **Layer의 개수도 Hyperparameter**이다.

Multi-haed의 **Head 개수도 중요한 Hyperparameter**이다.

# Implementation

Attention 개념을 이용한 **Translation model**을 생성하는 과정은 아래와 같다.

1. 변역할 언어와 변역 결과로 나올 언어의 Vocabulary를 각각 구성한다.
2. 각 단어를 Tokenization하고 Vocabulary를 이용하여 One-hot Vector로 만든다.
3. Key = 변역할 언어로 된 Vocabulary를 구성하는 각 단어들의 One-hot Vector를 Stack
4. Value = 변역 결과 언로 된 Vocabulary를 구성하는 각 단어들의 One-hot Vector를 Stack
5. Query = 변역할 문장을 One-hot vector로 만들고 Stack
6. Softmax(Query @ Key) # Value

**Attention Head Class**는 아래와 같이 정의할 수 있다.

```python
class Head(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the superclass (nn.Module)
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

    def attention(self, x):
        embedded_x = self.embedding(x)
        k = self.key(embedded_x)
        q = self.query(embedded_x)
        v = self.value(embedded_x)
        w = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # Query * Keys / normalization
        w = F.softmax(w, dim=-1)  # Do a softmax across the last dimesion
        return embedded_x,k,q,v,w
    
    def forward(self, x):
        embedded_x = self.embedding(x)
        k = self.key(embedded_x)
        q = self.query(embedded_x)
        v = self.value(embedded_x)
        w = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # Query * Keys / normalization
        w = F.softmax(w, dim=-1)  # Do a softmax across the last dimesion
        out = w @ v
        return out
```

___

**Positional Embedding**은 다양한 방법으로 구현할 수 있다.

1. **0 ~ N까지의 Index (위치)를 그대로 더하는 방법**

    - Positional Embedding으로 인하여 더하는 값을 큰 값을 사용하면 **Positional Vector의 영향이 커져 학습이 제대로 이루어지지 않는다.**
  
2. Index를 더하는 대신, **각 Positional Dimension마다 계수**를 사용한다. (ex 0.1)

위의 두 방법 모두 **선형적인** 방법이며, 이 방법을 이용하면 문장에서 **뒤에 위치한 단어일수록 선형적으로 크기가 증폭**된다.

이를 방지하기 위해, **주기함수**를 사용한다.

- **Sin, Cos** 함수를 이용한다.

___

**Transformer**는 `PyTorch Module`을 이용하면 구현하기 쉽다.

```python
nn.Transformer(nhead=16, num_encoder_layers=12)

# Training 시에는 tgt를 이렇게 사용
# Inference시에는 <BOS> Token이나 정답 출력의 앞부분만 넘겨주는 경우가 많음
src = torch.rand((문장개수, 한줄의길이, 각단어의EebeddingDimension))
tgt = torch.rand((문장개수, 한줄의길이, 각단어의EebeddingDimension))

out = transformer_model(src, tgt)
```

**Multi-head** 역시 `PyTorch Module`을 통해 쉽게 구현할 수 있다.

```python
multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first=False)

query = torch.rand((seq_length, batch_size, embed_dim))
key = torch.rand((seq_length, batch_size, embed_dim))
value = torch.rand((seq_length, batch_size, embed_dim))

attn_output, attn_weights = multihead_attn(query, key, value)
```

**Transformer Encoder**도 쉽게 구현할 수 있다.

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    dim_feedforward=1, # Encoder Block 내의 Feadforward Net의 Hidden Dimension을 정
    dropout=0
)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

x = torch.rand((seq_length, batch_size, embed_dim))
encoded = transformer_encoder(x)
```

# Transformers for Classification: Encoder

**Transformer Encoder**의 출력을 **FC Layer**를 통과하도록 하면, 결과물에 **argmax**를 사용해 Classification을 수행할 수 있다.

- 이 과정에서 Standardization을 위한 **Zero Padding**, Tokenizer, Vocab 등이 요구된다.


**Zero Padding**은 `torch.nn.utils.rnn`의 `pad_sequence()` 함수를 통해 쉽게 구현할 수 있다.

```python
# sequences 내에 있는 최대 길이 문장에 맞추어 다른 문장을 0으로 채운다.
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
```

`.state_dict()`: `torch.nn.Module`을 상속 받아 정의된 **모델 Class의 내부 구조 (Weight, Bachnorm, Dropout ...) Tensor**를 불러오는 함수

# Decoder-only Model

**GPT, LLaMa** 등의 기본 모델으로, Input에 맞는 **답변을 예측하여 생성**한다.

예측 및 생성 과정은 아래와 같다.

1. Input token
2. (Masked Self-Attention + Add & Norm + Feedfoward Layer + Add & Norm) x L
3. 각 Token마다 Vocab size의 vector로 만든다. (Linear Layer 이용)
4. Input의 마지막 Token에 대응되는 (3)의 Vector에서만 argmax를 찾아 해당 위치를 Index로 하여 단어를 찾는다.
5. (4)의 Token이 <EOS>가 아니라면, 기존 Input Token에 (4)의 Token을 추가하여 (1) ~ (4) 과정을 반복한다.

**Auto-regressive**: Masked self-Attention을 사용하여 이전 Token만을 이용해 다음 Token을 예측하는 것

**Decoder-only Model**을 Training할 때 사용할 수 있는 방법

1. **Fine-tuning**
2. **RLHF**

    - 사람의 피드백을 통한 강화학습
  
**Masked Self-Attention**

- **Attention Score Matrix**의 **우측 Upper Triangle**을 **-inf**로 설정한다.
- Softmax를 취하면 **-inf**는 0이 되기 때문이다.
- **Causal Attention**이라고도 부른다.

PyTorch를 이용하여 아래와 같이 구현할 수 있다.

```python
sz = src.shape[0]

def generate_square_subsequent_mask(sz,device=DEVICE):
    # torch.triu(torch.ones()): 우상단 삼각형이 1인 행렬 생성
    # == 1: 1인 부분을 True, 아닌 부분을 False
    # .transpose(0, 1): 좌하단 삼각형이 True, 그 외 false
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    # 0인 부분: -inf, 1인 부분 0.0으로 설정
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```

**Decoder-only model**을 **Training**하는 과정은 아래와 같다.

1. Input Token에 맞추어 Decoder-only model을 통과시킨다.
2. 결과로 나온 각 Token의 Vocab size의 logits vector를 이용하여 Cross Entropy Loss를 계산한다.
3. (2)에서 각 Token에 대해 구한 Loss를 평균낸다.

**Special Token**: 문장 시작 / 끝, Pad 등을 나타내기 위해 사용하는 Token

```python
UNK_IDX, PAD_IDX, EOS_IDX = 0, 1, 2
special_symbols = ['<unk>', '<pad>', '<|endoftext|>' ]

# special_first를 통해 Special token을 vocab에 먼저 지정한다.
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=special_symbols, special_first=True)
```

Padding을 한 부분은 **Model의 Output을 생성하는 과정에서 필요가 없으므로**, **Padding mask**도 필요하다.

- 일반적으로 **Padding Mask**와 **ignore_index**를 둘 다 사용한다.

  
```python
src_padding_mask = (src == PAD_IDX).transpose(0, 1)

# ignore_index를 통해 PAD Token을 무시한다.
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

**Decoder**는 `torch.nn.TransformerEncoder()`를 **동일하게 사용하고, mask**만 넘겨주면 된다.

```python
encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

# Forward
output = transformer_encoder(x, src_mask)
```

# Encoder-only model

**bert**이 대표적인 **Bidirectional model**

- 특정 단어를 예측할 때, Decoder-only model처럼 이전 단어만 확인하는 것이 아니라, **양방향 단어를 전부 활용**하여 예측

한 단어를 예측하기 위해 양방향의 단어들을 모두 활용하기 때문에, 예측 성능이 좋다.

- **Training (Pre-training)** 과정에서 예측할 단어를 **[MASK] Token**으로 두고, **[MASK] Token 위치에 들어갈 단어를 예측**한다.
- **Inference**시에는 Input에서 [MASK] Token에 대응되는 위치의 Logits에서 argmax를 사용한다.

하지만, Fine-tuning의 경우에는 **[MASK] Token이 직접적으로 등장**하지 않아 문제가 생긴다.

- 이 문제를 해결하기 위해 **Pre-training** 과정에서 **미리 Masked Language Modeling 규칙**을 사용한다.

- **Fine-tuning과 Pre-training 간의 괴리**를 줄이기 위해, **Training에서 미리 적용**한다. 

**Masked Language Modeling (MLM)**

- 전체 단어의 **85%는** 그대로 놔두고, 나머지 **15%를** Mask 대상으로 정한다.

    - 이 과정에서 두 경우 중 하나이므로 **Bernoulli Distribution**을 주로 사용한다.
       
- **15% 중, 80%는 [MASK] Token으로 대체하고, 10%는 Vocab내에서 임의의 Token으로 교체하고, 10%는 그대로 놔둔다.**
- 이때, **85%에 해당하는 Label을 [PAD]로 교체**해 **Loss 계산에 영향을 미치지 않도록 한다.**

**Next Sequence Prediction (NSP)**

**문장간의 연속성, 관련성**을 학습하기 위한 Task

두 문장이 있을 때, **첫 번째 문장 뒤에 두 번째 문장이 위치하는 것이 자연스러운지, 아닌지를 판단**한다.

먼저 **문장의 시작, 끝**을 구별하기 위해 **[CLS], [SEP] Token**을 사용한다.

- 두 문장은 **[CLS] ... [SEP] ... [SEP]로** 구성된다.

Positional Embedding에 추가로, 각 Token이 **두 문장 중 어떤 문장에 위치하는지**를 나타내기 위해 **Segment Embedding**을 사용한다.

- **첫 문장이라면 1, 두 번째 문장이라면 2** 같은 식으로 Data가 준비된다.
- **Special Token 정보까지 추가하려면 0 ~ 2의 3개의 수를 사용한다.**
- 이후, `nn.Embedding(2, embed_d)`를 통과하여 Embedding vector로 만든다.

**Encoder+Head Output**은 **두 번째 문장이 첫 번째 문장 뒤에 오는 것이 적합하다면 1, 적합하지 않다면 0**으로 나오게 된다.

Encoder를 통과하는 과정은 아래와 같다.

1. Input + Positional Embedding + Segment Embedding
2. Self-Attention Layer (Masked Self-Attionion 아님 / Padding mask는 그대로 사용)
3. **[CLS] Token 위치의 Hidden Vector**를 Head를 통과시켜 적합한지 아닌지의 결과를 얻는다.

    - [CLS] Token은 문장의 시작도 나타내고, **문장 전체 정보도 담도록 구현**된다.
    - 결과는 뒷 문장이 자연스러우면 1, 그렇지 않다면 0으로 나타난다.
    - Training 시에는 위 **2-class logits**을 이용하여 CrossEntropyLoss를 계산한다.
  
**BERT**는 **MLM과 NSP** 두 가지 Task를 모두 이용하여 **Pre-training**하고, 두 개의 Loss를 합하여 하나의 Loss로 사용한다.

- MLM만으로는 부족한 점이 있어 NSP까지 추가로 사용한다.

## Implementation

`transformers.BertTokenizer`를 통해 **Tokenization**할 수 있다.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# encode_plus(): 두 문장을 하나로 Encoding해준다.
# NSP를 위한 Data 준비 / # MLM은 그냥 tokenizer()만 수행하면 된다.
encoded_input = tokenizer.encode_plus(
    original_text,
    add_special_tokens=True,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)

from transformers import BertForPreTraining, BertTokenizer
import torch

# 아래와 같이 입력하면, 두 문장을 쌍으로 묶어
# BeRT model에 바로 넣을 수 있는 Dictionary 형태로 반환한다.
input = tokenizer(text1, text2, return_tensors="pt")
target_is_next = torch.tensor([]) # 모델에서 Loss 계산시 사용할 NSP Target
output = model(**input, next_sentence_label =target_is_next))

# NSP Ouput [Batch size, 2]
output.seq_relationship_logits

# MLM Output [Batch size, Sequence Length, Vocab size]
input = tokenizer(text, return_tensors="pt")
# input_ids: Text를 정수 ID로 매핑한 부분
output = model(input_ids = input[input_ids])
output = output.prediction_logits

# mask_token_id = [MASK] Token id를 반환
mask_positions = (input_ids[0] == tokenizer.mask_token_id)
# output에서 MASK 위치한 부분의 argmax를 찾음
index = torch.argmax(output[0, input_ids[0] == tokenizer.mask_token_id]).item()
# 정수 id를 token으로 바꿔줌
prediction = tokenizer.convert_ids_to_tokens([index])
```

위와 같이 **Pre-trained**된 **Tokenizer**를 사용하면, `Tokenizer.get_vocab()`을 통해 **미리 학습된 Vocabulary**를 얻을 수 있다.

- `len(tokenizer.vocab)`은 Vocabulary size를 return한다.

`nn.TrasformerEncoderLayer`를 사용해서 **Encoder를 구현**한다.

```
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dropout=dropout,batch_first=False)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
# src_key_padding_mask: padding mask만 사용한다.
transformer_encoder_output = transformer_encoder(bert_embeddings,src_key_padding_mask=padding_mask)
```
