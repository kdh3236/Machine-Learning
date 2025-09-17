## Embedding

**One-hot vector**

- cat: [0 0 0 1 0]과 같이 특정 단어를 하나의 Element만 1인 Vector로 변환하는 것

**Bag of Words**: 문장을 이루는 각 단어들을 **One-hot vector**로 만들고 각 Vector들을 더한 것

**Embedding Matrix**: Neural Network에서 자주 사용되는 방법

- 특정 단어를 나타내는 **One-hot vector**가 입력으로 들어왔을 때, **Embedding matrix**와 Matrix multiplication을 진행하면 결과로 나오는 벡터는 해당 단어를 **Embedding**한 결과이다.
- **Embedding Matrix**에서 **각 행은 각 단어를 Embedding**한 것이다.
- **Bag of Words**가 입력이라면, 결과는 문장을 이루는 **각 단어들의 Embedding Vector의 합**이다.
- 한 문장을 표현할 때, **Dimension이 감소**하게 된다.

**Embedding Bag**: 문장을 구성하는 각 단어들의 Embedding vector의 합 등으로 **문장을 하나의 Vector로 표현하는 방법**

**Embedding in PyTorch (Using Neural Network)**

1. 먼저 Vocabulary를 만든다.

2. 이후, Vocabulary에서 Unique한 문자의 개수를 센다.

3. **((2)에서 센 문자의 개수, Embedding dimension) 크기**의 Embedding Matrix를 만든다.

4. 각 문장을 구성하는 Token을 Vocabulary를 거쳐 **One-hot vector (Bag of Words**)로 만들고, **Embedding Matrix**와 연산하면 **Embedding bag**이 만들어진다.

    - Matrix Multiplication: Logits 형성, Activation function, 
  
위 방법대로 Neural Network를 구성하면, **Text Classifier**를 구현할 수 있다.

**Offset Parameter**: 여러개의 문장을 입력받아야 할 때, 각 문장을 구별하는 의미로 추가하는 값

- `.insert(0, 0)` 등, 0을 삽입하거나 `offsets=[문장 시작 위치 1, 문장 시작 위치 2]`와 같이 위치를 알려주는 방식 등으로 구현한다.

**위 과정 전체를 구현하는 방법**

먼저, Vocab을 생성하고, **Collate funciton**을 정의한다.

- **Collate function**에서 tensor로 만들고 device를 지정하며, **Labeling과 Offset**을 지정해준다.

이후, **Model class**를 정의한다.

```python
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        # nn.EmbeddingBag: Embedding Matrix를 구현
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        # Uniform Distribution으로 초기화
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


model = TextClassificationModel(vocab_size, emsize, num_class)
# Forward
model(text, offsets)
```

**text_pipeline**: 텍스트 데이터를 딥러닝 모델이 이해할 수 있는 숫자 형태의 시퀀스로 변환하는 일련의 과정

- tokenizer를 적용한다.

이제 Data를 분류하기 위한 Label을 설정하고, loss, optimizer 등을 세팅한다.

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
```


## N-grams

다음 **Text를 예측하기 위해 이전 `N`개의 Text를 참고**한다.

- 조건부 확률로 나타낼 수 있다. $P(w_{n+1} | w_{n}, w_{n-1}, ..., w_{1})$

- **Bigram**: $P(W_n|W_{n-1})$
- **Trigram**: $P(W_n | W_{n-1}, W_{n-2})$

- 이때의 `N`은 **Context length**이다.

`Neural Network`에서는 직전 Word를 직접 사용하는 대신 **Neural Network Approximation**을 이용한다.

- Context lenght만큼의 **Words를 Embedding**하고, **Concatenate** (Sum 등 다른 방법도 존재)하여 **하나의 Context Vector를 생성**하고 그것을 Input으로 받는다.

**N-grams in PyTorch**

1. Neural Network 내에 Embedding Layer를 추가한다.

    - `nn.Embedding`을 이용한다.

2. Collate function에 Slicing을 통해 Context vector를 생성하고, Training을 위한 Target vector를 생성하는 과정을 추가한다.

**Index to word decoder**

- **vocab.get_itos()**: 정수 ID 순서대로 매핑되는 문자열을 List로 반환한다.
- Prediction을 한 결과를 문자열로 확인하고 싶을 때 사용한다.

**nltk를 이용하여 N-grams를 다루는 방법**
  
```python
# nltk로 Tokenize한 Token 집합에서 각 Token의 개수를 셀 수 있다.
# FreqDist를 이용하면 각 Token이 Key, 나온 횟수가 Value가 된다.
fdist = nltk.FreqDist(tokens)

# tokens 순서대로 앞 뒤 두개의 Token을 Tuple로 묶음
bigrams = nltk.bigrams(tokens)
# bigrams로 두 개로 묶인 Tuple의 개수를 Count
nltk.FreqDist(bigrams)

# trigrams
nltk.trigrams(tokens)
```

이후,  `nltk.bigrams(tokens)`에서의 **전체 Count를 계산**해서 Vocabulary에서 각 word를 꺼내와서 input text에 추가한 다음, **input+word tuple**이 `nltk.FreqDist(bigrams)`의 key로 존재하면 **전체 Count로 나누어 조건부 확률을 계산**한다.

- 이를 통해 Prediction 할 수 있다.

- $$P(W_{\text{next}} | W_{\text{context}}) = \frac{\text{Count}(W_{\text{context}}, W_{\text{next}})}{\text{Count}(W_{\text{context}})}$$

```python
vocab_probabilities = {}  # Initialize a dictionary to store predicted word probabilities
context_size = len(list(freq_grams.keys())[0])  # Determine the context size from n-grams keys

# Preprocess input words and take only the relevant context words
my_tokens = preprocess(my_words)[0:context_size - 1]

# Calculate probabilities for each word in the vocabulary given the context
for next_word in vocabulary:
    temp = my_tokens.copy()
    # input에 vocabulary에서 뽑아온 한 단어를 추가
    temp.append(next_word)  # Add the next word to the context

    # Calculate the conditional probability using the frequency information
    # freq_grams[tuple(temp)]: input+word의 등장 횟수
    if normlize!=0:
        vocab_probabilities[next_word] = freq_grams[tuple(temp)] / normlize
    else:
        vocab_probabilities[next_word] = freq_grams[tuple(temp)] 

# 확률이 높은 순서대로 정렬 후 첫 번째 원소를 반환
vocab_probabilities = sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)
```

위에선 `nltk`를 이용하여 **N-grams**를 구현했다.

만약, bigram, trigram이 아니라면 임의로 N-grams을 생성하는 함수를 아래와 같이 작성할 수 있다.

```python
ngrams = [
    (
        [tokens[i - j - 1] for j in range(CONTEXT_SIZE)],  # Context words (previous words)
        tokens[i]  # Target word (the word to predict)
    ) for i in range(CONTEXT_SIZE, len(tokens))
]
```

이후, 생성한 ngrams을 입력받아서 Embedding한 결과를 Flatten한 이후, Linear layer의 입력으로 사용할 수 있다.

- Flatten 하는 이유는 N-grams의 경우. N개의 단어에 대한 각 ID가 List로 반환되기 때문에 이를 **한 Vector로 다루기 위함**이다.

```python
context, target=ngrams[0]
linear = nn.Linear(embedding_dim*CONTEXT_SIZE,128)
linear(context)

# Embedding
my_embeddings=embeddings(torch.tensor(vocab(context)))
# Flatten
my_embeddings=my_embeddings.reshape(1,-1)
linear(my_embeddings)
```
