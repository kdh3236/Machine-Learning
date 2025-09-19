# Embedding

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

이후, 생성한 **ngrams을 입력받아서 Embedding한 결과**를 **Flatten한 이후, Linear layer의 입력**으로 사용할 수 있다.

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

___

# Word2Vec

> **단어를 벡터로 Embedding하는 모델**

Neural Network에서 특정 단어를 Embedding하면 Vector Space에서의 Vector로 나타낼 수 있다.

- 단어들간의 벡터 덧셈 / 뺄셈을 통해 다른 단어를 나타낼 수도 있다.
- 유사한 단어는 Vector Space에서 근처에 위치한다.
- 두 Vector의 유사도는 **Inner Product**를 통해 구할 수 있다.

**Word2Vec**에서 `t` (target index)를 설정하면, Context vector는 $W_{t-1}, W_{t+1}$이 된다.

- 위 경우는 **Context Window**가 2인 경우이다.

## Continous Bag of Words (CBOW) Model

- **Word2Vec Model** 중 하나.
- 이 모델은 **Context vector**만을 Nueral Network의 Input으로 넘겨, Target word를 예측하도록 한다.
- **Context vector**는 $W_{t-1}, W_{t+1}$의 위치만 1로 만든 vector를 생성하고, $W_t$를 예측하도록 한다.

 **CBOW Model**을 구현해보자.

 먼저 **Training**을 위한 데이터는 아래와 같이 구성한다.

 - Target word가 될 수 있는 모든 Word에 대해 (Context, Target) 쌍을 구성한다.

```python
CONTEXT_SIZE = 2


cobow_data = []

# modified code

for i in range(CONTEXT_SIZE, len(tokenized_toy_data ) - CONTEXT_SIZE):

    context = (

        [tokenized_toy_data [i - CONTEXT_SIZE + j] for j in range(CONTEXT_SIZE)]

        + [tokenized_toy_data [i + j + 1] for j in range(CONTEXT_SIZE)]

    )

    target = tokenized_toy_data [i]

    cobow_data.append((context, target))
```

이후, **collate function**은 Context, Target을 각각 Vocab을 통해 정수 ID로 매핑하고 Offset을 설정한다.

**CBOW Model** class는 아래와 같이 정의한다.

```python
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.linear1 = nn.Linear(embed_dim, embed_dim//2)
        self.fc = nn.Linear(embed_dim//2, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offsets):
        out = self.embedding(text, offsets)
        out = torch.relu(self.linear1(out))
        return self.fc(out)
```

## Skip-Gram Model

- **Word2Vec Model** 중 하나
- Context Vector로 부터 Target Vector를 예측하는 것이 아니라, **Target Vector를 이용하여 Context Vector을 예측**하도록 하는 방식
- 위 방법대로 Training하면, **Target Vector가 자주 등장하는 Context Vector까지 학습하여 Vector로 Embedding**이 된다.

**Context Window = 2**인 예시를 보자.

- Target word = $w_t$라면, context vector는 ($w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$)가 된다.
- 이때 모델이 **P($w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$ | $w_t$)의 확률**이 최대가 되도록 할 수도 있고, **P($w_i$ | $w_t$)의 확률**이 최대가 되도록 할 수도 있다.

**CBOW Model**에서와 동일한 방식으로 (context, target) 쌍을 구성한 이후 아래와 같이 **Skip Gram Model**을 위한 데이터를 만들 수 있다.

- 여기선 P($w_i$ | $w_t$)의 확률이 최대가 되도록 구현한다.

- 이를 위해 **(context word, target word) 쌍을 구성**한다.

```python
skip_data_=[[(sample[0],word) for word in  sample[1]] for sample in skip_data]
```


**Skip-Gram Model** Class는 아래와 같이 정의할 수 있다.

```python
class SkipGram_Model(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.fc = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, text):
        out = self.embeddings(text)
        out = torch.relu(out)
        out = self.fc(out)
        
        return out
```

## GloVe

> Stanford에서 제작한 Large-scale data for word embeddings

큰 데이터셋을 이용해 미리 학습한 이런 모델을 사용하면 **시간도 절약되고 풍부한 표현**을 얻을 수 있다.

먼저 **GloVE**를 Load한다.

```python
from torchtext.vocab import GloVe,vocab
glove_vectors_6B = GloVe(name ='6B')

# torch.nn.Embedding.from_pretrained를 통해 Pre-trained된 GloVe Layer Weight를 Load
embeddings_Glove6B = torch.nn.Embedding.from_pretrained(glove_vectors_6B.vectors,freeze=True)
```
**GloVe**를 통해 **단어를 Token ID로 매핑**하고 **Vocab**을 생성하는 작업은 다음과 같이 할 수 있다.

```python
glove_vectors_6B.stoi # Word -> ID 정보가 담긴 Dictionary 반환

embeddings_Glove6B.weight # GloVe에서 각 단어의 정수 ID에 해당되는 Vector를 저장
# embeddings_Glove6B.weight는 Embedding Matrix를 갖고 있고, Index에 대응되는 행을 반환한다.
# One-hot vector와의 Matrix Multiplication과 동일하다. 
embedding_vector = embeddings_Glove6B.weight[word_to_index[word]] # Word에 대응되는 Vector를 반환

# Vocabulary 생성
# 0: Vocab에 ㅗㅍ함할 단어의 Minimum Frequency
vocab = vocab(glove_vectors_6B.stoi, 0,specials=('<unk>', '<pad>'))
# Vocab에 없는 단어가 오면, vocab["<unk>"]의 index로 세팅하도록 한다. 
vocab.set_default_index(vocab["<unk>"])
```

## gensim

> **Word2Vec Model**을 쉽게 다룰 수 있도록 하는 Python Package

```python
from gensim.models import Word2Vec

# Vector-dimension, Context Window
# min_count 이상 등장힌 단어만 Vocab에 추가, workers=학습에 사용할 CPU Core 개수
w2v_model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, workers=4)

# sentences (Data)를 이용하여 Vocab 생성
w2v_model.build_vocab(sentences, progress_per=10000)

# total_examples: Training에 사용할 단어의 총 개수
# w2v_model.corpus_count: w2v_model의 Vocab에 포함된 문장의 총 개수
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

# Finding similar words
similar_words = w2v_model.wv.most_similar("pizza")
print("Similar words to 'pizza':", similar_words)

# Calculating word similarity
similarity = w2v_model.wv.similarity("pizza", "pasta")
print("Similarity between 'pizza' and 'pasta':", similarity)

# Extract word vectors and create word-to-index mapping
word_vectors = w2v_model.wv

# a dictionary to map words to their index in vocab
word_to_index = {word: index for index, word in enumerate(word_vectors.index_to_key)}

# Create an instance of nn.Embedding and load it with the trained vectors
embedding_dim = w2v_model.vector_size
embedding = torch.nn.Embedding(len(word_vectors.index_to_key), embedding_dim)
embedding.weight.data.copy_(torch.from_numpy(word_vectors.vectors))
```

# Seq2Seq Model

> Input을 처리하는 Encoder와 Output을 생성하는 Decoder가 결합된 형태의 Model

**Seq2Seq Model**을 살펴보기 전 Neural Network을 이용한 **Sequence Data의 처리**에 대해 알아보자.

- 문장을 통한 Translation이나 Classification을 수행해야 하는 경우에 대해 살펴보자.

RNN 이전의 Neural Network는 문장을 구성하는 각 단어들을 **One-hot vector**로 생성하고, 문장을 구성하는 모든 **One-hot vector**를 합하여 Input으로 사용하였다.

- 그러나 이 방법은 **문장 내에서 단어를 구성하는 순서가 달라져도 Network가 그 의미 차이를 인지할 수 없다.**

- RNN 이전의 Neural Network가 Input을 받아들이는 방식은 **iid (independently and identically distributed)에서 Sampling을 하는 것**과 같다.

하지만, **Sequence Data**에선 이전 단어가 나왔을 때의 기억을 바탕으로 **iid**가 아니라 **조건부 확률**처럼 동작해야한다.

- 이를 위해 기존 Neural Network가 아닌 **새로운 구조의 Neural Network**가 요구되었다.

**RNNs**: Model이 지금까지 기억하고 있는 **Hidden State**와 **현재의 Input**을 이용하여 결과를 생성한다.

- Short-term memory

**RNNS**이 Short-term memory이기 때문에, 이를 보완하기 위한 **GRUs, LSTMs***가 사용된다.

___

**Seq2Seq Model**은 Encoder와 Decoder가 따로 존재하기 때문에, **Input의 길이와 Output의 길이가 달라도 된다**는 장점이 있다.

**Seq2Seq Model**은 주로 **통역**에 사용된다.

- 통역에 사용할 경우, Input sentence의 언어와 Output sentence의 언어가 다르기 때문에, **Input과 Output의 Vocab을 각각 구현**해야 한다.

통역을 위한 Seq2Seq Model을 Training 할 때, **모델이 Prediction을 잘하도록 설계**해야 한다.

- Loss로는 각 위치의 Label 예측과의 차이를 모두 계산하여 더한 **Cross Entropy Loss**를 사용한다.

**Encoder - Decoder Model with RNNs**

먼저, Encoder은 **Output을 생성할 필요가 없다**.

- Encoder에서 사용하던 **Hidden State을 Decoder**로 그대로 넘겨주면 된다.
- Encoder는 **Embedding of Inputs과 RNN Cell**로 이루어져 있다.
- Input은 Embedding layer를 거쳐 RNN Cell로 들어가고 직전 Hidden State도 RNN Cell로 들어간다.

Decoder는 Encoder에 Output을 생성하기 위한 **Linear Layer와 Softmax funciton이 추가**된다. 


PyTorch로 구현하기 위해선 **Encoder, Decoder** Class를 각각 따로 정의한다.

- Encoder object와 Decoder object를 모두 parameter로 받아 사용하는 **Seq2Seq** Class를 정의할 수도 있다.

Training 시에는 Decoder가 생성한 Output의 Argmax가 Target과 동일한지 확인하면 된다.

**Decoder**를 Training하는 경우에는 **Teacher Forcing**을 사용한다.

- 이전 출력의 오차가 누적되서 Training을 망치지 않도록, **특정 확률로 Training 시에는 Input 값으로 Ground Truth을 사용**한다.
- 예를 들어, Random value를 받아 그 값이 설정한 **확률값을 넘으면 Ground Truth**를 다음 Step의 Input으로, 넘지 않으면 **모델의 예측값을 그대로 사용**한다.

**Decoder**는 한번에 하나의 입력으로 출력을 만들도록 설계하고, 반복문을 이용하여 Training 한다.

- 이때, **시작 Token (Input)은 <BOS>**이다.

**Inference** 때는 **<EOS> Token**이 생성되면 종료하도록 한다.

**Seq2Seq Model**을 구현할 때에는 **Encoder와 Decoder의** **Hidden dimension이 동일**해야 하고, **RNN Cell의 Layer 개수가 동일**해야 한다.

# Evaluating Matrices

특정 문장이 생성될 확률은 두 가지로 나타낼 수 있다.

1. $P(W_t, ..., W_2, W_1)$ = $P(W_1)P(W_2|W_1)P(W_3|W_2, W_1)...P(W_t|W_{t-1}, ..., W_{1})$

2. $P(W_t, ..., W_2, W_1)$ = $P(W_t|W_{t-1}, \theta)P(W_{t-1}|W_{t-2}, \theta)...P(W_2|W_1, \theta)$

       - $\theta$는 모델이 학습한 파라미터, 분포 등을 나타낸다.

위에서 (2)의 경우에, **Cross Entropy Loss**를 사용할 수 있다.

- 이 때, **Target Probability Distribution과 Prediction Probability Distribution이 일치하면 Loss가 0**이 된다.

NLP 분야 또는 텍스트 생성형 모델의 **성능을 평가하는 여러 지표**가 있다.

- **Ground Truth Text와 Generated Text간의 Similarity**를 평가한다.

## Perplexity

PPL(W) = $e^{CELoss}$

- CELoss에 비례하여 증가하기 때문에, 값이 작을수록 좋은 모델로 평가된다.

## N-gram

Text를 각각 N-gram으로 묶었을 때, **Ground Truth의 N-gram과 일치하는 N-gram의 개수**를 사용한다.

- **Precision** = 일치하는 N-gram의 개수 / Generated에서의 N-gram의 개수
- **Recall** = 일치하는 N-gram의 개수 / Ground Truth에서의 N-gram의 개수

**F1-Score** = 2 x (Precision x Recall) / (Precision + Recall)

- **Harmonic mean**을 사용한다.

## Library

**NLTK Library**

- **BLEU (Bilingual Evaluation Understudy) score**: N-Grams을 기반으로 **생성된 문장과 Ground Truth 문장이 얼마나 유사한지**를 평가하는 지표

      - `nlrk.translate.bleu_score`

- **METEOR**

      - `nltk.translate.meteor _score`

**PyTorch Library**

- `torch.nn.CrossEntropyLoss`
