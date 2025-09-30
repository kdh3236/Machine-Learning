# Retrieval-Augmented Generation (RAG)

> **LLM을 최적화하기 위한 AI Framework**

- **관련된 지식이나, 전문 지식**을 추가하는 등에 유용하다.

## RAG process

1. 주어진 Prompt를 Encoding해서 **High-dimentional vector**로 만든다.
2. Vector DB에서 일치하는 정보를 **Retireve**한다.
3. 원래 **Prompt와 Retirever를 통해 찾는 Relevant context**를 함께 Generator (ChatBot)에 넘겨주어 Response를 생성한다.

각 과정을 자세히 살펴보자.

- `Retreiver` 부분 먼저 살펴보자.

먼저 **Knowledge Base**에 저장하고자 하는 **Context / Information / Document**를 **문장 또는 Chunk 단위로 나누고** 각각을 **BERT / GPT (Decoder-only) model** 등을 이용하여 **High-dimensional vector**로 만든다.

- 각 Token에 대응되는 Vector를 얻고 평균을 낸다.
- **Vector DB**를 구성하게 된다.

이후 Prompt를 **BERT / GPT (Decoder-only) model** 등을 이용하여 **High-dimensional vector**로 만든다.

**Retrieve** 과정에선 Vector DB에서 Prompt에 대응되는 **Vector와 가장 가까운 Vector**를 찾는다.

- **Cosine Similarity** 등의 방법을 이용한다.
- Hyperparameter $k$를 통해 **Top-k 개**를 사용하기도 한다.

**Retrieve 과정**에서 찾은 Context를 Prompt와 함께 Generator에 넘겨준다.  

## Dense Passage Retrieval (DPR) Context Encoder

**Document** 등을 Encoding 하는 과정에서 **Text를 Emedding**을 하고 **Question vector와 쉽게 비교**할 수 있도록 해주는 모델

`DPRContextEncoderTokenizer`, `DPRContextEncoder`를 통해 쉽게 구현할 수 있다. 

```python
from tokenizer import DPRContextEncoderTokenizer, DPRContextEncoder

model = ''
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model)

tokens_info = context_tokenizer(text)

encoder_model = ''
context_encoder = DPRContextEncoder.from_pretrained(encoder_model)

embeddings = []

outputs = context_encoder(**tokens_info)
embeddings.appemd(outputs.pooler_output)

context_embeddings = torch.cat(embeddings).numpy()
```

## Facebook AI Similarity Search (Faiss)

> **High-dimensional vectors의 집합에서 Searching하는 효율적인 알고리즘을 제공하는 라이브러리**

```python
import faiss

embed_dim = 768
context_embeddings = np.array(context_embeddings)

index = faiss.IndexFlatL2(embed_dim)
# Search할 수 있도록 Index에 추가
index.add(context_embeddings)
```

## Dense Passage Retrieval (DPR) Question Encoder

사용자의 Question에 맞추어 Tokenization하도록 도와주는 라이브러리

```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

tokenozer_model = ''
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(tokenizer_model)

encoder_model = ''
question_encoder = DPRQuestionEncoder.from_pretrained(encoder_model)

# Qeury and Context retrieval
question = ''
question_input = question_tokenizer(question, return_type='pt')
question_embedding = question_encdoer(**question_inputs).pooler_output.numpy()

# Top-k개를 찾음
D, I = index.search(question_embedding, k=3)
```

## BART

