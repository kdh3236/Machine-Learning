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

> **Document** 등을 Encoding 하는 과정에서 **Text를 Emedding**을 하고 **Question vector와 쉽게 비교**할 수 있도록 해주는 모델

- **BERT** model을 사용해도 된다.

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
# .pooler_output을 통해 벡터를 얻는다.
embeddings.append(outputs.pooler_output)

[문서 수 × 임베딩 차원]의 차원으로 합쳐 Vector DB를 얻는다.
context_embeddings = torch.cat(embeddings).numpy()
```

## Facebook AI Similarity Search (Faiss)

> **High-dimensional vectors의 집합에서 Searching하는 효율적인 알고리즘을 제공하는 라이브러리**

```python
import faiss

embed_dim = 768
context_embeddings = np.array(context_embeddings)

# L2 distance로 가장 가까운 벡터를 찾도록 하는 index
index = faiss.IndexFlatL2(embed_dim)
# Search할 수 있도록 Index에 추가
index.add(context_embeddings)
```

## Dense Passage Retrieval (DPR) Question Encoder

> **사용자의 Question에 맞추어 Tokenization하도록 도와주는 라이브러리**

```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

tokenozer_model = ''
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(tokenizer_model)

# Embedding encoder
encoder_model = ''
question_encoder = DPRQuestionEncoder.from_pretrained(encoder_model)

# Tokenization -> Embedding
question = ''
question_input = question_tokenizer(question, return_type='pt')
question_embedding = question_encdoer(**question_inputs).pooler_output.numpy()

# question_embedding과 비슷한 Top-k개를 찾음
D, I = index.search(question_embedding, k=3)  # D: 거리/점수, I: 인덱스(문서 행/ID)
```

## BART

> **Encoder-Decoder model로 Question+Context에 맞는 응답을 생성한다.**

- Question으로 Vector DB에서 Context를 찾고, Context를 바탕으로 Response를 생성한다.

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('')
tokenizer = BartTokenizer.from_pretrained('')

def generate_answer(contexts):
    input_text = ' '.join(contexts)
    # Context를 Tokenization
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
    # 응답 생성
    # length_penalty: 짧은 응답에 Penalty를 주어 긴 응답을 생성하도록 한다.
    # num_beams=4: 4개의 응답 중 가장 좋은 응답을 선택하도록 한다.
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, \
    length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')

    question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()

    D, I = index.search(question_embedding, k)

    return D, I

question = "what is mobile policy?"
# 유사한 Vector의 Index를 반환
_, I = search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5)
top_contexts = [paragraphs[idx] for idx in I[0]]
answer = generate_answer(top_contexts)
```

# Prompt Engineering

> **LLMs과는 소통하기 위해서 Designing하고 Refining하는 과정**

- LLMs이 **사용자의 요구 사항에 맞게 적절한 응답**을 할 수 있도록 돕는다.

**Prompt**는 네 가지로 나뉜다.

1. **Instruction**: AI가 **어떤 작업**을 해야하는지 간단하고 직접적으로 설명

2. **Context**: LLM이 Instruction을 따르도록 도와주는 정보

3. **Input data**: LLMs이 다루어야 하는 실제 정보

4. **Output indicator**: LLMs이 따라야하는 응답 형식

**Prompt**에는 여러 종류가 있다.

1. **Zero-shot prompt**: LLM을 **사전 Training 하거나 LLM에 사전 정보 제공 없이** Instruction을 주는 방법

2. **One-shot prompt**: LLM에 Instruction과 함께 **한 가지 예시**를 주는 방법

3. **Few-shot prompt**: LLM에 Instruction과 함께 **몇 개의 예시**를 주는 방법

4. **Chain-of-thought (CoT) Prompting**: LLM에 **추론 과정을 주어 사람의 사고 방식을 이용**할 수 있도록 하는 방법

5. **Self-consistency**: LLM이 같은 질문에 대해 여러 개의 추론을 하도록 한 뒤, **가장 답이 일관적으로 나오는 답**을 선택하도록 하는 방법 

## In-context Learning

> **Prompt engineering의 일종으로 Prompt를통해 LLM이 해야될 작업을 설명하는 것**

- 추가적인 Training이 필요하지 않다.
- 그러나 복잡한 작업을 하기엔 부족한 점이 있다.

# LangChain

> **LLM을 이용한 Application 개발에 도움을 주는 Framework**

LangChain에는 여러 구성요소가 있다.

### Model

LangChain은 여러 **Model**을 이용할 수 있다.

- IBM, OpenAI, Meta 등의 모델을 사용할 수 있다.

- `ModelInference()`를 이용하여 Model을 지정하고, `.generate()`를 통해 Response를 생성한다.

### Chat Message

**Chat Message**는 `role`과 `content` 두 부분으로 나뉜다.

- `role`은 **System message와 Human message** 등으로 나뉘고, `content`는 각 Message의 내용을 의미한다.
- `HumanMessage`, `SystemMessage`, `AIMessage`을 이용한다.

### Prompt templates

**User의 Query를 Clear instruction**으로 변경해준다.

`StringPromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholderTemplate`, `FewShotPromptTemplate` 등이 있다.

```python
# Model이 실제 입력받는 부분의 형식을 지정한다.
prompt = PromptTemplate.from_template(
    "Summarize this:\n{content}\n\nQuestion: {question}"
)

# Content는 RAG를 통해, Question은 User에게 받고 위 형식의 Prompt로 만든 후에 Response를 생성하도록 한다.
chain.invoke({"content": "...", "question": "..."})
```

### Example selector

`Example selector`를 통해 **적절한 Example**을 LLM에 주는 것이 주어진 Task를 정확히 수행하는 데 중요하다.

- `NGramOverlapExampleSelector`, `FewShotPromptTemplate` 등이 있다.

### Output parsers

Response를 사용자가 원하는 형식으로 생성할 수 있도록 해준다.

- Response를 `JSON`, `XML`, `CSV` 등으로 반환할 수 있도록 해준다.

- `CommaSeparetedListOutputParser()` 등을 이용할 수 있다.

LangChain을 이용하여 **RAG**도 구현할 수 있다.

### Documents

`Document()`를 이용하여 String을 Vector화 하여 Vector DB에 저장할 수 있다.

- 여러 형식의 파일을 저장할 수 있다.

#### Document Dataloader

`WebBaseLoader()`를 이용하면 Website의 문서를 받아올 수 있다.

#### Text splitter

`CharacterTextSplitter()`를 이용하면 파일을 구분자를 기준으로 여러 Chunk로 나눈다.

`RecursiveCharacterTextSplitter()`를 이용하여 여러 구분자를 이용하여 Chunk로 나눌 수 있다.

#### Vector database

`Chroma.from_documents`를 통해 Chunk를 Embedding하고 DB에 저장할 수 있다.

#### Document retriver

`docsearch.as_retriever()`와 `.invoke("Question")`을 통해 Question에 맞는 Context를 찾을 수 있다.

___

LangChain은 응답을 생성할 때, **응답의 각 부분을 Chain으로 나누고 여러 Chain을 묶어 하나의 응답처럼 반환하도록 할 수 있다.**

- 레시피를 예로 들면, 음식 선정 / 재료 / 레시피가 각각 한 Chain이고 3개의 Chain이 하나의 응답을 이루는 것이다.

### Chains

각 Chain에 대해 **Prompt Template**을 형성하고 `LLMChain`을 통해 Chain을 생성한다.

- 각 Chain은 **PromptTemplate + LLM (+ OutputParser)이** 하나로 묶인 단위이다.

각 Chain을 생성한 후, `SequentialChain()`을 통해 **각 Chain을 하나의 Process처럼 생성하도록 할 수 있다.**

이후, LLM의 응답은 **정해진 방식의 Chains을 따라야 한다.**

### Memory

대화에서 대화의 맥락이 유지되기 위해서는 Memory가 중요하다.

- `ChatMessageHistory`, `.add_ai_message()`, '.add_user_message()`를 통해 Memory 업데이트가 가능하다.

### Agents

LLM이 **무엇을 할지 스스로 계획하고, Tool을 선택적으로 호출하며, 결과를 읽고 다음 행동을 조정하는 동적 시스템**

- 동적이라는 것이 Chains와의 차이점이다.

`create_pandas_dataframe_agent()`등으로 agent를 지정하고 `.invoke()`로 응답 생성이 가능하다.
