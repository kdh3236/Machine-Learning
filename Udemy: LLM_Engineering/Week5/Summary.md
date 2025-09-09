## Day1

**RAG(Retrieval Augmented Generation)**: LLM이 사용자의 질문에 대한 답변의 질을 높이기 위해 **Knowledge base에 retrieve** 하는 것

LLM은 **두 가지 종류**로 나뉜다.

1. **Auto-regressive LLMs**: 과거의 Token 집합을 입력받아 미래의 Token을 예측한다.

2. **Auto-encoding LLMs**: 전체 입력을 받고, 그를 기반으로 Output을 생성

    - **BERT**가 대표적인 예시로, Classification, Sentiment Analysis 등에 사용된다.
  
    - 추가로, **Vector embedding**을 사용한다. 전체 입력 각각을 숫자로 바꾼다면, **지정된 차원에서의 특정 한 점**으로 나타나게 된다.
  
    - **Vocabulary**에 의해 **정수 ID로 변환된 Input을 벡터로 매핑**하게 된다.
  
        - **Embedding LLM**내에 **Embedding Matrix**가 존재한다.   
  
    - 모델이 **Embedding Dimension**을 $d$로 정했고, Input Token이 $l$개 있다면 $l$개의 $d$차원 벡터가 만들어진다. 
  
  
**Vector Embedding**

- Token, word, document 등 모든 입력을 Sequence로 바꿀 수 있다.

- 비슷한 의미의 입력은 Vector Space에서 비슷한 위치에 있어야 한다.

- **Vector Math**의 관점에서 Queen을 설명하기 위해 "King - Man + Woman"을 생각할 수 있다.

    - King에서 Man을 빼면 Queen에 비교적 가까워지고 Woman을 더한다면 Queen에 더욱 더 가까워진다.
 
더 정확하게는 **RAG**는 아래와 같이 동작한다.

1. 입력 Chat이 들어오면, Code가 질문을 받아 **Encoding LLM**에 전달하고 **Vectorize**된 결과를 받는다.

2. (1)의 결과로 나온 Vector를 **Vector Datastore**에 **Retrieve**한다.

    - **Vector Datastore**에 사전에 저장되어있던 정보도 모두 Vector 형태로 저장되어 있으며, 관련된 정보는 Question vector와 비슷한 위치에 있어야 한다.

3. (2)에서 얻은 추가적인 정보까지 추가하여 **Auto-regressive LLMs**에 **Prompt**로 전달하고 답을 얻는다.

Exercise에서 `glob()`을 사용하는 것을 확인할 수 있었다.

- pattern 아래의 **모든 파일과 폴더 경로를 List로 반환**한다.

- `recursive=True`라면 하위 폴더까지 탐색한다.

```python
glob.glob(pattern, recursive=False)
```


## Day2

**LangChain**: LLM Application을 빠르게 Build할 수 있게 해주는 Framework

Day2에서는 벡터 검색 대신, LangChain을 이용하여 Text 기반 검색하는 것을 해본다.

```python
# 파일을 불러오는 것을 도와줌
from langchain.document_loaders import DirectoryLoader, TextLoader
# Document를 받아 Chunk로 나눔
from langchain.text_splitter import CharacterTextSplitter

text_loader_kwargs = {'encoding': 'utf-8'}

documents=[]
doc_type = os.path.basename(folder)

# glob="**/*.md": 하위 폴더의 .md파일까지 전부 탐색
# loader_cls=TextLoader: 단순히 Text를 읽어 Document로  
loader = DirectoryLoader(folder_name, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

# load() 시점에 실제 Document object의 List가 반환된다.
folder_docs = loader.load()

for doc in folder_docs:
    # metadata 설정, doc_type: 폴더 이름
    doc.metadata["doc_type"] = doc_type
    documents.append(doc)

# 한 Chunk의 텍스트 개수가 최대 1000, 각 chunk마다 200 글자씩은 겹침
# 각 문서는 독립적으로 처리되고, 모든 chunk가 List 형태로 반환됨
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# metadata가 dictionary 형태로 확인됨
# doc_type은 파일명
chunk.metadata
chunk.metadata["doc_type"]

# chunk의 내용 확인
chunk.page_content
```

chunk에서 **일부분을 겹치게 하는 이유**는 아래와 같다.

1. **정보 손실 방지**

2. **같은 내용이 여러 Chunk의 중복 포함되면서 의미 기반 검색에서 의미적으로 가까운 결과가 나올 확률이 높아진다.**

추가로, 한 chunk에 여러개의 문서의 내용이 섞이진 않는다.


## Day3

**Vector Embedding Models**을 구현하는 방법은 여러 가지가 있다.

1. 특정 어휘가 나온 횟수에 따라 정수 ID를 매핑하는 것

        - 가장 간단하지만, 한 단어가 두 개 이상의 의미가 있는 등의 경우에 정확히 동작하지 않는다.

2. **word2vec**: Neural Network를 이용해 Vector로 매핑

3. **BERT**

4. **OpenAI Embedding**


**Chroma**: Open-source Vector Database

먼저 Vector DB에 대해 탐색하기 전에 **Embedding할 모델을 설정**해야 한다.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

이후, Vector DB를 생성한다. 같은 이름의 DB가 이미 존재한다면 **삭제 후 생성**할 수 있도록 한다.

```python
from langchain_chroma import Chroma

# 같은 이름의 DB가 존재한다면 삭제
# persist_directory: 쓸 폴더 경
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# documents, embedding model(function)을 지정한다.
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# DB 속 Document의 개수를 셀 수도 있다.
vectorstore._collection.count()

# DB 내의 Vector 집합을 얻는다.
collection = vectorstore._collection
# 하나의 Vector만 Sample로 얻는다.
# limit=1: Collection에서 가져올 최대 개수
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
# Sample의 길이가 한 vector의 Dimension이다.
dimensions = len(sample_embedding)

# 아래와 같은 방법으로 원하는 결과를 얻을 수 있다.
# include=[]: 원하는 정보를 입력
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
```

Exercise에서 한 vector의 dimension은 1,536이다.

사람은 3차원까지만 시각적으로 확인할 수 있기 때문에, **t-SNE**를 이용하여 **2차원 또는 3차원까지 차원을 축소해야 결과를 시각적으로 확인**할 수 있다.

```python
from sklearn.manifold import TSNE

# n_components: 차원수, random_state: seed 설정
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)
```

## Day4

**Key Abstractions in LandChain**

1. **LLM**

2. **Retriever**: 검색 엔진

3. **Memory**: 사용자와 대화한 기록

위 3가지를 이용하여 **Conversation chain**을 만들 수 있다.

- 대화 맥락을 함께 넣어 LLM을 호출하는 파이프라인

**Conversation_chain**을 생성하는 기본 구조는 아래와 같다.

```python
# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
# 대화 (질문/답변)을 메모리에 저장
# memory_key=: 체인이 프롬프트에 끼워 넣을 대화 기록의 키 이름
# return_messages=True: Text 객체 형식으로 저장
# 이후 memory는 자동으로 사용자와의 대화를 추적한다.
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
# RAG Chain 생성
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
```

전체 과정을 확인하면 아래와 같다.

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

query = "Can you describe Insurellm in a few sentences"
result = conversation_chain.invoke({"question":query})
# 답변만 출력
print(result["answer"])

# set up a new conversation memory for the chat
# 이전과 다른 새로운 대화를 시작하려면 Memory를 초기화해야됨
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# 초기화한 Memory에 맞추어 Conversation_chain 실행
# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
```

위는 **RAG**를 사용한 코드이며, Day 1 ~ 3와는 다르게 **직접적으로 특정 단어가 언급되지 않아도 질문에 대한 답변**을 하는 것을 확인할 수 있다.

## Day 5

`stdOutCallbackHandler`를 통해 **Langchain의 Background에서 어떠한 작업이 이루어지는지 확인**할 수 있다.

```python
from langchain_core.callbacks import StdOutCallbackHandler

# conversation_chain을 생성할 때, callbacks=[StdOutCallbackHandler()]만 추가해주면 된다.
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])
```

기존처럼 **하나의 chunk**만 LLM에 넘겨주게 되면, **정보가 부족**하여 제대로 된 답변을 하지 못 하는 경우가 많다.

이 경우, 아래와 같은 방법을 이용하여 해결할 수 있다.

1. **Overlap을 늘린다.**

2. **Chunks의 최대 크기를 늘리거나 줄인다.**

3. **Document 전체를 LLM에 넘겨준다.**

강의에서 사용한 방법은 `as_retreiver(search_kwargs={"k": })`를 이용한다.

```python
# 아래와 같이 사용하면 검색 결과 연관성이 가장 높은 n개의 Chunk를 LLM에 념겨주게 된다.
retriever = vectorstore.as_retriever(search_kwargs={"k": n})
```
