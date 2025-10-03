# LangChain Document Loader

> **원하는 형식 (Text, PDF, JSON, CSV, Web 등)의 파일을 Load하고 Parsing할 수 있다.**

- 어떤 파일 형식이던 LangChain의 **Document(텍스트 + 메타데이터) 형식**으로 변환해준다.

```python

# load할 파일 형식과 파일에 일치하는 Loader 정의
# PyPDFLoader, PyMuPDFLoader, UnstructeredMarkdownLoader, CSVLoader, WebBaseLoader, UnstructuredFileLoader
# 등을 파일 형식에 맞게 사용할 수 있다.
loader = Textloader('경로')

# 여러 파일을 Load하려면 loader 정의시 Parameter에 List를 넘겨주면 된다.

# 실제로 Load
data = loader.load()
````

대용량 파일을 Load하기 위해서 **Local Caching**을 사용할 수도 있다.

- Memory 최적화 관련 작업도 수행할 수 있다.

Load 속도를 높이기 위해서 **병렬 처리**할 수 있다.

특히, LLM의 입력 Token 개수에는 한계가 있기 때문에, **Loader를 통해 읽은 data를 전부 LLM에 넣지 않고, Pre-processing하는 것이 좋다.**

- **Context window**에 의해 제한된다.

# LangChain Text Splitter

> **Load한 파일을 CHunk 단위로 나누는 것을 도와주는 모듈**

**Text Splitter**는 **Separator, Chunk size, Chunk overlap and Length function**의 Parameter를 가진다.

- Length function은 Chunk의 Length가 계산되는 방식을 정의한다.

```python
# RecursiveCharacterTextSplitter를 통해 여러 구분자로 Chunk를 계속 나누어 갈 수도 있다.
# RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON)을 통해 Programming language 기준으로 나눌 수도 있다.
# MarkdownHeaderTextSplitter(): Markdown 문법에 맞추어 Split
text_splitter = CharacterTextSplitter(
  separator="",
  chunk_size=200,
  chunk_overlap=20,
  length_function=len,

# 주어진 Text를 split
texts = text_splitter.split_text(text)

# Text Splitter로 쪼개고 Document 형식으로 반환하도록 할 수 있다.
texts = text_splitter.create_documents([documents], metadatas=[])
```

# Embedding model

**Document**를 **Chunk**로 나눈 이후, Vector DB에 넣기 위해 Embedding해야 한다.

- 실습에선 `WatsonxEmbeddings()`를 이용하여 Embedding model을 호출한다.

```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

embed_params = {
    # Input이 길 경우 앞 3개 Token만 사용
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    # 원본 Text도 같이 반환
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

# Chunk로 나눈 Document를 Embedding
doc_result = watsonx_embedding.embed_documents(chunks)

# Query 역시 Embedding할 수 있다.
query_result = watsonx_embedding.embed_query(query)
```

`HuggingFaceEmbeddings` Embedding model을 사용할 수 있다.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)

query_result = huggingface_embedding.embed_query(query)

doc_result = huggingface_embedding.embed_documents(chunks)
```

이후, **비슷한 Vector Embedding을 갖는 Document끼리 Clustering**을 해야 한다.

# Chroma: Open Source Vector Database

Vector DB는 SQL같은 Traditional DB에 비해 **Search, Silmilarity 계산**에 이점을 보인다.

```python
from langchain.vectorstores import Chroma

# Vector DB 생성
ids = [str(i) for i in range(0, len(chunks))]
vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)

# vector DB에 들어있는 Document의 개수
vectordb._collection.count()

# id (index)에 대응되는 Document를 반환
vectordb._collection.get(ids=str(i))

# query와 가장 유사한 3개의 Document를 반환
docs = vectordb.similarity_search(query, k=3)
```

Vector DB에 **추가 및 수정**할 수 있다.

- 이떄의 Chunk는 Document 객체여야 한다.

```python
vectordb.add_documents(
    new_chunks,
    ids=["215"]
)

# 내용 변경
vectordb.update_document(
    '215',
    update_chunk,
)

# 삭제
vectordb._collection.delete(ids=['215'])
```

# Vector Store-Based Retriever

Vector DB에서 유사한 Vector를 찾도록 하는 다른 방법이 있다.

일반적인 **Retriever**는 아래와 같이 구현할 수 있다.

```python
# 원하는 Vector DB에 대한 Retriever 선택
# Top-3개 반환
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
# Query와 비슷한 Vector를 찾아오도록 한다.
docs = retriever.invoke(query)
```


### Maximal Marginal Relevance (MMR)

Query와 **관련성이 높은 Vector를 선택하면서 서로 중복되는 문서는 줄이도록** 하는 방법

**Retriever**가 **MMR** 방법을 사용하도록 할 수도 있다.

```python
retriever = vectordb.as_retriever(search_type="mmr")
docs = retriever.invoke(query)
```

### Similarity score threshold

Embedding vector간 **Score가 Threshold 이상인 Document만 반환**하도록 하는 방법

```python
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4}
)
docs = retriever.invoke(query)
```

`LangChain`에서 **여러 Retrieve 방식**을 사용할 수 있다.

### 1. Multi-Query Retriever

주어진 한 Query에 대해 **LLM이 직접 여러 버전의 Query를 생성하고 여러 Query를 이용하여 Retrieve**하는 방법

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# llm model도 함께 넘겨주어야 한다.
retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm()
)

docs = retriever.invoke(query)
```

### 2. Self-Query Retriever

주어진 한 Query에 대해 **LLM이 직접 Query의 의미를 설명하는 여러 부분 (String)과 Metadata Filter로 구성된 Query를 만들어 사용**하는 방법

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

# 검색을 위해 Metadata filter를 정의한다.
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

document_content_description = "Brief summary of a movie."

retriever = SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info,
)
```

### 3. Parent Document Retriever

Vector DB에 대한 **Query에 대해 작은 크기의 Child chunk를 통해 검색되고, 실제 반환은 큰 크기의 Parent chunk**로 하는 방법

- **작은 크기의 Chunk로는 의미 전달이 정확히 되지 않을 가능성**이 있기 때문에 실제로는 비교적 큰 크기의 Parent chunk를 반환한다.

```python
from langchain.retrievers import ParentDocumentRetriever

# 먼저 큰 크기로 Parent splitter로 나누고 나눈 것을 Child splitter로 또 나눈다.
parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator='\n')

# Parent를 보관하도록 한다.
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

# Build QA Bot using LLM and RAG

이제, LLM 모델을 이용하여 RAG를 참고하여 Query에 대해 답하도록 할 수 있다.

```python
from langchain_ibm import WatsonxLLM
from langchain.chains import RetrievalQA

query = "What this paper is talking about??"
watsonx_llm = WatsonxLLM(
    model_id='ibm/granite-3-2-8b-instruct',
    url="https://us-south.ml.cloud.ibm.com",
    project_id='skills-network',
    params={"max_new_tokens": 256},
)

qa = RetrievalQA.from_chain_type(
    llm=watsonx_llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=False
)
response = qa.invoke(query)
print(f"Response: {response}")
```
