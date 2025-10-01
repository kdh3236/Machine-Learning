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
  speparator="",
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
