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
