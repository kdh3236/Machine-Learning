# LangChain Document Loader

> **원하는 형식 (Text, PDF, JSON, CSV, Web 등)의 파일을 Load하고 Parsing할 수 있다.**

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
