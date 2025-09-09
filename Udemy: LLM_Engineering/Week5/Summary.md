## Day1

**RAG(Retrieval Augmented Generation)**: LLM이 사용자의 질문에 대한 답변의 질을 높이기 위해 Knowledge base에 retrieve 하는 것

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
