## Day1

**Multi-Agents Price predictor**를 구현한다.

- **Planning Agent**: 아래 세 Agent 중 어떤 Agent를 어떤 순서로 돌릴지를 결정한다.
- **Scanner Agent**: 데이터셋에서 이상적인 Item Search
- **Ensemble Agent**: Price 예측

  - **Frontier model with RAG, Specialist Agent, Random Forest**의 결과를 모두 본 이후에 예측한다.
  
- **Messaging Agent**: 최종 결과 메세지를 전

**Modal**: 파이썬 중심의 Serverless Platform

- Computing Unit에 따라 비용 부과
- 코드만으로 CPU/GPU 작업 수행 가능

**Modal**을 사용하는 방법은 아래와 같다.

```python
import modal
from modal import App, Image

# Token (API)를 생성하는 명령어
!modal setup

# 아래 부분은 다른 파이썬 파일에서 정의
# 앱 이름 설정
app = modal.App("hello")
# debian_slime의 OS를 사용하고 필요한 Library를 설치
image = Image.debian_slim().pip_install("Needed Library")
# Token Value 등을 설정
secrets = [modal.Secret.from_name("hf-secret")]
# GPU 하드웨어 설정
GPU="T4"

# 아래 Decorator로 정의되는 함수를 Modal의 기본 실행 단위로 설정
@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)

# 아래부터는 다시 원래 파일에서 실행
from 파이썬 파일명 import app, 함수명

# 실행
with app.run():
    # Local에서 실행
    reply=hello.local()

with app.run():
    # remote(): 원격 클라우드에서 실행
    reply=hello.remote()
```

먼저 **Ensemble Agent**의 구현을 살펴보자.

- 가장 먼저 이를 위한 **Specialist Agent를 구현한다.**

우리는 **LLM이 Prompt에서 설명하는 물건에 대해 가격을 예측하도록 하는 함수를** 정의하여 Modal의 실행 단위로 사용할 것이다.

```python
import modal
from pricer_ephemeral import app, price

# enable_output(): 출력(로그/진행상황)을 강제로 화면에 보여줌.
with modal.enable_output():
    with app.run():
        result=price.remote("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio")
result
```

이후, Modal을 이용하여 **앱을 배포할 수 있다.**

먼저, **함수를 이용**하여 앱을 배포하는 과정을 알아보자.

- 이전 작업과 동일하며, 파이썬 파일에 함수가 정의된 것이다.

```python
# pricer_service.py 모듈을 Modal에 배포하는 명령어
!modal deploy -m pricer_service

# 배포되어 있는 앱, 함수를 다룰 수 있는 Handle 설정
# 첫 Argument "pricer-service"는 앱 이름(코드에서 modal.App("pricer-service")로 만든 그 이름).
# 두 번째 Argument "price"는 @app.function으로 데코레이트된 함수 이름.
pricer = modal.Function.from_name("pricer-service", "price")

# Prompt를 기반으로 Inference
pricer.remote("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio")
```

이후, **Class를 이용**하여 앱을 배포하는 과정을 알아보자.

- 파이썬 파일에 Class가 정의된다.

먼저 Class를 정의해야 한다.

```python
@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets, 
    gpu=GPU, 
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume}
)
class Pricer:

    @modal.enter() # Class instance가 한 번 올라올 때 실행되도록 한다.
    def setup(self):
        # 보통 모델 로드, 토크나이저 준비, DB 연결을 미리 Load한다.
        pass

    @modal.method() # 원격에서 호출 가능한 클래스 메서드
    def price(self, description: str) -> float:
        pass
```

이후 아래와 같이 사용할 수 있다.

```python
# pricer_service2.py 모듈을 Modal에 배포하는 명령어
!modal deploy -m pricer_service2

# 배포되어 있는 앱, Class를 다룰 수 있는 Handle 설정
# 첫 Argument "pricer-service"는 앱 이름(코드에서 modal.App("pricer-service")로 만든 그 이름).
# 두 번째 Argument "price"는 @app.function으로 데코레이트된 함수 이름.
Pricer = modal.Cls.from_name("pricer-service", "Pricer")
# Object 생성
pricer = Pricer()

# 객체 내 함수를 실행하여 Inference
# 이때도 함수를 거쳐 remote()를 호출한다.
reply = pricer.price.remote("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio")
```

Class를 사용하면 `@model.enter()` 덕분에 미리 **Load하고 재사용하여 효율적으로 동작**하도록 할 수 있다.

**프롬포트를 받아들이고 배포된 앱을 더 쉽게 호출할 수 있는 Agent**를 생성할 수도 있다. 

```python
# Agent class는 따로 구현
# self.log()를 가짐
from agents.agent import Agent

class SpecialistAgent(Agent):
    """
    An Agent that runs our fine-tuned LLM that's running remotely on Modal
    """

    name = "Specialist Agent"
    color = Agent.RED

    # 생성자에서 배포된 Module을 호출한다. 
    def __init__(self):
        """
        Set up this Agent by creating an instance of the modal class
        """
        self.log("Specialist Agent is initializing - connecting to modal")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
        self.log("Specialist Agent is ready")
    # price 함수를 따로 호출하여 Inference 한다.
    def price(self, description: str) -> float:
        """
        Make a remote call to return the estimate of the price of this item
        """
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result


# 실제 사용
agent = SpecialistAgent()
agent.price("iPad Pro 2nd generation")
```

## Day2

**Frontier model with RAG**를 다루기 위해 Dataset에 대해 **RAG Database**를 만들자.

- 여기선 LangChain을 사용하지 않는다.

**Chroma Client**: DB랑 직접 소통하는 핸들
**Collection**: 논리적인 Table

```python
import chromadb

# DB 폴더명 
DB = "products_vectorstore"

# PersistentClient(): Local 폴더를 저장소로 사용하는 Client를 생성
client = chromadb.PersistentClient(path=DB)

# 만들거나 삭제할 Collection name
collection_name = "products"

# 현재 Clinet에 존재하는 Collenction name list가 반환
existing_collection_names = client.list_collections()

# 같은 이름의 Collection이 존재한다면 삭제
if collection_name in existing_collection_names:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")

# Collection 생성
collection = client.create_collection(collection_name)
```

이후, **SentenceTransformer**를 통해 **텍스트를 벡터로 매핑**할 수 있다.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# List의 한 Element를 벡터하나로 매핑해서 List로 반환한다.
vector = model.encode(["Well hi there"])[0]
```

벡터로 매핑한 결과를 **collection.add()를** 통해 Chroma DB에 추가할 수 있다.


```python
collection = client.create_collection(collection_name)

documents = [description(item) for item in train[i: i+1000]]
vectors = model.encode(documents).astype(float).tolist()
metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
ids = [f"doc_{j}" for j in range(i, i+len(documents))]

# id는 절대 겹치면 안 된다.
collection.add(
    ids=ids,
    documents=documents,
    embeddings=vectors,
    metadatas=metadatas
)
```

