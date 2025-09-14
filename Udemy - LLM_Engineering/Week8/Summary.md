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

이제 **RAG Pipeline**을 구현하자.

- Chroma DB의 `collection.query()`를 이용하여 쉽게 구현할 수 있다.

```python
# 생성한 collection을 가지고 옴.
client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection('products')

# DB에서 item()과 유사한 n_results 만큼의 List를 반환
# 아래 코드를 함수로 만들면, item에 대해 RAG Pipeline을 이용하는 것처럼 만들 수 있다.
# 아래 documents와 prices를 이용하여 prompt를 만들고 Frontier model에서 답을 내도록 요청할 수 있다.
results = collection.query(query_embeddings=vector(item).astype(float).tolist(), n_results=5)
documents = results['documents'][0][:]
prices = [m['price'] for m in results['metadatas'][0][:]]
```

마지막으로 **Frontier model with RAG**를 **Agent로 만들기 위해 Function으로 Wraping할 수 있다.**

- 이 함수는 **RAG Pipeline**을 거쳐 DB에서 유사한 정보를 얻고, 그를 바탕으로 **Prompt를 생성해 Frontier 모델에 답변**을 요청한다.

**RandomForest Agent**는 이미 구현된 RandomForest Model을 사용하고 **Item을 입력 받아 Vectorize**하고 **Vectorized DB에서 Random forest 알고리즘**을 수행한다.

이제 **Special Agent, RandomForest Agent 그리고 Frontier Agent**를 **Ensemble**헤보자.

- 각 Agent가 예측한 결과와 각 결과에서의 최댓값, 최솟값을 이용하여 LinearRegression을 진행한다.
- LinearRegression의 결과 (Weights, Parameter)를 .pkl 파일로 저장한다.
- .pkl 파일로 저장한 모델을 불러오고 마찬가지로 각 Agent가 예측한 결과와 각 결과에서의 최댓값, 최솟값을 이용하여 가격을 예측한다.

```python
# 모델의 가중치, 파라미터를 저장할 때 joblib을 사용한다.
import joblib
from sklearn.linear_model import LinearRegression

# 저장
joblib.dump(lr, 'ensemble_model.pkl')
# 불러오기
joblib.load('ensemble_model.pkl')

# Training
specialists = []
frontiers = []
random_forests = []
prices = []
for item in tqdm(test[1000:1250]):
    text = description(item)
    specialists.append(specialist.price(text))
    frontiers.append(frontier.price(text))
    random_forests.append(random_forest.price(text))
    prices.append(item.price)

mins = [min(s,f,r) for s,f,r in zip(specialists, frontiers, random_forests)]
maxes = [max(s,f,r) for s,f,r in zip(specialists, frontiers, random_forests)]

X = pd.DataFrame({
    'Specialist': specialists,
    'Frontier': frontiers,
    'RandomForest': random_forests,
    'Min': mins,
    'Max': maxes,
})

# Convert y to a Series
y = pd.Series(prices)

np.random.seed(42)

lr = LinearRegression()
lr.fit(X, y)
```

## Day3

**Structured Outputs**: JSON 형식 대신 Python Class로 답변을 정의하여 더욱 안정감있게 대답하도록 한다.

- 모델이 Class의 Instance를 생성한다.

- 모델이 생성한 JSON이 정확히 스키마에 맞는지 확인되고, 맞지 않으면 에러

- 프롬프트만으로 JSON을 대충받는 것보다 신뢰성이 높다.

```python
# 인터넷에서 물건 판매 정보를 Scraping
# 따로 정의한 Python class
from agents.deals import ScrapedDeal, DealSelection

# Top 10 Deals를 불러온다.
deals = ScrapedDeal.fetch(show_progress=True)
```

Class로 답변 형식을 지정하는 것은 아래와 같이 할 수 있다.

```python
from pydantic import BaseModel

# BaseModel을 상속하면 타입 주석에 맞게 자동으로 __init__()이 호출된다.
class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """
    product_description: str
    price: float
    url: str

class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """
    deals: List[Deal]

# List[Deal]은 아래의 JSON 형식과 완벽히 동일하다.

"""
{
  "deals": [
    {
      "product_description": "str",
      "price": 3.0,
      "url": "~~"
    }
  ]
}
"""
```

이후에는 Frontier model을 호출할 때, 위에서 정의한 **DealSelection class**를 사용할 수 있다.

```python
# beta.chat.completions.parse: class 객체로 답변할 수 있도록 함.
completion = openai.beta.chat.completions.parse(
  model="gpt-4o-mini",
  messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt}
  ],
  response_format=DealSelection
)

# 답변 형식도 기존과 다름
result = completion.choices[0].message.parsed  
```

마지막으로 Website에서 Deal을 추천받고, Detail을 바탕으로 Frontier Model에게 Deal에 대해 **가격을 포함하여 정해진 형식의 정리본**을 요청한다.

- 이를 **Scanner Agent**로 사용한다.

## Day4

**Agentic AI Solution의 특징**

1. 큰 문제를 작은 문제로 나누어 단계별로 처리

2. Structed outputs / Fucntion Calling / Using tools

3. 모든 Agent가 공통으로 사용할 수 있는 부분이 존재

4. Planning Agent: 각 Agent가 어떤 동작을 할지

5. Autonomy & Memory: 사람의 도움 없이 AI가 자동으로 처리한다.

**Planning Agent**와 **Messaging Agent**의 구현에 대해 살펴보자.

먼저 **Messaging Agent**로 두 가지를 고민할 수 있다.

- **Twilio**와 **Pushover**이 있는데 강의에서는 **Pushover**를 추천한다.
- 미국 내 규제 때문에 **Twilio**는 서류 작업이 많이 필요하다고 한다.

```python
### twilio를 사용하는 경우
from twilio.rest import Client

account_sid = os.getenv('TWILIO_ACCOUNT_SID', 'your-sid-if-not-using-env')
auth_token = os.getenv('TWILIO_AUTH_TOKEN', 'your-auth-if-not-using-env')
self.me_from = os.getenv('TWILIO_FROM', 'your-phone-number-if-not-using-env')
self.me_to = os.getenv('MY_PHONE_NUMBER', 'your-phone-number-if-not-using-env')
self.client = Client(account_sid, auth_token)
self.log("Messaging Agent has initialized Twilio")

self.log("Messaging Agent is sending a text message")
message = self.client.messages.create(
          from_=self.me_from,   # 발신 번호 (Twilio 콘솔에서 승인/구매한 번호)
          body=text,            # 보낼 텍스트
          to=self.me_to         # 수신 번호
)


### Pushover를 사용하는 경우
self.pushover_user = os.getenv('PUSHOVER_USER', 'your-pushover-user-if-not-using-env')
self.pushover_token = os.getenv('PUSHOVER_TOKEN', 'your-pushover-user-if-not-using-env')
self.log("Messaging Agent has initialized Pushover")

self.log("Messaging Agent is sending a push notification")
conn = http.client.HTTPSConnection("api.pushover.net:443")
conn.request(
      "POST",
      "/1/messages.json",
      urllib.parse.urlencode(
        {
          "token": self.pushover_token,
          "user": self.pushover_user,
          "message": text,
          "sound": "cashregister"
        }
      ),
      { "Content-type": "application/x-www-form-urlencoded" }
)
conn.getresponse()

### 공통
# opportunity는 Opportunity 객체로 실제 사용자에게 알림을 보낼만한 LLM 답변 객체를 의미
text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
text += f"Estimate=${opportunity.estimate:.2f}, "
text += f"Discount=${opportunity.discount:.2f} :"
text += opportunity.deal.product_description[:10]+'... '
text += opportunity.deal.url

# 채널별 발송
if DO_TEXT:
    self.message(text)  # [Twilio] 호출
if DO_PUSH:
    self.push(text)     # [Pushover] 호출

self.log("Messaging Agent has completed")
```

이제 **Planning Agent**에 대해 살펴보자.

- 여러 Agent를 호출해야 한다. (Class object를 주로 활용하는 이유)
- `plan()` 함수를 호출해서 Planning 한다.

이전에 정리할 점이 있다.

- **Deal(원천 데이터)**: 스크래핑/피드에서 올라온 실제 상품 정보(가격, 설명, 링크 등) / 반환 형식을 지정하기 위해 구현한 Class이다.

- **Opportunity(판단 결과)**: Deal + 당신의 가격 추정(estimate) + 평가 메타데이터를 합쳐 “알림 보낼 만한가?”를 표현하는 객체.

```python
class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 50

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Planning Agent is initializing")
        # Agent load
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

      # Scrapping을 하고 LLM을 거쳐 얻은 Deal 객체 하나를 받아서
      # discount: 예측 가격과 실제 판매 가격의 차이 를 얻어
      # 원래의 deal 객체와 estimate, discount를 묶어 Opportunity 객체로 넘김
      def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal
        :param deal: the deal, summarized from an RSS scrape
        :returns: an opportunity including the discount
        """
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EnsembleAgent to estimate them
        3. Use the MessagingAgent to send a notification of deals
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory)
        if selection:
            # 한 Batch마다 5개의 deal을 다룸
            # Oppurtunity 객체의 List
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            # 내림차순으로 정렬
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
            if best.discount > self.DEAL_THRESHOLD:
                # Discount가 DEAL_THRESHOLD 이상이면 알람을 보냄
                self.messenger.alert(best)
            self.log("Planning Agent has completed a run")
            return best if best.discount > self.DEAL_THRESHOLD else None
        return None
```

**Memory**: 기억과 재사용으로 연결되고, **중복 알림·불필요한 모델 호출·비용을 크게 줄일 수 있다.**

전체 코드의 **초기화, 실행, 시각화**를 담당하는 **Agent Framework**는 아래와 같이 구현할 수 있다.

```python
class DealAgentFramework:

    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    # Setting 및 Initialize
    def __init__(self):
        init_logging()
        load_dotenv()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection('products')
        self.planner = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")
        
    def read_memory(self) -> List[Opportunity]:
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            opportunities = [Opportunity(**item) for item in data]
            return opportunities
        return []

    def write_memory(self) -> None:
        data = [opportunity.dict() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        self.init_agents_as_needed()
        logging.info("Kicking off Planning Agent")
        # planning agent 호출
        # Best discount opportunity 객체를 받아옴
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent has completed and returned: {result}")
        if result:
            self.memory.append(result)
            # 메모리에 추가
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=10000):
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection('products')
        result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=max_datapoints)
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        categories = [metadata['category'] for metadata in result['metadatas']]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors

# 이 코드 이후 Tenrminal에서 python 명령어로 실행하면
# DealAgentFramework().run()이 실행된다.
if __name__=="__main__":
    DealAgentFramework().run()
```

## Day5

이제 구현한 모델의 **UI**를 생성하고, 모델이 **사람의 입력없이 자동으로 동작할 수 있도록 구현해야 한다.**

먼저, **UI**는 **Gradio**를 이용하여 구현한다.

