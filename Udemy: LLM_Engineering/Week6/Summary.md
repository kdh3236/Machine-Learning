**Fine-tuning Frontier model**

## Day1

Training을 위해 **어떤 Dataset**을 사용할지도 중요하다.

사용할 수 있는 **Dataset**은 아래와 같다.

1. **Own proprietary data**
   
2. **Kaggle**

3. **HuggingFace datasets**

4. **Synthetic data**: LLM, 알고리즘에 의해 생성된 데이터

5. **Data by Specialist companies**

**데이터를 다루기 위해 확인해야 할 것**들이 있다.

- 사용하고자 하는 **Feature의 분포, 최대 길이** 등을 알아야 한다.

- 이를 **그래프로 시각화**하여 확인하는 것이 좋다.

- 사용하고자 하는 범위를 벗어난 Data를 버리는 작업이 필요할 수도 있다.

먼저 **HuggingFace의 Dataset을 Load하는 방법**을 살펴보자.

```python
from datasets import load_dataset

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",   # 1) 데이터셋 ID 
    f"raw_meta_Appliances",              # 2) Category (여기선 Appliances)에 해당하는 Meta data를 받아옴
    split="full",                        # 3) Dataset 전체를 사용
    trust_remote_code=True               # 4) 로컬에서 실행 허용
)
```

LLM을 Training할 때, 일반적으로 **Dataset을 깔끔하게 정리해주고 Data에 대해 Prompt를 생성할 수 있는 Class를 정의한다.**

- Data의 종류에 맞는 Prompt 생성하는 함수 

- Data의 특징에 따라 특정 텍스트를 Cutting하는 등의 작업

- 추가로, 아래 코드에서 **Training prompt에는 가격 정보가 보이도록 구현**하고, **Testing prompt에는 가격 정보가 보이지않고 모델이 예측할 수 있도록 구현**한 것을 확인할 수 있다.


```python
# Amazon의 판매 데이터를 예시로 한 Item Class의 예시를 확인해보자.
from typing import Optional
from transformers import AutoTokenizer
import re

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

MIN_TOKENS = 150 # Any less than this, and we don't have enough useful content
MAX_TOKENS = 160 # Truncate after this many tokens. Then after adding in prompt text, we will get to around 180 tokens

MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price
    """
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    title: str
    price: float
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None
    include = False

    def __init__(self, data, price):
        self.title = data['title']
        self.price = price
        self.parse(data)

    def scrub(self, stuff):
        """
        Clean up the provided text by removing unnecessary characters and whitespace
        Also remove words that are 7+ chars and contain numbers, as these are likely irrelevant product numbers
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(' ')
        # 아마존의 설명서에는 숫자가 많음
        # Token은 정수 ID를 사용하기 때문에, 이를 혼란하지 않도록 숫자를 무시하라고 추가
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        """
        Parse this datapoint and if it fits within the allowed Token range,
        then set include to True
        """
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            # 여기서 Max_token 같은 부분이 Hyperparameter에 해당된다.
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Return a prompt suitable for testing, with the actual price removed
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Return a String version of this Item
        """
        return f"<{self.title} = ${self.price}>"
```

## Day2

**LLM을 Training하고 Applying하기 위해 5가지 단계가 필요하다.**

1. **Understand**: Project에 대한 이해를 바탕으로 적합한 품질의 Data를 선택

2. **Prepare**: LLM Model을 비교하고 Data Processing

3. **Select**: LLM 선택하고 Training

4. **Customize**: Prompting, RAG, Fine-tuning을 통해 성능 개선

      - **Prompting**: **비용이 적고 구현하기 쉬우며** 성능 향상이 쉽게 됨. / **Token 개수에 의해 제한**이 되며 Token이 많아질수록 느리고 비싸짐
      - **RAG**: **확장 가능성, 효율적, 적은 데이터로 정확성 개선** 가능 / 구현이 어렵고 **최신 데이터를 유지**해야 한다.
      - **Fine-tuning**: 특정 프로젝트에 **적합한 심화 지식**을 모델이 학습하도록 할 수 있음 / 구현이 어렵고 데이터가 많이 필요하며, **Training하는데 시간이 많이 든다.**

5. **Productionize**: 모델과 Platform 사이에 API를 결정, 지속적으로 모델 체크

Exercise에서는 **Data를 Load하기 위한 Class를 따로 정의했다.**

여기선, `load_dataset()`을 이용해 load한 dataset을 다루는 함수에 대해서만 알아보자.

```python
# 주어진 범위 내의 Data만 선택
dataset.select(range(i, min(i + CHUNK_SIZE, size)))

# ProcessPoolExecutor: 데이터셋 다운로드를 병렬로 처리
chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
with ProcessPoolExecutor(max_workers=workers) as pool:
    for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
        results.extend(batch)
```

이후, 데이터를 **Curated**하여 **Training에 안정성을 부여할 필요**가 있다.

- Exercise에서는 여러 Category 중 `Automative` Category의 데이터 개수가 비교적 많았다.

- **Category별 데이터 분포를 균일하게 유지하는 것이 Training에서의 성능에 도움**이 되기 때문에, Automative의 데이터 비중을 비교적 줄인 새로운 데이터셋을 만드는 작업이 필요하다.

```python
# defaultdict(list): dictionary 기본 value를 list 형태로 고정 
slots = defaultdict(list)  
for item in items:
    slots[round(item.price)].append(item)

np.random.seed(42)
random.seed(42)
sample = []
for i in range(1, 1000):
    slot = slots[i]
    # 가격이 240$ 이상이면 그냥 추가
    if i>=240:
        sample.extend(slot)
    # 가격이 1~239$인데 해당 물품 개수가 1200개 이하이면 그냥 추가
    elif len(slot) <= 1200:
        sample.extend(slot)
    # 가격이 1~239$인데 해당 물품 개수가 1200개 이상이면 제한을 둠
    else:
        # 랜덤으로 1200개를 뽑기 위한 가중치 부여
        # Automative data 개수가 많기 때문에 가중치는 1 부여
        weights = np.array([1 if item.category=='Automotive' else 5 for item in slot])
        weights = weights / np.sum(weights)
        # 가중치에 맞게 slot 중 1200개만 뽑음
        selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
        selected = [slot[i] for i in selected_indices]
        sample.extend(selected)
```
이후, Training과 Testing을 위해선 Dataset을 나누어야 한다.

- 일반적으로, **Testing Data가 전체의 5~10%를** 차지하도록 한다.

```python
random.seed(42)
random.shuffle(sample)
train = sample[:25_000]
test = sample[25_000:27_000]
```

마지막으로 작업한 Dataset을 **Huggingface Hub에 올릴 수 있다.**

```python
train_prompts = [item.prompt for item in train]
train_prices = [item.price for item in train]
test_prompts = [item.test_prompt() for item in test]
test_prices = [item.price for item in test]

# Dataset.from_dict: 사용자가 지정한 Key에 맞추어 List value를 Table 형태로 생성함
# 이때, 모든 Key의 List의 길이가 동일해야 한다. 
train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})

# DatasetDict: train/test를 key로 Dataset을 생성
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

DATASET_NAME = "username/dataset_name"
# push_to_hub: hub에 push
dataset.push_to_hub(DATASET_NAME, private=True)

# Dataset을 Local에 .pkl 파일 형식으로 저장할 수도 있다.
with open('train_lite.pkl', 'wb') as file:
    pickle.dump(train, file)

with open('test_lite.pkl', 'wb') as file:
    pickle.dump(test, file)
```


## Day3

**Baseline**: 비교 기준이 되는 기본 모델·방법·프롬프트 설정

- Baseline을 기준으로 두고, 구현하면 구현이 빨라진다.
- Benchmark를 기준으로 모델을 비교, 평가한다.
- 같은 환경에서 모델의 성능을 비교하고 측정 수 있다.
- **LLM은 반드시 Baseline의 성능보다 좋게 나와야 사용할 가치가 있어진다.**

**Traditional MLs은 LLM의 Baseline으로 좋다.**

이번 강의에서는 **Baseline으로 사용하기 위한 Traditional ML을 구현하는 방법**에 대해 공부한다.

여러 **Baseline을 Test하기 위해** 어떤 함수를 Parameter로 받아 실행할 수 있도록 해주는 `Tester Class`를 구현한다.

- 실제로 이렇게 **Wrapping하는 Test Class를 구현하는 것은 중요하다.**

- 하나의 class로 여러 개의 Baseline을 Test해볼 수 있도록 도와준다.

```python
class Tester:
   def __init__(self, predictor, title=None, data=test, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
       

   def run():
      self.predictor()

   @classmethod
   def test(cls, fcn):
      # cls(fnc)하면 __init__(fcn)을 호출하여 새로운 Class 객체 생성
      cls(fnc).run()
```

이제 **각 Baseline 함수를 구현하여, Tester class를 이용하여 Test**하면 된다. 

사용한 방법들은 아래와 같다.

- **Random**하게 예측하는 방법

- 항상 같은 **상수값으로 예측**하는 방법

- **Feature Engineering**을 이용하여 **가격와 가장 관련이 높은 Feature를 사용하여 Lienar Regression으로 예측**하는 방법

- **Bag-of-Words**로 Document를 Vector로 만들어 **Linear Regression으로 가격을 예측**하는 방법

- **Word2Vec**로 Document를 Vector로 만들어 **Linear Regression으로 가격을 예측**하는 방법

- **Support Vector Machines**을 이용하여 예측하는 방법

- **Random Forest**를 이용하여 예측하는 방법

 각 **Baseline**의 예측에 대해 **평균을 내고 그래프로 시각화**하여 결과를 확인했다. 

 
 ## Day4

 **Frontier LLM Model**에 Test data를 주고 결과를 살펴보자.

**Day3**에서 사용한 Tester class를 동일하게 사용하고, 기존에 **Frontier model을 다루었던 것과 동일하게 함수를 구현**한다.

 ```python
def get_price(s):
    s = s.replace('$','').replace(',','')
    # 가격에 해당하는 숫자만 Return 되도록 한다.
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0

def gpt_4o_mini(item):
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages_for(item),
        # 같은 질문에는 같은 대답을 주기를 강제
        seed=42,
        max_tokens=5
    )
    reply = response.choices[0].message.content
    return get_price(reply)

Tester.test(gpt_4o_mini, test)
```

## Day5

**Frontier Model을 Fine-tuning**

1. Training Dataset을 jsonl format으로 생성하고 OpenAI에 업로드

2. Training / Validation loss가 감소하게 Training

3. 결과를 확인하고 (1) ~ (2) 과정을 반복

**OpenAI는 이미 많은 데이터로 학습을 시켰기 때문에, 공식적으로 Fine-tuning을 위해선 50 ~ 100개의 예시만 Training에 활용하는 것을 추천한다.**

가장 먼저 해야할 작업은 **jsonl** 형식의 파일로 만드는 것이다.

- **jsonl**: OpenAI가 기대하는 형식으로 각 행이 하나의 JSON 객체을 가지는 파일

```python
def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        # json -> str
        messages_str = json.dumps(messages)
        # json 형식으로 맞추고 각 행에 넣음
        result += '{"messages": ' + messages_str +'}\n'
    return result.strip()

# jsonl 파일 생성
def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)

write_jsonl(fine_tune_train, "fine_tune_train.jsonl")

with open("fine_tune_train.jsonl", "rb") as f:
    # fine-tuning 목적으로 파일 생성
    train_file = openai.files.create(file=f, purpose="fine-tune")
```

이제 **OpenAI Fine-Tuning에 대해 알아보자.**

```python
# Fine-tuning 작업 생성
# 이 함수를 통해 실질적으로 Training을 시작한다.
openai.fine_tuning.jobs.create(
      training_file=train_file.id,
      validation_file=validation_file.id,
      model="gpt-4o-mini-2024-07-18",
      seed=42,
      hyperparameters={"n_epochs": 1},
      # 외부 연동 설정: Weights & Biases(W&B)로 파인튜닝 로그/지표를 보
      integrations = [wandb_integration],
      # 모델이 이름에 붙이는 접미사
      suffix="pricer"
)

# limit=의 개수만큼의 Page
# Job 객체를 반환하기 위해서는 .data[0]까지 사용해야 한다.
openai.fine_tuning.jobs.list(limit=1).data[0]

# data: job 객체를 반환
job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id
# job_id에 해당되는 job에 대한 세부 정보가 반환됨. (Token 개수, 작업 상태 등) 
openai.fine_tuning.jobs.retrieve(job_id)

# 이벤트 Log: loss, checkpoint, complete message 등이 출력됨
# 해당 Job에 해당하는 Event log를 List 형태로 반환한다.'
# .data를 붙이는 이유는 Job 객체로로 반환받기 위함이다.
openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data
```

이후 Training을 시킬 수 있다.

Training에서 **Loss, system log, Weight 등을 시각적으로 확인할 수 있도록** **Weights & Biases**를 이용할 수 있다.

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# Log in to Weights & Biases.
wandb.login()
# Sync the fine-tuning job with Weights & Biases.
# job_id에 해당하는 작업을 시각화하도록 연결한다.
WandbLogger.sync(fine_tune_job_id=job_id, project=project_name)
```

Training이 완료된 이후에는 fine-tuning이 완료된 모델을 사용할 수 있다.

```python
# job_id에 해당하는 Model 이름을 불러온다.
fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model

response = openai.chat.completions.create(
        # 여기에 Fine-tuned model을 사용해서 fine-tuned model을 이용할 수 있다.
        model=fine_tuned_model_name,
        messages=messages_for(item),
        seed=42,
        max_tokens=7
)
```

이렇게 내가 **원하는 데이터셋을 이용하여 Fine-tuning 하더라도 성능이 그렇게 좋지 않은 경우**가 많다.

**성능이 좋지 않은 여러 가지 이유**가 있다.

1. Frontier Model은 애초에 굉장히 많은 데이터로 Training 되었기 때문에, 적은 Data로 Fine-tuning을 하더라도 해당 Domain에 대한 전문성이 나타나지 않을 수 있다.

2. Model이 이전 Training한 Memory를 잊어버리며 Fine-tuning한 결과를 까먹을 수도 있다.

3. 가격 예측과 같이 문제 상황이 명확한 경우에는 시간과 비용 측면에서 Fine-tuning이 오히려 좋지 않을 수 있다.

**Fine-Tuning이 일반적으로 좋은 상황은 아래와 같다.**

1. **프롬프트만으로는 안 되는 ‘스타일/톤’ 고정**

2. **특정 출력 유형의 ‘재현성(일관성)’ 향상**

3. **복잡한 지시 불이행 보정**

4. **Edge case(희귀 패턴) 처리**

5. **Prompt로 규정하기 어려운 ‘새로운 기술/과업’ 습득**

즉, **특수한 작업이거나 Model의 답변의 형식을 제한할 필요**가 있을 때 Fine-tuning을 사용하는 것이 좋다.

일반적으로 **Data의 개수가 부족하면 Traditional ML 방법**이 오히려 더 적합하다.
