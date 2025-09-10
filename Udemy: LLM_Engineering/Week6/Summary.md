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


