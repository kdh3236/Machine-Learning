## Day1

**Data Science를 위한 환경 설정**

1. Git Repository 생성하기
   
2. README.md 생성
  
    - Anaconda 환경 또는 Python pip 환경으로 세탕하면 좋음

3. OpenAI API Key 설정

4. .env file 생성

5. Jupyter lab 실행

**openAI API 설정**

OpenAI를 비롯한 다른 모델들은 system prompt와 user prompt를 받는 것을 기대함

- **System prompt**: 모델의 역할, 모델이 전체적으로 해야하는 작업 설명 
- **User prompt**: 제공할 파일, 텍스트 등의 구체적인 형식, 어떤 것을 중점적으로 확인할 지에 대한 구체적인 지시

``` python
# 아래 List of Dictionaty 형태로 Message 제공 
messages = [
    {"role": "system", "content": "You are a snarky assistant"},
    {"role": "user", "content": "What is 2 + 2?"}
]

# 아래 코드를 통해 채팅이 가능
# reponse_format을 통해 응답 형식을 지정할 수 있다.
# stream=True라면 Model의 대답이 Chunk 단위로 나누어 즉시 전송된다.
# temperture는 0~2 사이 실수값을 가지고, 2에 가까울수록 모델에 창의성이 부여된다.
response = openai.chat.completions.create(
   model="gpt-4o-mini",
   messages=messages,
   response_format={"type": json_object},
   stream=True,
   temperture=0.7
)
print(response.choices[0].message.content)
```

## Day2

**Frontier Model**

- NLP 같은 분야에서 좋은 성능을 보이는 large-scale AI model

**Closed-Source Frontier**

- GPT, Gemini, Perplexity 와 같은 비공개 소스 모델

**Open-source Frontier**

- Llama, Gemma와 같은 오픈 소스 모델


**Three ways to use models**

1. **Chat interface**: ChatGPT와 같은 사용자와 직접 상호작용하는 인터페이스

2. **Cloud APIs**: Cloud에 존재하는 모델 사용자와 Local에서 Code를 통해 상호작용

   - Cloud에서 모델을 관리하기 때문에 고가의 장비가 필요하지 않

   - 네트워크 지연, 호출 시마다 비용 부담, 평균 처리량 높음
  

3. **Direct inference**: API를 호출하여 직접 자신의 장비로 모델을 실행

   - 모델의 구조, 추론 그리고 하드웨어 등에 직접 관여할 수 있다.
  
   - 데이터가 다른 회사의 Cloud로 유출되지 않는다.
  

**Ollama를 이용하여 API 호출 비용 없이 대화하는 방법**
  
```python
# Ollama는 아래 URL에서 동작한다.
OLLAMA_API = "http://localhost:11434/api/chat"


# 1. 먼저 HTTP Call로 상호작용할 수 있다.
# 어떤 모델을 쓸 지, 어떤 대화를 할 지, 스트리밍을 할 지를 JSON 형식으로 전달
payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
print(response.json()['message']['content'])

# 2. OpenAI의 Library를 이용할 수 있다.
# Model은 ollama를 사용하면서 openAI의 문법을 그대로 사용하기 위함
from openai import OpenAI
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

response = ollama_via_openai.chat.completions.create(
    model=MODEL,
    messages=messages
)

print(response.choices[0].message.content)

# 3. ollama package 이용

# requests library를 이용해서 HTTP Call으로 하는 대신, Package를 이용할 수 있다.
import ollama

response = ollama.chat(model=MODEL, messages=messages)
print(response['message']['content'])
```

## Day3

**Frontier LLMs의 성능**

1. Synthesizing Information: 정보를 구조화해서 서치하고, 연구하여 요약하여 답한다.

2. Fleshing out a skeleton: 몇몇 간단한 노트로부터 구조화된 결과물을 낼 수 있다.

3. Coding

**Frontier Model의 한계**

1. 대부분의 특정 Domain에 대해 전문적이지 않다.

2. Recent Events에 즉각적으로 대응할 수 없다.

3. Some Mistakes

## Day4

LLM은 **Attention -> Transformer**의 등장으로 발전하기 시작했다.

LLM의 버전이 향상될수록 **Parameters의 개수가 증가**한다.

**Token**

- 초기 Neural Networks는 **Character 단위**로 Training 되었다.

   - 그리나, 이 방법은 각 문자들을 단어로 구성하고 연속적인 의미를 이해해야 하는 등 요구되는게 많았다.
 
- 다음으로 **Word 단위**로 Training을 하기 시작하였다.

   - 그러나, 한 Word가 문맥에 따라 다른 의미를 가질 수 있고, 장소나 이름 등의 모르는 단어에 대한 표시가 부족했다.
 
- 마지막으로 **Token**을 이용한다.

   - Token은 **정확히 하나의 단어**로 이루어질 수도 있고, **단어의 일부**로 이루어질 수도 있다.
 
   - 흔한 단어인 경우, 각 단어가 하나의 Token에 대응된다.
 
   - Token이 단어의 시작을 나타내는 경우, **leading space**를 갖는다.
 
   - 띄어쓰기 또한 의미 해석에 중요한 역할을 하기 때문이다. 
 
   - 중간부터 시작하는 Token은 가장 앞이 빈칸이 아니다.
 
**Rule-of-thumb for English Writing (GPT)**

- 1 token ~~ 4 Characters
- 1 token ~~ 0.75 words
- 1000 tokens ~~ 750 words

일반적으로 Symbol이나 수식을 표현하기 위해선 더 많은 Token이 필요하다.
  

**Context Window**: **모델이 한 번의 응답에서 고려할 수 있는 총 토큰 용량(입력+출력 합)**

- 다음 Token을 생성하기 위해 이전에 존재하는 Token의 개수

- ChatGPT의 경우에, 새로운 질문을 할 때 이전 모든 대화에 현재 질문이 추가되어 하나의 긴 Prompt가 생성된다.

- **지금까지의 모든 대화와 입력 등을 기반으로 다음 토큰을 예측 할 때까지의 모든 대화**를 Context Window라고 한다.

## Day5

**One-shot Prompting**: Prompt를 입력할 때, 예시도 함께 제시하는 방법

**System Prompt**를 통해 모델의 답변의 형식, 말투 등을 설정할 수 있다.

**Multi-shot Prompting**: Prompt에 대한 하나의 예시만 주는 것이 아니라, 여러 예시를 제공하는 방법

