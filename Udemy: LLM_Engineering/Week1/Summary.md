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
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
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
