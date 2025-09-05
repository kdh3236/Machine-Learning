## Day1

GPT와 OLLAMA에 이어서 **Claude API를 사용하는 방법**을 알아보자.

openai와 다르게 **max_token이 명시**되어야 하고, **system_prompt를 user_prompt와 구별**하여 입력해야 한다.

- max_token은 출력 Token의 최대 개수를 의미한다.

```python
message = claude.messages.create(
  model="claude-3-5-sonnet-20240620",
  max_tokens=200,
  temperture=0.7
  system=system_message,
  messages=[
    {"role": "user", "content": user_prompt},
  ],
)

print(message.content[0].text)
```

streaming이 필요한 경우에는 openai와 다르게, `messages.stream()`을 사용해야 한다.

```python
result = claude.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

with result as stream:
    for text in stream.text_stream:
            print(text, end="", flush=True)
```

마찬가지로, **Google API를 사용하는 방법**에 대해 알아보자.

모델 호출하는 데에는 **모델명과 System_prompt**만 입력하고, `generate_content()` 함수에 user_prompt를 전달하여 메세지를 생성한다.

```python
gemini = google.generativeai.GenerativeModel(
  model_name='gemini-1.5-flash',
  system_instruction=system_message
)

response = gemini.generate_content(user_prompt)
print(reponse.text)
```

아래 코드처럼 OpenAI Libaray를 이용하여 Google API KEY를 사용할 수도 있다.

```python
gemini_via_openai_client = OpenAI(
    api_key=google_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = gemini_via_openai_client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=prompts
)
print(response.choices[0].message.content)
```

여러 명이 대화한다고 생각한다면, 단순히 system/user prompt로 나누는 것이 아니라 assistant와 추가적인 user prompt를 사용할 수 있다.

```
[
  {"role": "system", "content":},
  {"role": "user", "content":},
  {"role": "assistant", "content":},
  {"role": "user", "content":}
]
```
