## Day1

GPT와 OLLAMA에 이어서 **Claude API를 사용하는 방법**을 알아보자.

openai와 다르게 **max_token이 명시**되어야 하고, **system_prompt를 user_prompt와 구별**하여 입력해야 한다.

- max_token은 **출력 Token의 최대 개수**를 의미한다.

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

아래 코드처럼 **OpenAI Libaray를 이용**하여 Google API KEY를 사용할 수도 있다.

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

여러 명이 대화한다고 생각한다면, 단순히 system/user prompt로 나누는 것이 아니라 **assistant와 추가적인 user prompt**를 사용할 수 있다.

```
[
  {"role": "system", "content":},
  {"role": "user", "content":},
  {"role": "assistant", "content":},
  {"role": "user", "content":}
]
```

## Day2

**Gradio**: 머신러닝 및 딥러닝 모델을 쉽게 배포하고 웹 인터페이스를 생성할 수 있도록 도와주는 Python 라이브러리

```python
import gradio as gr

# fn: 함수, inputs, outputs: 입출력의 데이터 타입
# flagging_mode="never": 빈공간이 없도록 한다.
# share=True: 내 Local 주소뿐만 아니라, 다른 사람들에게 공유할 수 있는 URL 생성
gr.Interface(fn=함수명, inputs="textbox", outputs="textbox", flagging_mode="never").launch(share=True)

# 생성한 웹 UI를 Broswer로 바로 연다.
gr.Interface(fn=함수명, inputs="textbox", outputs="textbox", flagging_mode="never").launch(inbrowser=True)
```

**launch(share=True)를** 이용하여 실행하면, 아래와 같이 두 가지 웹 URL이 뜬다. 

- Running on local URL:  http://127.0.0.1:7861
- Running on public URL: https://c1f6ab5bdc2722c539.gradio.live

특이하게도, **Public URL에 들어가서 실행해도, Local을 거쳐서 실행**하게 된다.
- URL에서 함수를 실행하면, 함수의 실행 결과가 Local shell에 뜨게 된다.

아래와 같은 방법으로 **Dark Mode로 설정**할 수도 있다.

```python
force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never", js=force_dark_mode).launch()
```

Input와 output의 **줄 위치도** 정할 수 있고, **Labeling**도 할 수 있다.

- Input와 Output의 개수는 하나가 아니여도 된다. 여러개인 경우 List의 Element로 넘겨주면 된다.

```python
view = gr.Interface(
    fn=message_gpt,
    inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    flagging_mode="never"
)
view.launch()
```

추가로, `gr.Markdown`을 통해 **마크다운 형식**으로 답변을 받을 수도 있다.

`gr.Dropdown([선택지1, 선택지2])`를 통해 선택할 수 있도록 할 수도 있다.
