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

## Day3

`ChatInterface(type="messages")`를 사용하면 **챗봇 형태의 결과물**을 얻을 수 있다.

```python
gr.ChatInterface(fn=chat, type="messages").launch()
```

**Q) 아래 형식의 Messages를 LLM이 어떻게 이해하는 것일까? (GPT를 예시로)**

[
    {"role": "system", "content": "system message here"},
    {"role": "user", "content": "first user prompt here"},
    {"role": "assistant", "content": "the assistant's response"},
    {"role": "user", "content": "the new user prompt"},
]

A) 위의 Messages를 **Token의 연속**으로 바꾼다. **System prompt에는 특별한 표시(구분)가 있는 Token**을 사용하여, Training을 통해 **System prompt에서 특별한 지시를 받았다면 다음 Token은 System prompt 따라야한다는 것을 학습**한다.


## Day4

**Tool**: LLM이 답만 만들어내는 것을 넘어서, **외부 기능·API·데이터 소스**를 **실행 시점(inference time)에 호출해서 일을 처리**하게 해주는 인터페이스

기본적으로, LLM에게 **사용자가 특정 동작을 요청할 때에는 외부 기능, API 또는 구현된 함수를 호출**하라고 Training 시킬 수 있다.

- 이때, 구현된 함수나 외부 기능에 대한 **Dictionary 형태의 정보을 LLM에 제공**해야 한다.

- 이 정보 또한, LLM에는 **Series of Token으로 변환**되어 전달된다.

**함수 이름, 역할, Parameter 종류와 개수가 명시 되어야 LLM이 정확하게 호출할 수 있기 때문에 중요하다.**  

```python
price_function = {
    # 함수명
    "name": "get_ticket_price",
    # 함수에 대한 설명
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        # Parameter를 Dictionary 형태로 정의해야 함. object->Json의 Object를 의미
        "type": "object",
        # 이 Dictionary의 Key가 Parameter이다.
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        # 반드시 포함되어야 하는 Key
        "required": ["destination_city"],
        # properties에 정의되지 않은 Parameter의 사용을 금지
        "additionalProperties": False
    }
}
```

위에서 정의한 Dictionary를 OpenAI의 `tools=` Argument에 넘겨서 LLM이 사용할 수 있도록 구현할 수 있다.

```python
tools = [{"type": "function", "function": price_function}]

response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
```


LLM의 `response.choices[0].finish_reason()`이 **tool_calls**와 같다면 이는 **LLM이 tool을 호출해야 함**을 의미한다. 

이때, `response.choices[0].message`에는 **호출하고자 하는 tool에 맞는 parameter가 적혀있다.**
- message로부터 parameter를 뽑아낼 수 있어야 한다.

`response.choices[0].message.tool_calls`는 **LLM이 요청한 tool과 해당하는 Parameter가 담긴 List**를 반환한다.

- 이 중 `response.choices[0].message.tool_calls.function.arguments`를 Dictionary 형태로 불러오고 정의된 Parameter에 맞추어 함수를 호출한다.

찾은 정보를 **role: tool**에 맞추어서 LLM에 전달한다.

이후 다시 LLM에 전달되는 **Message에는 LLM의 Tool 호출 요청과 호출 결과가 모두 포함**된다.

- **LLM은 상태를 따로 저장하지 않기** 때문에, **두 가지 모두 포함되어야지 기존에 자신이 Tool을 요청했고 답을 얻어온 사실을 바탕으로 사용자의 질문에 대한 답변**을 할 수 있기 때문이다.   

```python
def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        # 최종 응
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    # json -> dictionary
    arguments = json.loads(tool_call.function.arguments)
    # Model이 생성한 arguments에는 `destination_city` key에 대한 Value로 사용자가 원하는 City가 들어있
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        # dictionary -> json
        "content": json.dumps({"destination_city": city,"price": price}),
        # "tool_call_id": tool_call.id: 어떤 Tool을 호출한 것인지 식별
        "tool_call_id": tool_call.id
    }
    return response, city
```
