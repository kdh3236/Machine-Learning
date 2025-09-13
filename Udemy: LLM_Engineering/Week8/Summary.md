## Day1

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

```

이후, **Class를 이용**하여 앱을 배포하는 과정을 알아보자.

- 파이썬 파일에 Class가 정의된다.



