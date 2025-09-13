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

# 아래 Decorator로 정의되는 함수를 Modal의 기본 실행 단위로 설정
@app.function(image=image)

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

우리는 **LLM이 Prompt에 대해 응답하는 함수를** 정의하여 Modal의 실행 단위로 사용할 것이다.

