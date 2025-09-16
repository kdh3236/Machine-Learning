**Natural Language Processing (NLP)** 분야를 학습하기 위해 필요한 **Library**에 대해 알아보자.

먼저 **Huggingface**부터 살펴보자.

- **transformers**
- **AutoTokenizer**
- **AutoModelForSeq2SeqLM**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model 선택
model_name = "facebook/blenderbot-400M-distill"

# Model_name에 맞는 Seq2Seq Model을 찾아 반환한다.
# 이후, model.generate()로 응답을 생성할 수 있다.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Model에 맞는 Tokenizer를 찾아 반환
tokenizer = AutoTokenizer.from_pretrained(model_name)

### Tokenizer와 model을 이용하여 답변을 생성하는 과정은 아래와 같다.
input_text = input()
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
```
