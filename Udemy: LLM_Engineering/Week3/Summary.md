# Day1

Huggingface 및 Colab에 대한 Introduction

# Day2

Huggingface에는 **두 가지 API가** 존재한다.

1. **Pipelines**: 단순히 **원하는 작업에 대한 설명만 제공**하는 Higher level APIs

2. **Tokenizers and Models**: 모델에 대한 세부적인 부분까지 다룰 수 있는 Lower level APIs


**Pipeline**을 사용하는 것은 굉장히 간단하다.

일반적으로 pipeline() 내에 "" 안에는 **사용하고자 하는 목적이 정확히 명시**되어야 하고, Model을 지정하지 않는다면 **그 Task에 대한 Default model이 사용**된다.

- 반드시 **정해진 공식 Alias를** 사용해야한다.

- EX) sentiment-analysis

Pipeline Object를 생성한 이후에는 **주어진 Task에 맞는 Parameter를 제공하여, 결과를 생성**할 수 있다.

- 분류 계열: top_k, return_all_scores, truncation

- 생성 계열: max_new_tokens, do_sample, temperature

- EX) Classification이면 분류할 Class의 종류, Text Generator라면 최대/최소 길이 

```python
my_pipeline = pipeline("the task I want to do", model=, device=)

# 단순한 예시임
result = my_pipeline(my_input, max_tokens=, candidate_labels=)
```



