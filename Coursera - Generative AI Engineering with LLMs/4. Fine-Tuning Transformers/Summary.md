## HuggingFace

`pipeline()` API를 이용하여 쉽게 NLP Task를 처리할 수 있다.

```python
transformers.pipeline(
    task: str,
    model: Optional = None,
    config: Optional = None,
    tokenizer: Optional = None,
    feature_extractor: Optional = None,
    framework: Optional = None,
    revision: str = 'main',
    use_fast: bool = True,
    model_kwargs: Dict[str, Any] = None,
    **kwargs
)

ai = pipeline()
# Task에 맞는 결과를 생성
# Task 종류에 따라 결과를 생성할 때 넣는 Parameter 종류가 다르다.
result = ai(prompt)
```

- **task**: NLP Task로 정해진 작업명이 있다.

    - 'text-classification', 'text-generation', 'question-answering' 등
- **model**: Hub에서 인식할 수 있는 model identifier
- **tokenizer**: path to a directory, or a pre-loaded tokenizer instance인 string

`pipeline()`은 사용이 쉽고 효율적이여서 **간단한 구현이나 빠르게 제작이 필요할 때 효율적**이다.

그러나, **Custom task**를 수행해야 하거나, **성능이 중요한 경우**에는 사용하지 않는 것이 좋다.
