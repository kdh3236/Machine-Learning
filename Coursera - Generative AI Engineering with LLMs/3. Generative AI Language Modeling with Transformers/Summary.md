# Positional Encoding

**문장을 구성하는 단어의 순서**는 문장의 의미에 중요한 영향을 끼친다.

- Transformer 같은 모델에서 **입력 문장의 순서 정보도 보존**하기 위해서 **Positonal Vector를** 사용한다.

Learnable Parameter가 아니라, **고정된 Positional Vector**로는 **Sin, Cos**을 사용한다.

**Sin, Cos**의 식은 아래와 같다.

- $PE[pos,2i]$ = $sin(\frac{pos​}{10000^{\frac{2i}{d_{model​}}​}})$
  
- $PE[pos,2i+1]$ = $cos(\frac{pos​}{10000^{\frac{2i}{d_{model​}}​}})$

- $pos$ = 각 단어의 위치 (0, 1,..., N)

- $i$ = Encoding Dimension이 m이라고 하면 **0 ~ m까지의 Dimension index**

- **0~m-1** 까지의 index를 같는 Array의 각 Element를 $PE$ 식에 맞춰서 채워넣는다.

**Leanable Parameter**로 Neural Network가 **Positional Encoding Vector 자체를 학습**하도록 구현할 수도 있다.

두 방법 중 하나로 구한 **Positional Vector**를 **문장을 Tokenizaion한 결과에서의 각 단어와 해당 위치의 Positional vector를 더하여** Input으로 사용한다.

# Segment Embedding

특정 단어가 **어떤 문맥, 화자, 타입** 등에 속하는 지를 나타낸 Vector

- **BERT** 등에서 주로 사용된다.

- Segment Embedding까지 사용하면 **Token + Positional Vector + Segment Vector**로 구성된다.
