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

# Attention Mechanism

> 문장 중 **특정 한 부분에 집중**하는 방식으로 문장을 해석하는 방법

**Query, Key, Value**로 이루어져 있다.

- **Query**: 지금 원하는 정보
- **Key**: 검색에 도움이 되는 정보
- **Value**: 실제 찾고자 하는 정보

중요한 점은 **Key와 Value**에서 **같은 행은 같은 Token (Data)에 대한 정보**를 나타내야 한다는 점이다.

- Self-Attention이던 Cross-Attention이던 **Key, Value는 같은 X로부터 얻는 이유**이다.

- Query가 달라도, Key를 통해 Value에서 연관된 정보를 얻을 수 있도록 하기 위함이다.

**Query와 Key의 Inner Product**를 통해 원하는 정보를 탐색한다.

이후, **Softmax**를 사용하는데, 이는 **값의 범위를 제한**하기 위함이다.

Word Embedding 관점에서 기존에 학습하지 못 하던, **문맥적인 단어 간의 관계도 학습**할 수 있다.

Sequence Data를 처리할 때, 한 Token씩 따로 처리하지 않고, **하나의 matrix로 묶어서 처리**한다.

- Query, Key, Value 모두 동일한 방법을 사용한다.

## Self-Attention Mechanism

**Scaled dot-product 방식으로 설명한다.**

Input X를 **Tokenization + Embedding**한 이후에, **Query, Key, Value Matrix**와 곱하여 **Key, Query, Value**를 얻는다.

- **Query, Key, Value Matrix**는 Learnable Parameter이며, Bias term도 사용하기도 한다.

이후, **Query와 Key를 Inner Product**하고 **첫 번째 Dimension의 제곱근**으로 나눈다.

- 나누는 이유는 Variance가 너무 커져 Score 값이 너무 커지는 것을 방지하기 위함이다.

- 결과로 나오는 행렬에 **Softmax Function**을 적용하고 이 행렬을 $A$라고 한다.

$A$를 Value와 Matrix Multiplication한 결과로 나오는 Matrix가 **Context Matrix**이다.

- Attention Score ($A$)를 봤을 때, Value에서 **특정 단어마다 집중할 위치가 표시**되기 때문이다.

이후, Output을 생성하기 위한 **Output Matrix**와 곱하고 **평균**내어 **MLP**에 넘겨주어 Classiciation이나 Prediction 등을 수행할 수도 있다.

## Multi-head Attention

각 **Head**는 **독립적인 Parameter를 사용하는 Self-Attention Module로** 이루어져 있고, **같은 Input(X)를** 사용하여 병렬적으로 처리하는 방법

결과로 나온 Head 개수만큼의 $H'$ Matrix를 전부 **Concatenate**하고, **Output Matrix를 통해 Output을 생성**한다. 

# Transformer

> **Self-Attnetion Module을 기본으로 사용하며, Encoder와 Decoder로 구성된다.**

먼저, **Encoder**부터 살펴보자.

1. Embedding된 Input에 Positional Vector를 더한다.
2. **Multi-head Self-Attention Module**에 (1)의 결과를 넘겨준다.

    - 일반적으로, 여기선 Masking을 사용하지 않는다.
    
3. Add and Normalization
4. FC Layer
5. Add and Normalization

**Decoder**는 아래와 같이 구성된다.

1. Decoder Input을 Embedding하고 Positional Vector를 더한다.
2. **Masked Multi-haed Self-Attention module**을 사용한다.

      - Decoder에서는 특정 단어 기준 미래 단어를 미리 알 수 있으면 안 된다.
  
3. Add and Normalization
4. Encoder의 Output을 Query로 이용하고, (2)에서의 출력을 Key, Value로 사용한 **Multi-head Cross-Attention Module**
5. Add and Normalization
6. FC Layer
7. Add and Normalization
8. Linear Layer
9. Softmax -> Output Probabilities

**Translation**의 경우, Encoder Input: Source Langauge / Decoder Input: Target Language로 구성된다.

일반적으로 **LLM**에서는 Encoder 입력 = 질문(소스) + (선택) 정보 컨텍스트 / Decoder 입력 = 정답의 앞부분(BOS/이전 생성 토큰) 으로 구성된다.

**Transformer Block**을 **여러 Layer로 구성**해서 더 복잡한 Sequence Data를 처리할 수도 있다.\

- 이 경우 **Layer의 개수도 Hyperparameter**이다.

Multi-haed의 **Head 개수도 중요한 Hyperparameter**이다.
