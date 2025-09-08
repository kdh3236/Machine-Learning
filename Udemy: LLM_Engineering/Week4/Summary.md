## Day1

**목적에 맞는 모델을 선택하는 방법**

모델을 선택할 때, 기본적으로 아래의 것들을 고려해야 한다.

**Basic 1**

1. **Open-source vs Closed-source**
2. **Release Data**: 미자막 Training 날짜
3. **Parameters**: Model의 Power
4. **Training tokens**
5. **Context length**

**Basic 2**

1. **Inference cost**: API/구독 비용 등
2. **Training cost**
3. **Build cost**
4. **Time to Market**: 출시까지 걸리는 시간
5. **Rate limits**: 사용 한도
6. **Speed**
7. **Latency**
8. **License**

**The Chinchilla Scaling Law**: Parameter의 개수는 Training Token의 개수에 비례한다.

- Training Token의 개수가 Parameter 개수가 표현할 수 있는 양보다 많다면, Training Token을 늘리는 것은 의미가 없다.

- 예를 들어 Training Token을 두 배 늘린다면, 두 배 많은 Parameter가 필요하다.

**Benchmark**: 모델의 성능을 측정하는 평가 지표

가장 **자주 사용되는 7가자의 Benchmark**는 아래와 같다.

- ARC / DROP / HellaSwag / MMLU / TruthfulQA / Winograde / GSM8K

- 각각 다른 성능을 평가하기 위해 사용된다.

특별한 목적을 가진 **3개의 Benchmark**도 존재한다.

- ELO: Chess같은 게임이나 스포츠에서의 Chat을 평가
- HumanEval: Python coding 평가
- MultiP-LE: Broader coding을 평가

위의 Benchmark에 비해 **비교적 어려운 6개의 Benchmark**도 존재한다.

- GPQA: Graduate Tests
- BBHard: Future Capabilities
- Math lv 5: Math
- IFEval: Difficult Instructions
- MuSR: Multistep Soft Reasoning
- MMLU-PRO: Harder MMLU

Benchmark는 비교에 유용하지만, **아래와 같은 문제점**이 있다.

1. 하드웨어, 관점 등에 따라 결과값이 달라질 수 있다.

2. Scope가 굉장히 좁다.

3. Training Data가 유출된다.

4. Overfitting의 가능성이 있다.

5. Frontier Model은 자신이 Benchmark에 의해 평가당하는 것을 인지하기도 한다.

**Huggingface의 Leaderboard/LLM Benchmark**를 통해 여러 Model을 여러 Benchmark로 평가하고 순위를 매긴 결과를 살펴볼 수 있다.

- Model의 종류, Parameter의 개수, Benchmark 종류 등을 선택할 수 있다.
