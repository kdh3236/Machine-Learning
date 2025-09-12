## Day1

전체 Model을 학습하는 것은 **Parameter 개수가 너무 많기 때문에 오래 걸린다.**

학습해야 할 Parameter의 개수를 크게 줄이기 위해서 **LoRA**를 사용한다.

**LoRA**의 동작 과정은 아래와 같다.

1. **Freeze the weights**
  - Forward Pass를 진행하고, 예측 결과를 확인

2. **전체 모델 구조에서 수정하고자 하는 Target Module을 설정**

3. **Lower dimension (Rank) Adaptor Matrices를 생성한다.**
  - 이 Layer는 적은 개수의 Parameter를 갖는다.
  - 정확하게는 **두 개의 LoRA Matrices**가 사용된다.
  - Backpropagation에서 **실제로 Update 되는 행렬**이다. 

4. **(2)에서의 Traget Module에 (3)에서의 Adaptor를 적용한다.**
  - Forward 과정에서 원래 Module의 출력값에 **LoRA Matrices를 이용한 계산값을** 더하여 Output으로 사용한다.
