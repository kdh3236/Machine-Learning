## Day1

전체 Model을 학습하는 것은 **Parameter 개수가 너무 많기 때문에 오래 걸린다.**

학습해야 할 Parameter의 개수를 크게 줄이기 위해서 **LoRA**를 사용한다.

**LoRA**의 동작 과정은 아래와 같다.

1. **Freeze the weights**
  
    - Forward Pass를 진행하고, 예측 결과를 확인

2. **전체 모델 구조에서 수정하고자 하는 Target Module을 설정**

3. **Lower dimension (Rank) Adaptor Matrices를 생성한다.**

    - **적은 개수의 Parameter**를 사용할 수 있도록 한다.
    - 정확하게는 **두 개의 LoRA Matrices**가 사용된다.
    - Backpropagation에서 **실제로 Update 되는 행렬**이다. 

4. **(2)에서의 Traget Module에 (3)에서의 Adaptor를 적용한다.**
    
    - Forward 과정에서 원래 Module의 출력값에 **LoRA Matrices를 이용한 계산값을** 더하여 Output으로 사용한다.

**QLoRA**는 **Quantization + LoRA**를 의미한다.

- 기본 모델의 Parameter의 개수는 유지하되, Precision을 줄인다.
- 32 bits -> 8 bits, 4 bits
- LoRA Metrices는 그대로 유지하는 경향이 있다.

**LoRA Fine-Tuning**을 위한 3가지 중요한 **Hyperparameter**에 대해 알아보자.

1. **R**: Low Rank Matrices의 Rank ,dimension

    - 일반적으로 8부터 시작하여 2배씩 늘리는게 좋다.
    - 너무 큰 R값은 의미가 없다.

2. **Alpha**: Scaling factor that multiplies the lower rank matrices

     - 원래 Model과 LoRA Matrices 사이에서 **어디에 중점**을 둘 지 정한다.
     - 일반적으로 **R값의 2배**를 사용한다.

3. **Target Modules**: Which layers of the neural network are adapted

      - 일반적으로 Attention head layers를 Target하는 경우가 많다.

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# Quantization 부분은 아래와 같이 진행할 수 있다.

# 8 bits로
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# 4 bits
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# LoRA
# PerfModel(): Base model 위에 LoRA Adaptor를 올린다.
# FINETUNED_MODEL: Weight를 Load할 Hub ID 또는 Local 경
fine_tuned_model = PeftModel.from_pretrained(BASE_MODEL, FINETUNED_MODEL)

# 원래는 처음부터 PeftModel()으로 불러오지 않고 model.save_pretrained(FONTUNED_MODEL)을
# 통해 Local이나 Hub에 저장해야 한다.
```
