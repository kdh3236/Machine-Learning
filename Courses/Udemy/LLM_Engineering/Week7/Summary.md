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
# base_model:  AutoModelForCausalLM을 통해 Load된 Model object
# FINETUNED_MODEL: Weight를 Load할 Hub ID 또는 Local 경로, LoRA Weight가 불러와진다.
fine_tuned_model = PeftModel.from_pretrained(BASE_MODEL, FINETUNED_MODEL)

# 원래는 처음부터 PeftModel()으로 불러오지 않고 model.save_pretrained(FONTUNED_MODEL)을
# 통해 Local이나 Hub에 저장해야 한다.
```

## Day2

Fine-tuning을 할 Model을 선택할 때 고려할 점이 있다.

1. **Parameter의 개수**

2. **Base vs Instruct Varient**

    - **Base**: 기본 모델, Prompt와 대화를 바탕으로 다음 Token을 예측
    - **Instruct Varient**: 특정 포맷의 답변을 잘 하며, 지시를 비교적 잘 따른다.
  
## Day3 

**QLoRA**를 위해선 **두 개의 Hyperparameter**가 추가로 요구된다.

- **Dropout**
- **Quantization**

**Five Important Hyper-parameters for Training**

1. **Epochs**

2. **Batch Size**

3. **Learning Rate**

4. **Gradient Accumulation**: Training 속도를 높이는 방법

      - 미니 배치를 통해 구해진 gradient를 n-step동안 Global Gradients에 누적시킨 후, 한번에 업데이트
      - GPU 메모리 부족으로 인하여 큰 Batch size를 사용하지 못 할 때, 원하는 크기만큼의 Batch size를 사용할 수 있도록 해준다.

5. **Optimizers**

우리는 가격을 예측하는 LLM Model을 만들고 있다.

- **가격 (숫자)에 대해서만 예측**을 해야하기 떄문에 아래와 같은 사전 작업이 필요하다.

- 전체 Prompt를 넣으면 **상품 정보에 대한 답변이 나올 수 있다.**

- 우리는 Prompt에서 $ 기호 뒤에 나오는 정보 (가격)만 예측하기를 원한다. 

```python
from trl import DataCollatorForCompletionOnlyLM
# Response_template="~": "~" 뒤에 올 Token을예측하도록 한다.
response_template = "Price is $"
# "~" 뒤 Token만을 예측하도록 하기 위해, "Price is $" 이전 Prompt는 Masking 처리 해서 Loss 및 Backpropagation에서 제외한다.
# response_template 직후 구간만 정답 레이블로 둔다.
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
```

이후, **LoRA에 대한 hyperparameter 값을 먼저 정의한다.**

```python
from peft import LoraConfig

lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)
```

이후, **Training Hyperparameter 값도 정의한다.**

```python
from trl import SFTConfig

train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_strategy="no",
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    # Checkpoint 간격
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dataset_text_field="text",
    save_strategy="steps",
    hub_strategy="every_save",
    # Huggingface hub에 Push 하기 위함
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True
)
```

마지막으로 **Trainer**를 정의하고 Training을 진행할 수 있다. 

```python
from trl import SFTTrainer
from peft import LoraConfig

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    peft_config=lora_parameters,
    args=train_parameters,
    data_collator=collator
)

# Training 시작
fine_tuning.train()

# Push our fine-tuned model to Hugging Face
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
```

## Day4

**Training Loss**를 확인해보면 **각 Epoch 내에서는 Loss가 진동하다가, 각 Epoch의 마지막 부분에서는 Loss가 급감**하는 것을 확인할 수 있다.

- 이는 Epoch가 반복될수록 **Model은 같은 데이터를 반복해서 보게 되는 것이기 떄문이다.**
- 일종의 **Overfitting**이다.

Training 시 주의할 점은 **Gradient가 0이 되지 않아야 한다는 점이다.**

- Gradient가 0이 되면 **학습이 전혀 진행되지 않는다.**

Training 한 Model을 Huggingface에 올릴 때, **일정한 Training step마다 commit**하는 것이 좋다.

- Training_config의 `save_steps=`를 통해 가능하다.

## Day5

Training 과정에서 예측한 Token을 이용하여 Loss를 구하는 방법에 대해 알아보자.

- **예측한 Token이 맞을 확률**로 다루고 싶다.

- **Cross Entropy Loss**를 사용한다.

**Day4**에서 Huggingface에 올린 **Fine-tuned Model**을 불러오는 방법은 아래와 같다.

```python
from peft import PeftModel

FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

if REVISION:
  fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)
else:
  fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

# Inference
attention_mask = torch.ones(inputs.shape, device="cuda")
outputs = fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=3, num_return_sequences=1)
```
