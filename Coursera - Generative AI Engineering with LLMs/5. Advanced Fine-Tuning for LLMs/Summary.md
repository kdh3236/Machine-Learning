# Instruction-tuning

> **Pre-training 이후, DPO/RLHF 적용 전에 적용하는 STF(Supervised Fine Tuning) 단계**

- Fine-Tuning의 일종

Instruction-Tuning은 세 가지로 구성되며 Input은 선택적으로 사용한다.

1. **Instruction**: 모델이 어떤 작업을 해야하는지 정한다.
2. **Input**: Context나 배경지식
3. **Output**: Response or Expectation

**모델이 Instuction + Input을 이해하고 원하는 Output을 생성해낼 수 있도록 하는 것이 목표이다.**

- GPT-like model이 다음 Token을 예측하는 것에는 능숙하지만, 질문에 대한 답을 하는 것에는 능숙하지 않다.
- 때문에, Instuction-Tuning을 통해 **모델이 질문에 대한 답을 하는 능력을 기르는 동시에 일반화 성능과 Task-specific model이 될 수 있도록 한다.**

모델의 학습을 돕기 위한 **Special Token**이 있다. **Training 시 Instruction, output을 구별하기 위해** **###Prompt, ###Response**를 사용한다.

- Task의 종류에 따라 ###User, ###Bot / ###Question, ###Answer이 되기도 한다.

## Instruction masking

Training을 위해 **Teacher forcing**을 사용할 때, **Instruction과 Output을 Concatenate**하고 **각 단계에서는 다음 Step의 Token을 예측하도록 Training**된다. 

- Concatenate할 때, Instruction과 Output을 구분하는 Special token이 필수적이다.

**Instruction Masking**은 **Model이 Output에 대해서만 예측할 수 있도록 Loss를 구할 때, Instuction을 Masking한다.**

- 모델의 Loss가 각 Token에서의 Cross Entropy Loss라고 할 때, **최종 Loss는 Output에 대한 Cross Entropy Loss의 합**이다.
- Overfitting을 방지하고, Response에만 집중할 수 있는 효과가 있다. 

**주의할 점**

1. 모델마다 Instruction, Output을 구분하는 Special token이 다를 수 있다.
2. 일부 라이브러리에는 Instruction-Tuning이 Default로 지정되어 있다.

**LoRA와 연결**

LoRA는 Fine-tuning을 위한 방법 중 하나로 메모리 용량과 연산량 감소에 효과적이다.

**Instruction-Tuning과 LoRA를 함께 사용하면, 원하는 Task에 맞는 모델을 효율적으로 Fine-Tuning 할 수 있다.**   

## Implementation of Insturction-Tuning with HuggingFace

먼저, Dataset의 Instuction과 Response에 Special Token을 추가하는 함수를 만들어야 한다.

- Special Token을 추가한 이후, Tokenization 한 뒤에 새롭게 Dataset을 생성한다.

```python
# Training을 위한 Formatting
def formatting_prompts_func(mydataset):
    output_texts = []
    for i in range(len(mydataset['instruction'])):
        text = (
            f"### Instruction:\n{mydataset['instruction'][i]}"
            f"\n\n### Response:\n{mydataset['output'][i]}</s>"
        )
        output_texts.append(text)
    return output_texts

# Testing을 위한 Formatting
def formatting_prompts_func_no_response(mydataset):
    output_texts = []
    for i in range(len(mydataset['instruction'])):
        text = (
            f"### Instruction:\n{mydataset['instruction'][i]}"
            f"\n\n### Response:\n"
        )
        output_texts.append(text)
    return output_texts
```

이후, HuggingFace의 `pipeline`을 이용할 수 있다.

- `text-generation` Task를 이용한다.

```python
gen_pipeline = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        batch_size=2,
                        max_length=50,
                        truncation=True, # max_length를 넘는 부분을 잘라냄
                        padding=False, 
                        return_full_text=False # 반환되는 문자열에 Prompt를 추가하지 않음, Reponse에 집중하도록 한다.
)
```
LoRA와 연결하여 사용하기 위해 `LoraConfig()`를 이용한다.

```python
lora_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA
    lora_dropout=0.1,  # Dropout rate
    task_type=TaskType.CAUSAL_LM  # CausalLM(오토리그레시브 생성) 모델이면 CAUSAL_LM 사용
)

model = get_peft_model(model, lora_config)
```

이후, `SFTTrainer`를 통해 쉽게 Training 할 수 있다.

- `formatting_func`와 `data_collator`가 중요하다.

```python
# Model이 Loss 계산에 대상, 학습하고자 하는 부분의 시작 Prefix를 지정
response_template = "### Response:\n" 
# Instruction Maksing을 자동으로 수행해준다.
# Tokenizer를 입력받아 Token 단위로 Masking 적용 / Tokenization은 하지 않음
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = SFTConfig(
    output_dir="/tmp",
    num_train_epochs=10,
    save_strategy="epoch",
    fp16=True,
    per_device_train_batch_size=2,  # Reduce batch size
    per_device_eval_batch_size=2,  # Reduce batch size
    max_seq_length=1024,
    do_eval=True
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    formatting_func=formatting_prompts_func, # Dataset에 Formatting 적용
    args=training_args,
    packing=False,
    data_collator=collator, # Instruction masking 적용
)
```

# Reward Modeling

> **모델이 질문에 대해 생성한 답이 적절한지를 Score로 나타내는 방법**

이를 통해 모델이 **적절한 Response**를 생성하도록 하며 **Model optimization을 유도하고 User preference도 적용**할 수 있다.

모델에 주어진 Question을 $Q$, 모델이 생성한 Response를 $R$라고 하면 **Reward model**은 **Response가 적절한지 나타내기 위해 $P(Q, R)$를 Score로 사용**한다.

- 표현을 확률처럼 한거지 Score라고 보는 것이 맞다.

## Reward model training

먼저 같은 Query $Q$에 대해 같은 모델이 생성한 두 가지 응답에 대해 **사람이 자신의 선호도를 바탕으로 하여 직접** **좋은 응답 $R_1$과 나쁜 응답 $R_2**$으로 분리한다.

- 이 경우, $P(Q, R_1 | \theta)$ > $P(Q, R_2 | \theta)$로 나타낼 수 있다.

- $\theta$는 모델의 Parameter로, $R_i$가 모델에 의해 생성된 응답이기 때문에 사용된다.

모델은 아래의 Loss를 이용하여 Training하며 $P(Q, R_1 | \theta)$, $P(Q, R_2 | \theta)$를 생성할 수 있게 되고 **사람의 선호도를 학습**할 수 있게 된다.

**Reward model** 역시 **LoRA**와 함께 사용하여 **LoRA Adapter만 Training하여 메모리와 시간을 절약할 수 있다.**

### Reward model loss: Bradley-Terry model

먼저 좋은 응답의 Score와 나쁜 응답의 Score의 차이를 확인해보자.

- $P(Q, R_1 | \theta)$ - $P(Q, R_2 | \theta)$

이후, 위 수식에 Sigmoid 함수를 적용한다.

- $P(R_1 > R_2 | Q)$ = $\sigma(P(Q, R_1 | \theta)$ - $P(Q, R_2 | \theta))$

$P(R_1 > R_2 | Q)$에 -1을 곱하고 Log를 취하여 **Negative Log Likelihood Problem**으로 다룬다.

- $\theta^*$ = $argmin_\theta[-log \sigma(P(Q, R_1 | \theta)$ - $P(Q, R_2 | \theta))]$

- 위 수식을 한 Data에 대한 Loss로 사용하고 Batch 내에서 전부 더하여 최종 Loss로 사용한다.

- -1을 곱하고 Log를 취했기 때문에, **$P(R_1 > R_2)$를 최대화하는 것과 동일하다.**

- 좋은 응답을 더 좋게, 나쁜 응답을 더 나쁘게 만들어 **결정 경계를 확실히 한다.**

## Implementation of Reward modeling with HuggingFace

먼저, Dataset에서 미리 분류되어 있는 좋은 응답 (Chosen)과 나쁜 응답 (Reject)를 분리하여 각각 Dataset으로 만든다.

- 이후 각각을 Tokenization한다.

```python
get_res=lambda dataset,res:[  "\n\nHuman: "+prompt + "\n\nAssistant: "+resp for prompt, resp in zip(dataset["train"]["prompt"], dataset["train"][res])]

chosen_samples=get_res( dataset,'chosen')
rejected_samples=get_res( dataset,'rejected')
```

이후, Dataset preprocessing이 필요하다. **`inputs_id`와 `attention_mask`는 `RewardTrainer`에서 사실상 필수**이기 때문에 Chosen과 Rejected 각각에서 반드시 찾아서 사용해야 한다.

```python

def preprocess_function(examples):
    # Tokenize the 'prompt_chosen' text with truncation and padding to the maximum length
    tokenized_chosen = tokenizer(examples['prompt_chosen'], truncation=True, max_length=max_length, padding="max_length")
    
    # Tokenize the 'prompt_rejected' text with truncation and padding to the maximum length
    tokenized_rejected = tokenizer(examples['prompt_rejected'], truncation=True, max_length=max_length, padding="max_length")
    
    # Return the tokenized inputs as a dictionary
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],  # Token IDs for 'chosen' responses
        "attention_mask_chosen": tokenized_chosen["attention_mask"],  # Attention masks for 'chosen' responses
        "input_ids_rejected": tokenized_rejected["input_ids"],  # Token IDs for 'rejected' responses
        "attention_mask_rejected": tokenized_rejected["attention_mask"],  # Attention masks for 'rejected' responses
    }
```

이제 Training을 위해 `Trainer`와 `argumnet`를 설정한다.

- `RewardTrainer`를 사용한다.

이떄, **`LoraConfig`를 `RewardTrainer`에 함께 넘겨주어 LoRA까지 쉽게 구현**할 수 있다.

```python
# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=3,  
    num_train_epochs=3, 
    gradient_accumulation_steps=8,
    learning_rate=1.41e-5,
    output_dir="./model_output3",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["attn.c_attn", "attn.c_proj"]  # Target attention layers
)

# Initialize RewardTrainer
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    peft_config=peft_config,
)

trainer.train()
```
