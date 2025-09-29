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

# LLMs as Distributions

먼저, Input에 대해 특정 Output이 나올 확률을 표현할 수 있다.

- $P(W_{t+1}|W_t ... W_1)$

- Output에 대해 **Softmax**를 적용한다.

이후, Input($W_t ... W_1$)에 대해 **가능한 모든 Output(Vocab)에 대한 각각의 Softmax 확률을 계산**하고 Vocab에 대한 **각 단어에 대한 막대 그래프**로 나타내면 **Distribution**의 형태를 나타난다.

- 이 떄의 Distribution은 **Categorical distribution**이다.

모델의 Output을 위 Distribution에서의 Sampling으로 나타낼 수 있다.

## Temperature

`Temperature`를 $t$라고 하면 아래와 같이 변형된 Softmax 식을 사용한다.

- $\frac{exp(\frac{t}{s_i})}{\sum_iexp(\frac{t}{s_i})}$

그 결과, **표현되는 Distribution이 Normalization 된다.** 

- $t$의 값이 크면 클수록 **Randomness가 강해진다.**


## Top-K Sampling

먼저 적절한 `Temperature`을 이용하여 확률(Softmax) 계산을 한다.

이후, **확률이 가장 높은 K만 선택**한다.

마지막으로, **선택한 K의 확률 합이 1이 되도록 Normalization**한다.

# LLM에서의 Policy

> **생성 프로세스를 안내하여 모델이 다양한 텍스트 생성 경로를 탐색하고 더욱 다양하고 상황에 적합한 출력을 생성하도록 하는 것**

- 강화학습에서의 Policy와 어느 정도 비슷하다.

**Randomness**를 활용하여 모델이 더 창의적으로 답할 수 있도록 한다.

- **Temperature, Top-K** 등에 의해 Randomness가 부여된다.

LLMs에서는 강화학습과 다르게 **Model Parameter를 이용하여 샘플링**한다고 생각하면 된다.
- Tearch forcing만으로 Policy가 정의되고 이후 RLHF 등을 통해 Parameter를 조절한다.
- Policy = $\pi_\theta(y_t | x, y_{<t})$
  
### Rollouts

모델이 Policy에 따라 생각한 각 응답을 `Rollout`이라고 하며, 각 응답의 집합을 `Rollouts`라고 한다.

라이브러리에서는 Rollouts 저장 형식이 다르다.

- Response만 저장될 수도 있고 Query+Response가 저장될 수도 있다.

# Reinforcement Learning from Human Feedback (RLHF)

먼저 $r(X, Y)$로 표현되는 **Reward function**에 대해 살펴보자.

- **Input Query를 입력**으로 받고, **사용자의 피드백**을 제공한다.

- 보통, **Encoder를 사용한다.**

Query만 모아놓은 Rollout을 $X$, 각 Query를 $X_i$라고 하자.

한 Query에 대한 응답을 $Y_i$라고 하고, n번째 Query에 대한 Response는 $Y_{n, i}라고 하자.

이제 **Expected Reward를 공식화**할 수 있다.

- $E[r]$ ~ $\frac{N}{1}\sum_{n=1}{N}\sum_{k=1}{N}\frac{1}{K}r)(X_n, Y_{n, k})$ = $E_{X ~ D}E_{D_X}[r(X, Y)]$

- $N$: Query의 개수 / $n$: Individual query / $k$: Individual response

- $E_{X~D}$: Expectation over data distribution D / $E_{D_X}$: Expectation over model's response distribution for input X

**Monte-carlo method**에 의해 **Reward에 대한 Expectation을** 아래와 같이 정의할 수 있다.

- $E[r_Y]=E_{D_X}[r(X, Y)]=\sum_Yr(X, Y)p(Y|X)$

RLHF를 적용하는 과정은 다음과 같다.

1. Query를 Model에 넣어 Response를 생성한다.

    - 이때, Query $X$에 대응되는 Response $Y$를 Rollout한다.
    - Query + Response 형태
    - 여러개의 Response가 생성될 수 있다.
         
2. Query + Response를 Reward model에 넣어 Reward value를 얻는다.

3. Reward를 기반으로 Parameter $\theta$를 업데이트한다.

   
# Proximal Policy Optimization (PPO)

> **Policy gradient objective function을 Maximize하는 방법 중 하나**

**Policy gradient method**는 Policy $\pi_\theta$가 최대가 되도록 하는 $\theta$를 찾는 방법이다.

- 이를 위한 여러 방법으로 **PPO**, **Cliped surrogate objective** 등이 있다.

### Cliped surrogate objective

Policy를 너무 급격하게 바꾸면 학습이 망가지기 때문에 **너무 큰 변화는 잘라내는 역할**을 한다.

### PPO Objective function

PPO Objective function을 다루기 전에 **KL Divergence penalty**부터 알아야 한다.

- **KL Divergence penalty**: Update된 Policy가 기존 Policy에서 너무 멀어지지 않도록 Penalty를 주는 term이다.

추가로, **Policy $\pi_\theta$에서 Sampling된 결과는 Reward $r(X, Y)$를 최대화**해야 한다.
 
결과적으로 **Objective function**은 아래와 같이 구성된다.

- $J(θ)=E[r(X,Y)−βKL(π_θ​(⋅∣X)∥π_{prev}​(⋅∣X))$
- $β$는 Hyperparameter이다.

**KL Divergence penalty**를 제외하고 Policy 부분만 자세히 나타내면 다음과 같다.

- $E[r_y|\theta]=E_{Y \sim\mathcal \pi_\theta(Y|X)}[r(X, Y)]$

  - 한 Query에 대한 기댓값

- $E_{X \sim\mathcal D}[E_{Y \sim\mathcal \pi_\theta(Y|X)}[r(X, Y)]$

  - 전체 Query에 대한 기댓값

- $\pi_*(X, Y)=argmax_{\pi}E_{X \sim\mathcal D}[E_{Y \sim\mathcal \pi_\theta(Y|X)}[r(X, Y)]$ 

$E[r_y|\theta]$의 Joint expectation을 Maximizing 하는 대신, **Log-derivative trick**을 사용할 수 있다.

### Log-derivative trick

**Analytical Expectation**으로 변환하여 $\theta$에 대해 **직접적인 최적화**가 가능하도록 한다.

- $E[r_y|\theta]=\sum_Yr(X, Y)\pi_\theta(Y|X)$

- $\nabla E[r_y|\theta]=\sum_Yr(X, Y)\nabla_\theta\pi_\theta(Y|X)$

- $\nabla_\theta\pi_\theta(Y|X) = \nabla_\theta log(\pi_\theta(Y|X))\pi_\theta(Y|X)$


  - $∇logf=\frac{f}{∇f}$를 이용한다. 

- $E_{X \sim\mathcal D}[\nabla E[r_Y|\theta]] = \nabla_\theta E_{X \sim\mathcal D}[E[r_Y|\theta]]$

### Tips for training

1. Human feedback을 통해 model을 평가

2. Moderate $β$에서부터 시작

3. Temperature을 점점 증가시킴

## Implementation of PPO with HuggingFace

두 가지 모델의 각 응답을 비교하가 위해서 Model 두 개를 Load한다.

```python
from trl import PPOConfig, AutoModelCausalLMWithValueHead, PPOTrainer

config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5
)

model_1 = AutoModelForCausalLMWithValueHaed.from_pretrained(config.model_name)
model_2 = AutoModelForCausalLMWithValueHaed.from_pretrained(config.model_name)
```

`LengthSampler`를 이용하면 **Input을 범위 내의 임이의 길이로 잘라주어 일반화 능력이 향상**된다.

```python
input_min_text_length, input_max_text_length = 2, 8

# [2, 8]의 Uniform Distriburtion에서 Sampling 한 값을 Input size로 정한다.
# 정해진 Input size보다 짧으면 Padding, 길면 Truncate
input_size = LengthSampler(input_min_text_length, input_max_text_length)

def tokenize(sample):
        # [: input_size()]를 통해 짤라내도록 한다.
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
```

이후, `Collator function`을 정의해 Batch로 반환할 수 있도록 한다.

이제 `PPOTrainer`를 생성한다.

- Reward function으로는 Huggingface `pipeline`을 이용하여 **Output에서 'score' key를 이용**한다.
- 높은 Score는 모델에 대한 높은 신뢰도를 의미한다.

```python
ppo_trainer = PPOTrainer(
            config,
            model,
            tokenizer,
            dataset-dataset,
            data_collator=collator
)
```

이후, Output을 생성하기 위해 **Argument**를 설정한다.

```python
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": 50256,
}

# Response 생성
response = ppo_trainer.generate(query, **generation_kwargs)
```

Output에서 **사용자가 원하는 결과**만을 뽑아서 `Reward`로 넘겨줄 수 있다.

- 이 예시에서는 POSITIVE인 부분만 뽑아서 모델에 넘겨준다.

```python
positive_scores = [
    item["score"]
    for output in pipe_outputs
    for item in output
    if item["label"] == "POSITIVE"
]
rewards = [torch.tensor(score) for score in positive_scores]
```

Query, response, reward를 사용하여 **모델 Parameter를 재조정**한다.

```python
# stats에는 Training에 대한 정보가 담겨 있다.
stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

# Training log
ppo_trainer.log_stats(stats, batch, rewards)

stat['ppo/loss/total'] # Loss
stat['ppo/mean_scores'] # Mean score
```

# DPO (Direct Preference Optimization)

**DPO**는 **사용자에게 여러 Output을 주고 어떤 것을 선호하는지 선택**하도록 한다.

**PPO**가 Reward를 통해 **Model에서 Parameter를 조정하라는 Signal을** 주는 대신에, **DPO**는 직접 **Model의 Parameter를 조정**한다.  

- **PPO**를 **Reward-based methods**라고 하고, **DPO**를 **Reward-free methods**라고 한다.

**DPO**는 3가지 구성요소를 갖는다.

- **Reward function**

  - **Encoder**를 이용한다.
    
- **Target decoder**

  - $\pi_\theta$로 표현된다.
  - **업데이트 대상**이 된다.
    
- **Reference model**

  - $\pi_{ref}$로 표현된다.
  - 미리 **SFT**를 통해 학습되고 고정되어 있다.
  - **Reference model에서 Target model이 크게 벗어나지 않도록 한다.**
 
**DPO**를 이용하여 Fine-tuning하기 위해서, **Positive와 Negative로 나누어진 데이터**를 수집해야 한다.
 
### Objective Function

- $\pi_{*} (X, Y)$ = $argmax_{\pi} {E_{Y \sim\mathcal \pi_\theta(Y|X)}[r(X, Y)]}$ - $\beta \cdot D_{KL}[\pi_\theta (Y|X) || \pi_{ref}(Y|X)]]$

**Opjective function**을 **Closed-form으로 만들고, Optimal solution**을 구하기 위해 간략화할 수 있다.

- $\pi_{*} (Y|X)$ = $argmax_{\pi_\theta} E_{X \sim\mathcal D, Y \sim\mathcal \pi_\theta(Y|X)}(ln(\frac{\pi_{ref}(Y|X)}{\pi_\theta(Y|X)}) - ln(Z(X)))$

- $\pi_{*} (Y|X)$ = $argmax_{\pi_\theta} E_{X\sim\mathcal D, Y \sim\mathcal \pi_\theta(Y|X)}(ln(\frac{\pi_{ref}(Y|X)}{\pi_\theta(Y|X)}))$

결국 **Optimal solution**을 추가하는 것은 $\theta_\pi$와 $\theta_r$의 **KL Divergence**를 구하는 것과 동일하다.

- 모델은 **DPO**를 통해 $\theta_\pi$와 $\theta_r$의 차이를 줄인다.

- **PPO**에서 Parameter 업데이트와 RL 단계가 나누어져 있던 것을 **하나의 Objective function으로 쉽게 표현할 수 있다.**

우리는 임의의 함수를 만들어 **Score**로 사용한다.

- 각 카테고리의 Score를 모두 더하면 1이 나와야 한다.

- 때문에 Policy에 맞는 **Score function은 $Z(x)$로 정규화**되어야 한다.

- $Z(x)$는 모든 Score의 합이고, 각 Score를 $Z(x)$로 나눈 값을 사용하면 된다.

### Loss

**Bradley-Terry**를 이용한다.

Loss = $-E_{(X, Y_W, Y_L) \sim\mathcal D} ln(\sigma(r(X, Y_W| \phi) - r(X, Y_L | \phi)))$

최종적으로 $-\sigma βln((r(X, Y_W| \phi)) - βln(r(X, Y_L | \phi)))$를 Loss로 사용한다.

- $r(X, Y_W)$ = $βln(\frac{\pi_{ref}(Y_W|X)}{\pi_\theta(Y_W|X)}) + βln(Z(X))$
- $r(X, Y_L)$ = $βln(\frac{\pi_{ref}(Y_L|X)}{\pi_\theta(Y_L|X)}) + βln(Z(X))$
- $Y_W$: 해당 답변을 선호하는 경우, $Y_L$: 해당 답변을 선호하지 않는 경우

결국, **선호하는 답변과 선호하지 않는 답변에 대한 Score의 차이가 커지도록 한다.**

## Implementation of DPO with HuggingFace

먼저, 데이터를 **Positive와 Negative로 나누기 위해 Processing한다.**

```python
# Define a function to process the data
def process(row):
    # delete unwanted columns
    del row["prompt_id"]
    del row["messages"]
    del row["score_chosen"]
    del row["score_rejected"]
    # retrieve the actual response text
    row["chosen"] = row["chosen"][-1]["content"]
    row["rejected"] = row["rejected"][-1]["content"]

    return row
```

LoRA Config를 정의한다.

```python
peft_config = LoraConfig(
        # The rank of the low-rank adaptation weights
        r=4,
        # The target modules to apply the low-rank adaptation to
        target_modules=['c_proj','c_attn'],
        # The task type for the low-rank adaptation
        task_type="CAUSAL_LM",
        # The scaling factor for the low-rank adaptation weights
        lora_alpha=8,
        # The dropout probability for the low-rank adaptation weights
        lora_dropout=0.1,
        # The bias mode for the low-rank adaptation
        bias="none",
)
```

`DPOConfig`를 설정하고 `DPOTrainer`를 지정한다.

```python
# DPO configuration
from peft import get_peft_model
training_args = DPOConfig(
    # beta: Hyperparameter, PPO의 Temperature와 비슷하게 동작한다. 
    beta=0.1,
    output_dir="dpo",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,
    logging_steps=10,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    evaluation_strategy="epoch",
    warmup_steps=2,
    fp16=False,
    save_steps=500,
    #save_total_limit=2,
    report_to='none'
)

trainer = DPOTrainer(
        model=model,
        # None이면 지금 모델과 동일한 Model의 가중치 초기화된 복사본을 만들어 사용
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        #max_prompt_length=512,
        max_length=512,
)
```

Fine-Tuning한 이후, Output을 생성하는 방법은 아래와 같다.

```python
generation_config = GenerationConfig(
        # Use sampling to generate diverse text
        do_sample=True,
        # Top-k sampling parameter
        top_k=1,
        # Temperature parameter to control the randomness of the generated text
        temperature=0.1,
        # Maximum number of new tokens to generate
        max_new_tokens=25,
        # Use the end-of-sequence token as the padding token
        pad_token_id=tokenizer.eos_token_id
    )

outputs = dpo_model.generate(**inputs, generation_config=generation_config)
inputs = tokenizer(PROMPT, return_tensors='pt')
```
