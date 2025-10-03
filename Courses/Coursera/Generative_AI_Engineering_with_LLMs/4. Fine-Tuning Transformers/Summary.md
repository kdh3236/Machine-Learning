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
 
## Pre-training Implementation with HuggingFace

BERT Model을 Pre-tuning하기 전에, **Tokenizer도 내 데이터에 맞게 Fine-tuning** 할 수 있다.

```python
# Pretrained tokenizer를 불러온다.
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# 미리 구현한 Batch iterator를 이용하여 Tokenizer를 Fine-tuning
# vocab_size는 일반적으로 Pre-trained tokenizer와 일치하는 것을 추천한다.
bert_tokenizer = bert_tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=30522)
```

BERT를 **Pre-Training**하기 위해서 **Config**를 정의해야 한다.

- Config는 Model의 구조를 결정한다.
- 실습에선 `BertConfig`와 `BertForMaskedLM`을 이용한다.

```python
from transformers import BertConfig,BertForMaskedLM

config = BertConfig(
    vocab_size=len(bert_tokenizer.get_vocab()),  # Specify the vocabulary size(Make sure this number equals the vocab_size of the tokenizer)
    hidden_size=768,  # Set the hidden size
    num_hidden_layers=12,  # Transformer block의 개수
    num_attention_heads=12,  # Set the number of attention heads
    intermediate_size=3072,  # Feed forward layer의 size
)

model = BertForMaskedLM(config)
```

Python의 `datasets` Package에는 **Dataset에 Tokenization 함수를 적용**할 수 있는 `map()` 함수가 구현되어져 있다.

```python
from datasets import load_dataset

def tokenize_function(examples):
    return bert_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# 파이썬 내장 map() 함수와 동작은 동일하다.
# batched = True: Batch dim을 첫 번째 차원으로 설정한다.
# remove_columns=["text"]를 통해 토큰 ID로 대체된 텍스트 열을 제거한다.
# 결과로 input_ids, token_type_ids, attention_mask와 같은 모델 입력에 필요한 열로 대체된 새로운 데이터셋이 된다. 
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 원하는 범위 내의 Index에 해당되는 Data를 모아 새로운 Dataset 객체로 만든다.
# Slicing과 다르게 객체로 반환한다.
dataset.select([i for i in range(1000)])
```

LLM의 Pre-training에서는 MLM과 NSP를 이용한다. HuggingFace에서는 이를 위해서 **Training 중에 MLM을 위해 랜덤으로 마스킹을 지원하고 Tokenization까지 수행**하는 함수를 지원한다.

```python
from transformers import DataCollatorForLanguageModeling

# bert_tokenizer를 통해 tokenization
# mlm=True, mlm_probability=0.15: 15% 확률로 일부 Text가 Masking 된다.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=bert_tokenizer, mlm=True, mlm_probability=0.15
)
```

이제 HuggingFace의 `Trainer`와 `TrainingArguments`를 이용하여 쉽게 Training 할 수 있다.

```python
training_args = TrainingArguments(
    output_dir="./trained_model",  # Specify the output directory for the trained model
    overwrite_output_dir=True, # 덮어씀
    do_eval=True, 
    evaluation_strategy="epoch", # Epoch마다 Evaluation
    learning_rate=5e-5,
    num_train_epochs=10,  # Specify the number of training epochs
    per_device_train_batch_size=2,  # Set the batch size for training
    save_total_limit=2,  # Limit the total number of saved checkpoints
    logging_steps = 20
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Start the pre-training
trainer.train()

# Evaluation
eval_results = trainer.evaluate()
```

## Fine-tuning

Neural Network를 처음부터 Training하는 것은 비용적으로 비싸기 때문에 사용하는 방법

- 시간과 비용을 아끼도록 해준다.

LLM을 Fine-tuning하는 방법에는 **세가지 방법**이 있다.

- **Self-supervised fine-tuning**: 마스킹된 단어를 예측하는 등

  - Pre-training과 비슷하다. 
- **Supervised fune-tuning**: Label을 예측하도록 하는 방법
- **Reinforcement fine-tuning**: 사람의 피드백을 통해 학습하는 방법

  - RLHF
  - Direct Preference Optimization (DPO)
 
### STF Trainer 

> **Supervised Fine-Tuning Trainer**

- Training 작업을 간결하고 자동화
- 에러가 더 적도록 보장
  
 
## Implemention of Fine-tuning

**PyTorch**만 이용하여 Fine-tuning하기 위해선 모델을 Load하고 `required_grad`를 조정하여 원하는 Parameter만 조정하도록 할 수 있다.

- Model 위에 FC Layer를 올려 FC Layer만 Training하는 방법
- FC Layer와 함께 중요한 Parameter만 조정하는 방법
- 모델 전체 Parameter를 조정하는 방법

**Training 대상이 되는 Parameter 개수가 많아질수록 시간과 비용이 많아지나, 성능은 높아진다.**

- 성능과 비용은 Trade-off이고, 상황에 맞추어 선택해야 한다.

**HuggingFace**를 이용하면 Fine-tuning이 비교적 편리하다.

먼저 `datasets` package를 통해 dataset을 load하고 수정할 수 있다.

```python
from datasets import load_dataset

# tokenization 후 Text는 필요없기 때문에 제거rainer.train()

# Evaluation
eval_results = trainer.evaluate()
```

## Fine-tuning

Neural Network를 처음부터 Training하는 것은 비용적으로 비싸기 때문에 사용하는 방법

- 시간과 비용을 아끼도록 해준다.

LLM을 Fine-tuning하는 방법에는 **세가지 방법**이 있다.

- **Self-supervised fine-tuning**: 마스킹된 단어를 예측하는 등

  - Pre-training과 비슷하다. 
- **Supervised fune-tuning**: Label을 예측하도록 하는 방법
- **Reinforcement fine-tuning**: 사람의 피드백을 통해 학습하는 방법

  - RLHF
  - Direct Preference Optimization (DPO)
  
## Implemention of Fine-tuning 

**PyTorch**만 이용하여 Fine-tuning하기 위해선 모델을 Load하고 `required_grad`를 조정하여 원하는 Parameter만 조정하도록 할 수 있다.

- Model 위에 FC Layer를 올려 FC Layer만 Training하는 방법
- FC Layer와 함께 중요한 Parameter만 조정하는 방법
- 모델 전체 Parameter를 조정하는 방법

**Training 대상이 되는 Parameter 개수가 많아질수록 시간과 비용이 많아지나, 성능은 높아진다.**

- 성능과 비용은 Trade-off이고, 상황에 맞추어 선택해야 한다.

**Classifier**만 Unfreeze해서 Training하는 방법은 아래와 같이 구현할 수 있다.

```python
for param in model.parameters():
    param.requires_grad = False

dim=model.classifier.in_features  # Load한 model 내부에 Classifier만 가져옴
model_fine2.classifier = nn.Linear(dim, 2)
```

하지만 **PyTorch**만을 이용하면, 모델 구성 + 데이터 전처리 + Training 과정을 모두 코딩해야 한다.

**HuggingFace**를 이용하면 Fine-tuning이 비교적 편리하다.

- 데이터 처리, 모델 Load, Training 과정 모두 라이브러리 함수를 사용할 수 있다.

먼저 `datasets` package를 통해 **dataset을 load하고 수정**할 수 있다.

```python
from datasets import load_dataset

# tokenization 후 Text는 필요없기 때문에 제거
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Rename 
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set the format of the dataset to return PyTorch tensors instead of lists
tokenized_datasets.set_format("torch")
```

이후, **dataset에 맞는 Dataloader 객체를 생성하고 모델은 Load**한다.

```python
# Create a training data loader
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=2)

# Create an evaluation data loader
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=2)

# Classification 목적으로 사용하는 모델을 Load할 때, 분류하고자 하는 Label의 개수에 맞춰
# num_labels를 수정해야 한다.
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

마지막으로 `Trainer`를 통해 Training할 수 있다.

- 그러나 이 방법은 Model의 모든 Parameter를 조정하기 때문에 비용과 시간이 많이 든다.

**STF Trainer**를 이용하여 Training할 수도 있다. 

- **Supervised Fine-Tuning Trainer**
- Training 작업을 간결하고 자동화
- 에러가 더 적도록 보장

```python
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# 모델이 학습할 데이터 형식 지정
instruction_template = "### Human:" # 사용자의 질문 시작 Token
response_template = "### Assistant:" # 정답 부분 시작 Token

# 데이터 형식에 맞추어 Tokenization
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

# dataset_text_field와 data_collator 등을 통해 추가적인 기능을 제공하는 Trainer
trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text", # dataset의 "text" key에 대응되는 부분만 사용
    data_collator=collator, # collator 함수 이용
)
```

# Parameter Efficient Fine Tuning (PEFT)

> **업데이트되는 Parameter의 개수를 줄여 비용과 시간을 아낄 수 있도록 하는 방법**

세부적으로 세 가지 방법으로 나뉜다.

## 1. Selective Fine-Tuning

> **Network의 일부 Sub-layers만 Training 하는 방법**

- 그러나 Pre-trained Transformer의 경우 **Parameter의 개수가 굉장히 많기 때문에 효과적이지 않다.**

## 2. Additive Fine-Tuning

> **기존 Model의 Parameter는 냅두고, 추가적인 Layer나 Adapter를 추가하고 추가된 부분만 Training하는 방법**

- Transformer의 경우에는 Encoder에서 Feed forward 다음에 Adapter를 추가한다.
- 이때, Adapter는 Dimension을 줄이는 역할을 한다.
- Transformer는 **기존 Model의 Parameter를 통해 기본적인 언어에 대한 이해는 유지하면서, Adapter를 통해 Task에 대한 이해를 추가**한다.

## 3. Reparameterization Fine-Tuning

> **기존 Model에 추가적으로 Parameter만 추가하고 추가된 Parameter만 Training하는 방법**

두 가지 방법으로 나눌 수 있다.

- **Soft prompting**

  - Embedding된 **Prompt 앞에 모델이 학습할 수 있는 Parameter vector를 추가하고, 이를 Training**한다.

- **LoRA**

## LoRA: Low-Rank Adaptation 

> **Model에 기존 Weight에 기존 Weight보다 낮은 차원을 갖는 Parameter 집합을 추가하고, 추가된 Parameter만 Training하는 방법**

기본적인 아이디어는 다음과 같다.

- 대규모 사전 학습 모델을 특정 작업에 맞게 미세 조정할 때, **모델의 가중치 행렬이 변하는 양(ΔW)은 사실상 복잡한 고차원 행렬이 아니라, 소수의 중요한 변화 방향만을 담고 있는 저차원 행렬**로 표현될 수 있다."

- 수학적으로는 **고차원 행렬의 저랭크 근사**로 생각할 수 있고, **기본적인 언어 모델에서 특정 작업을 수행하기 위해서는 해당 Task를 수행할 수 있도록 몇 개의 중요한 방향으로만 수정**하면 된다는 생각에 기반한다.

- 원래 모델의 파라미터를 $W$라고 할 때, $W - (d, k)$라고 해보자. 이때, **LoRA는 $BA$ 행렬로 구성된다.**

    - $B - (d, r)$, $A - (r, d)$로 $BA - (d, d)$이다.
    - $B$를 통해 Downsampling하고 $A$를 통해 Upsampling한다.
    - **Fine-Tuning 시, $BA$ 행렬만 Training한다.**

- 원래의 행렬에서 **특정 Task를 수행하기 위한 Parameter 변화분을 $\nabla W = W - W^*$라고 할 때, $\nabla$ = $BA$로 근사할 수 있다.**

일반적으로 **모델의 원래 Weight와의 Matrix multiplication을 수행한 결과와 LoRA Adapter를 수행한 결과를 더한다.**

- 이때 $W + BA$에 BA의 영향력을 조절할 수 있도록 **Scailing constant**를 사용할 수 있다.

  - $W + \frac{\alpha}{r}BA$로 나타낼 수 있고, $r$은 rank, $\alpha$는 Hyperparameter이다. 

**Transformer**에서는 Key, Query, Value에 LoRA Adapter를 추가하는 것이 일반적이다.

LoRA에서 발전된 **두 가지 모델**이 있다.

- **QLoRA**: Quantization이 추가된 방법
- **DLoRA**: 추가되는 Parameter matrix의 Rank가 Input에 따라 변하는 방법


## Implementation of LoRA with PyTorch

기존 Transformer encoder model의 **linear layer를 Adapter로 변경하는 방법**을 살펴보자.

**Adapter**는 **Model dim -> Bottleneck -> Model dim**의 2-Layer 구조를 갖는다.

먼저 **Feature Adapter**는 입력 **X를 받아 Adapter를 적용한 것과 X를 더한다.**

이후, Transformer encoder의 모든 Linear layer를 Adapter로 변경한다.

```python
    def __init__(self, bottleneck_size=50, model_dim=100):
        super().__init__()
        self.bottleneck_transform = nn.Sequential(
            nn.Linear(model_dim, bottleneck_size),  # Down-project to a smaller dimension
            nn.ReLU(),                             # Apply non-linearity
            nn.Linear(bottleneck_size, model_dim)  # Up-project back to the original dimension
        )

    def forward(self, x):
        transformed_features = self.bottleneck_transform(x)  # Transform features through the bottleneck
        output_with_residual = transformed_features + x      # Add the residual connection
        return output_with_residual

class Adapted(nn.Module):
    def __init__(self, linear,bottleneck_size=None):
        super(Adapted, self).__init__()
        self.linear = linear
        model_dim = linear.out_features
        if bottleneck_size is None:
          bottleneck_size = model_dim//2   # Define default bottleneck size as half the model_dim

        self.adaptor = FeatureAdapter(bottleneck_size=bottleneck_size, model_dim=model_dim)

    def forward(self, x):
        # First, the input x is passed through the linear layer
        x=self.linear(x)
        # Then it's adapted using FeatureAdapter
        x= self.adaptor(x)
        return x
```

이후 반복문을 통해 Transformer Encoder의 **Linear layer를 Adapted로 바꾼다.**

**LoRA Class 자체를 구현할 수도 있다.**

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear.to(device)
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        ).to(device)

    def forward(self, x):
        # 결과를 더함
        return self.linear(x) + self.lora(x)
```

이후, Model의 특정 Layer를 `LinearWithLoRA`로 변경한다.

## Implementation of LoRA with HuggingFace

`peft` Package를 이용한다.

```python
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    task_type = TaskType.SEQ_CLS, # 감정 분류를 위한 작업임을 명시
    r = 8, # rank
    lora_alpha = 16, # For scailing 
    lora_dropout = 0.1,
    target_modules = ['q_lin', 'k_lin', 'v_lin'])

# 모델에 LoRA Adapter 추가
model = get_peft_model(model, lora_config)
```

이후, `Trainer`를 이용하여 Training 할 수 있다.

Custom task라면 TaskType을 명시해야 할 수도 있다.

## QLoRA

> **Numerical variable의 Precision을 줄여 Quantization하는 LoRA**

- **정확도와 성능을 일부분 포기하고 Memory 효율화**를 진행한다.
- 일반적으로 32 / 16 bit를 8 / 4 bit로 줄인다

    - QLoRA는 4 bit에 초점을 맞추며, **NormaFloat(NF4)를** 이용한다.
- Parameter를 저장하기 위한 Memory를 크게 줄일 수 있다.
