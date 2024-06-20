

<!-- Provide a quick summary of what the model is/does. -->

It's a fine tuned version of Phi-2 model by Microsoft on Amod/mental_health_counseling_conversations.


## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
The above model, with applicable changes to the generation_config file passed to model.generate() fucntion can lead to generation of better resulst which could then be used for Health Counseling Chatbot dev.



## Bias, Risks, and Limitations

The model was developed as a proof-of-concept type hobby project and is not intended to be used without careful considerations of its implications.

[More Information Needed]


## How to Get Started with the Model

Use the code below to get started with the model.

```python

new_model = "YuvrajSingh9886/medicinal-QnA-phi2-custom"

prompt = "I have been feeling more and more down for over a month. I have started having trouble sleeping due to panic attacks, but they are almost never triggered by something that I know of."

tokens = tokenizer(f"### Question: {prompt}", return_tensors='pt').to('cuda')
tokenizer.pad_token = tokenizer.eos_token
outputs = model.generate(**tokens, max_new_tokens=1024, num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
                         )
print(tokenizer.batch_decode(outputs,skip_special_tokens=True)[0])

```

## Training Details

### Training Data


#### Hardware

Epcohs: 10
Hardware: (1) RTX 4090 (24GB VRAM) 48GB 8vCPU (RAM)
          Hard Disk: 40GB


[More Information Needed]

### Training Procedure

QLoRA was used for quantization purposes.

Phi-2 model from Huggingface with BitsandBytes support


#### Preprocessing [optional]

```python

def format_phi2(row):
    question = row['Context']
    answer = row['Response']

#     text = f"[INST] {question} [/INST] {answer}".replace('\xa0', ' ')
    text = f"### Question: {question}\n ### Answer: {answer}"

    return text
```

#### Training Hyperparameters


LoRA config-
```bash
# LoRA attention dimension (int)
lora_r = 64

# Alpha parameter for LoRA scaling (int)
lora_alpha = 16

# Dropout probability for LoRA layers (float)
lora_dropout = 0.05

# Bias (string)
bias = "none"

# Task type (string)
task_type = "CAUSAL_LM"

# Random seed (int)
seed = 33
```

Phi-2 config-

```bash
# Batch size per GPU for training (int)
per_device_train_batch_size = 6

# Number of update steps to accumulate the gradients for (int)
gradient_accumulation_steps = 2

# Initial learning rate (AdamW optimizer) (float)
learning_rate = 2e-4

# Optimizer to use (string)
optim = "paged_adamw_8bit"

# Number of training epochs (int)
num_train_epochs = 4

# Linear warmup steps from 0 to learning_rate (int)
warmup_steps = 10

# Enable fp16/bf16 training (set bf16 to True with an A100) (bool)
fp16 = True

# Log every X updates steps (int)
logging_steps = 100

#L2 regularization(prevents overfitting)
weight_decay=0.0

#Checkpoint saves
save_strategy="epoch"
```

BnB config

```bash
# Activate 4-bit precision base model loading (bool)
load_in_4bit = True

# Activate nested quantization for 4-bit base models (double quantization) (bool)
bnb_4bit_use_double_quant = True

# Quantization type (fp4 or nf4) (string)
bnb_4bit_quant_type = "nf4"

# Compute data type for 4-bit base models
bnb_4bit_compute_dtype = torch.bfloat16

```

### Results

Training loss: 2.229
Validation loss: 2.223


## More Information [optional]

[Phi-2](https://huggingface.co/microsoft/phi-2)

## Model Card Authors [optional]

[YuvrajSingh9886](https://huggingface.co/YuvrajSingh9886)
