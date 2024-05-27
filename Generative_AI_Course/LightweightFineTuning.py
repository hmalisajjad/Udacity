from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from transformers import DataCollatorWithPadding
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=28)
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_dataset('go_emotions')

def convert_multi_label_to_single(examples):
    examples['labels'] = [labels[0] if labels else 0 for labels in examples['labels']]
    return examples

dataset = dataset.map(convert_multi_label_to_single, batched=True)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

encoded_dataset = encoded_dataset.map(lambda examples: {'labels': examples['labels']}, batched=True)

encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataset = encoded_dataset['train']
test_dataset = encoded_dataset['test']

data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4,  
    fp16=True, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()

results = trainer.evaluate()

print(results)