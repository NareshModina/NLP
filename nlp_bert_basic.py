import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset #First install datasets library pip/pip3 install datasets
import numpy as np
import evaluate

checkpoint = "Bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data = load_dataset("glue", "mrpc")
def tokenization(sequence):
    return tokenizer(sequence["sentence1"], sequence["sentence2"], truncation=True)
tokenized_data = data.map(tokenization, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# evaluation setting
def evaluation(eval_preds):
    metrics = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds #logits and label_ids
    predictions = np.argmax(logits, axis = -1)
    return metrics.compute(predictions=predictions, references=labels)

trainer = Trainer(model, 
                  training_args, 
                  train_dataset=tokenized_data["train"],
                  eval_dataset=tokenized_data["validation"],
                  tokenizer=tokenizer,data_collator=data_collator, 
                  compute_metrics = evaluation)

# Launch training with evaluation
trainer.train()
