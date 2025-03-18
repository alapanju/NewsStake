from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)

import os
from datasets import load_dataset
from datasets import Dataset
import evaluate
import torch
import numpy as np
import json
from collections import Counter
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
device = "cuda"

model_name_or_path = "roberta-large"
task = "mrpc"
num_epochs = 20
lr = 1e-3
batch_size = 8

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    return clf_metrics.compute(predictions= predictions, references=labels)
    #return accuracy.compute(predictions=predictions, references=labels)

#dataset = load_dataset("glue", task)



#identify the unique stakeholder types and enlist them
json_file_path = "/home/alapan/test_files/demonetization/sc_dataset_edited.json"
with open(json_file_path,'rb') as f:
    data_sc = json.load(f)

stakeholder_types_list = []
stakeholder_phrase_list = []
stakeholder_context_list = []

for record_item in data_sc:
    stakeholder_types_list.append(record_item['type'])
    stakeholder_phrase_list.append(record_item['phrase'])
    stakeholder_context_list.append(record_item['context'])

unique_s_type_list = list(set(stakeholder_types_list))#unique stakeholder types

print(f'Number of training data before adding negative samples = {len(stakeholder_types_list)}')
print(f'Number of unique stakeholder types = {len(unique_s_type_list)}')

# Count the frequency of each item in the list
frequency_counter = Counter(stakeholder_types_list)

# Print unique items with their frequencies
for item, frequency in frequency_counter.items():
    print(f"Item: {item}, Frequency: {frequency}")


def prepare_model_data(texts, labels, s_phrase, hypothesis_placeholder):
    sequences = []
    hypotheses= []
    outputs   = []
    for i in range(len(texts)):
        text = texts[i]
        phrase = s_phrase[i]
        label = labels[i]#actual stakeholder type
        for j in range(len(unique_s_type_list)):
            if unique_s_type_list[j]==label:
                sequences.append(text)#sentential context
                hypotheses.append(hypothesis_placeholder.format(phrase, label))#hypothesis
                outputs.append(1)#label
            else:
                #topic = find_topic(i, train)
                if random.randint(0,8)==0 and label in unique_s_type_list:
                    sequences.append(text)
                    hypotheses.append(hypothesis_placeholder.format(phrase, unique_s_type_list[j]))
                    outputs.append(0)

    return sequences, hypotheses, outputs


train_sequences, train_hypotheses, train_true_outputs = prepare_model_data(stakeholder_context_list, stakeholder_types_list, stakeholder_phrase_list, "The entity {} belongs to the stakeholder group of {}")


suffling_list = np.arange(len(train_sequences))
np.random.shuffle(suffling_list)
suffling_list = list(suffling_list)



#randomization
train_sequences = [train_sequences[ri] for ri in suffling_list]
train_hypotheses = [train_hypotheses[ri] for ri in suffling_list]
train_true_outputs = [train_true_outputs[ri] for ri in suffling_list]

print(f'After suffling and negative sampling number of samples: sent= {len(train_sequences)}, hypothesis={len(train_hypotheses)}, true_labels = {len(train_true_outputs)}')


train_premise, val_premise, train_hypo, val_hypo, train_label, val_label = train_test_split(train_sequences, train_hypotheses, train_true_outputs, test_size=.3)

print(f"The train and validation splitting done:")
print(len(train_premise))
print(len(val_premise))
print(len(train_hypo))
print(len(val_hypo))
print(len(train_label))
print(len(val_label))
#print(train_premise[0])
#print(train_hypo[0])
#print(train_label[0])

'''
datadict = {'sentence1' : [], 'sentence2': [], 'label' : []}

datadict['sentence1'].append(train_premise)
datadict['sentence2'].append(train_hypo)
datadict['label'].append(train_label)
ds_train = Dataset.from_dict({'sentence1' : datadict["sentence1"], 'sentence2' : datadict["sentence2"], 'label' : datadict["label"]})

datadict1 = {'sentence1' : [], 'sentence2': [], 'label' : []}

datadict1['sentence1'].append(val_premise)
datadict1['sentence2'].append(val_hypo)
datadict1['label'].append(val_label)

ds_val = Dataset.from_dict({'sentence1' : datadict1["sentence1"], 'sentence2' : datadict1["sentence2"], 'label' : datadict1["label"]})

dataset["train"] = ds_train
dataset["validation"] = ds_val
print(dataset)
#print(f'ds_train[7]=\n{ds_train[0]}')
#print(f'ds_val[7]=\n{ds_val[7]}')
'''

columns = ["sentence","hypothesis","label"]
train_df = pd.DataFrame(columns=columns)
train_df["sentence"] = train_premise
train_df["hypothesis"] = train_hypo
train_df["label"] = train_label
train_df.to_csv('all_aspect_train.csv')
train_dataset = Dataset.from_pandas(train_df)

#print("\ntraining dataset size={}".format(len(train_sequences)))

val_df = pd.DataFrame(columns=columns)
val_df["sentence"] = val_premise
val_df["hypothesis"] = val_hypo
val_df["label"] = val_label
val_df.to_csv('all_aspect_dev.csv')
val_dataset = Dataset.from_pandas(val_df)


if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

#tokenizer.model_max_length = 512

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence"], examples["hypothesis"], truncation='longest_first', max_length=491)
    return outputs


tokenized_datasets_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence", "hypothesis"],
)

tokenized_datasets_train = tokenized_datasets_train.rename_column("label", "labels")

tokenized_datasets_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence", "hypothesis"],
)

tokenized_datasets_val = tokenized_datasets_val.rename_column("label", "labels")


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets_train, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets_val, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)

tokenized_datasets_train.set_format("torch")
tokenized_datasets_val.set_format("torch")


print(tokenized_datasets_train.features)
print(tokenized_datasets_val.features)

peft_config = PromptEncoderConfig(peft_type="P_TUNING", task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)

model.print_trainable_parameters()
'''
training_args = TrainingArguments(
    output_dir="/home/alapan/llm/roberta_p_tuning",
    learning_rate=1e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit = 2,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("/home/alapan/llm/roberta_p_tuning")
'''

optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

import torch
torch.cuda.empty_cache()
num_epochs = 10
metric = evaluate.load("glue", task)
accuracies = []
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        #print(batch['labels'].shape)
        #print(batch['input_ids'].shape)
        #print(batch['attention_mask'].shape)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    accuracies.append(float(eval_metric['accuracy']))
    print(f"\nepoch {epoch}:", eval_metric,"\n")
accuracies.sort(reverse = True)
print("\nbest accuracy:", accuracies[0])
