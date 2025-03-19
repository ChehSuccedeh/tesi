#%%
RANDOM_STATE = 42 
REDUCE_DATASET = True
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

df = load_dataset("csv", data_files='csic_database.csv', split='train')

    
print(df)
features = df.column_names[1:]
print(features)
# Creazione delle colonne 'label' e 'payload'

def add_payload(entry):
    entry["Payload"] = ""
    for feature in features:
        if feature != 'Label':
            if(entry[feature] == None):
                entry[feature] = ""
            entry["Payload"] += feature + " " + str(entry[feature]) + " "
                 
    print(entry)
    return entry

df = df.map(add_payload, with_indices=False, remove_columns=features)
    
print(df)
labels = ["Normal", "Anomalous"]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
# print(df_train, df_test)
# print(label2id, id2label, labels)

#%%
df = df.train_test_split(test_size=0.2, seed=RANDOM_STATE, )
#%%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(data):
    # print(data)
    to_tokenize = data["Payload"]
    encode = tokenizer(to_tokenize, padding="max_length", truncation=True)
    
    return encode
    

#%%
encoded_dataset = df.map(tokenize_function, batched=True)

print(encoded_dataset["train"][0])
print(encoded_dataset["train"][0].keys())
tokenizer.decode(encoded_dataset["train"][0]['input_ids'])
# %%
encoded_dataset.set_format("torch")
# %%
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="single_label_classification", num_labels=len(labels), id2label=id2label, label2id=label2id)
# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    logging_dir='./logs',
    logging_steps=10,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    output_dir='./output',
    save_strategy='epoch',
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

#%%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
)

trainer.train()
# %%


# %%
