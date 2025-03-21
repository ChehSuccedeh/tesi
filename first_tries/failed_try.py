#%%
RANDOM_STATE = 42 
REDUCE_DATASET = True
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

df = load_dataset('csic_database.csv')

df = pd.read_csv('csic_database.csv')
if (REDUCE_DATASET):
    df = df.sample(frac=0.1, random_state=RANDOM_STATE)
    

# Creazione delle colonne 'label' e 'payload'
df['Payload'] = df.apply(lambda row: ' '.join([f"{col} {row[col]}" for col in df.columns if col != 'Label']), axis=1)
df = df[['Label', 'Payload']]
print(df.head())
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
labels = ["Normal", "Anomalous"]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
print(df_train, df_test)
# print(label2id, id2label, labels)

#%%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(data):
    print(data)
    to_tokenize = data
    encode = tokenizer(to_tokenize, padding="max_length", truncation=True)
    
    return encode
    

#%%
encoded_dataset = df_train['Payload'].apply(tokenize_function).apply(pd.Series)
encoded_dataset["label"] = df_train['Label'].apply(lambda x: label2id[x])
print(encoded_dataset.iloc[0])
print(encoded_dataset.iloc[0].keys())
tokenizer.decode(encoded_dataset.iloc[0]['input_ids'])
# %%
# encoded_dataset.set_format("torch")
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
import torch
encoded_dataset = pd.DataFrame(encoded_dataset, dtype=torch.float)

    
#%%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
)

trainer.train()
# %%


# %%
