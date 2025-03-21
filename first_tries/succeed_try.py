#%%
RANDOM_STATE = 42 
REDUCE_DATASET = True
#%%

from datasets import load_dataset, Value

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

new_features = df.features.copy()
new_features["Label"] = Value(dtype="int32")
# df = df.cast(new_features)
# print(df.features)
# print(df_train, df_test)
# print(label2id, id2label, labels)

#%%
df = df.train_test_split(test_size=0.2, seed=RANDOM_STATE, )
#%%
from transformers import AutoTokenizer

def str_to_int(label):
    if label == "Normal":
        return 0
    else:
        return 1

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(data):
    # print(data)
    to_tokenize = data["Payload"]
    encode = tokenizer(to_tokenize, padding="max_length", truncation=True)
    encode["Label"] = [str_to_int(label) for label in data["Label"]]
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
import torch
import numpy as np
from sklearn.metrics import accuracy_score

def metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    predictions = sigmoid(torch.Tensor(predictions))
    # translate prediction to 0 or 1
    y_pred = np.zeros(predictions.shape)
    y_pred[np.where(predictions >= threshold)] = 1
    y_true = labels
    # calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        "accuracy": accuracy,
    }
    return metrics

def compute_metrics(eval_pred):
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    result = metrics(predictions=preds, labels=eval_pred.label_ids)
    return result

#%%
# print(encoded_dataset["train"][0]["Label"].type())
print(encoded_dataset["train"]["input_ids"][0])

output = model(input_ids=encoded_dataset["train"]["input_ids"][0].unsqueeze(0), labels=encoded_dataset["train"][0]["Label"])
print(output)

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
