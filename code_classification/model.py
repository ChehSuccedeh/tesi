#%% 
# Constants
MODEL_PATH = "huggingface/CodeBERTa-language-id"
#! pip install transformers, keybert, bertviz

#%% 
# Loading the model and tokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from transformers import TextClassificationPipeline

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer
)


#%% 
# Base Testing
short_code = """
ef tokenize_function(data):
    print(data)
    to_tokenize = data
    encode = tokenizer(to_tokenize, padding="max_length", truncation=True)
    
    return encode
"""
code_to_test = """
'use strict';

const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const { execSync } = require('child_process');

const path = './users.db';

if (!fs.existsSync(path)) {
    console.error('users.db file not found. Creating DB using db_creation.js');
    try {
        const output = execSync('node ./db_creation.js');
        console.log(`db_creation.js: ${output.toString()}`);
    } catch (err) {
        console.error(`Error executing script: ${err.message}`);
        throw err;
    }
}

const db = new sqlite3.Database(path, (err) => {
    if (err) {
        console.error(err.message);
    } else {
        console.log('Connected to the database.');
    }
});

module.exports = db;
"""

# not working
# inputs = tokenizer(code_to_test)
# print(inputs["input_ids"])
# output = model(inputs)[0]
# language_id = output.argmax()
# print(language_id)

print(pipeline(code_to_test))
#%% 
# trying keyBert

from keybert import KeyBERT
kb_model = KeyBERT(model=model)

kb_model.extract_keywords(code_to_test, highlight=True)
#%%
# trying Bertviz

from bertviz import head_view, model_view, neuron_view

inputs = tokenizer.encode(short_code, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]
tokens = tokenizer.convert_ids_to_tokens(inputs[0])

head_view(attention, tokens)
# model_view(attention, tokens)

# neuron view is not working (new(): str type not accepted for attention number)
# neuron_view.show(model=model,model_type="bert", tokenizer=tokenizer, sentence_a=code_to_test)

# %%
