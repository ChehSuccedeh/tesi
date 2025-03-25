#%% 
# Constants
MODEL_PATH = "huggingface/CodeBERTa-language-id"
js_code = """
import mod189 from './mod189';
var value=mod189+1;
export default value;
"""
python_code = """
def tokenize_function(data):
    print(data)
    to_tokenize = data
    encode = tokenizer(to_tokenize, padding="max_length", truncation=True)
    
    return encode
"""
CODE_TO_TEST = js_code
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


# not working
# inputs = tokenizer(CODE_TO_TEST)
# print(inputs["input_ids"])
# output = model(inputs)[0]
# language_id = output.argmax()
# print(language_id)

print(pipeline(CODE_TO_TEST))
#%% 
# trying keyBert

from keybert import KeyBERT
kb_model = KeyBERT(model=model)

kb_model.extract_keywords(CODE_TO_TEST, highlight=True)
#%%
# trying Bertviz

from bertviz import head_view, model_view, neuron_view

inputs = tokenizer.encode(CODE_TO_TEST, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]
tokens = tokenizer.convert_ids_to_tokens(inputs[0])
print(tokenizer.decode(inputs[0]))


head_view(attention, tokens)
model_view(attention, tokens)

# neuron view is not working (new(): str type not accepted for attention number)
# neuron_view.show(model=model,model_type="bert", tokenizer=tokenizer, sentence_a=CODE_TO_TEST)

# %%
# Trying LIME
import lime, torch
from lime.lime_text import LimeTextExplainer
class_names = [
    "go",
    "java",
    "javascript",
    "php",
    "python",
    "ruby",
]

def prediction(texts):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    tensor_logits = outputs[0]
    # print(tensor_logits)
    probs = torch.nn.functional.softmax(tensor_logits).detach().numpy()
    # print(probs)
    return probs

# print(tokenizer(CODE_TO_TEST, return_tensors='pt', padding=True))

explainer = LimeTextExplainer(class_names=class_names)

explaination = explainer.explain_instance(CODE_TO_TEST, prediction, num_features=15, num_samples=100, top_labels=1)

# print(explaination.available_labels())
explaination.show_in_notebook(text=CODE_TO_TEST)

#%%
