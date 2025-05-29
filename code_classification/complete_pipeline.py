#%%
# pip install transformers lime shap 
# pip install tree_sitter tree_sitter_go tree_sitter_java tree_sitter_javascript tree_sitter_php tree_sitter_python tree_sitter_ruby
#%%
MODEL_PATH = "huggingface/CodeBERTa-language-id"
LIME_SAMPLES = 100
#%%
# Loading the model and tokenizer
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)

pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    top_k=None
)
#%%
inputs_to_explain = ["def ciao():\n\tend = idk() + random + \"return_\"\n\treturn", "secondo esempio a caso"]
#%%
# Base model output
model_output = pipeline(inputs_to_explain)
print(model_output)
class_names = [x["label"] for x in model_output[0]]
class_names.sort()
# print(f"{class_names=}")
#%%
# LIME
from lime.lime_text import LimeTextExplainer
import torch
class LIME_predictor():
    def __init__(self, model, tokenizer):
        self.model=model
        self.tokenizer=tokenizer
        
    def prediction(self, texts):
        outputs = self.model(**self.tokenizer(texts, return_tensors="pt", padding=True))
        tensor_logits = outputs[0]
        probabilities = torch.nn.functional.softmax(tensor_logits, dim=1).detach().numpy()
        return probabilities

def LIME_explain_single(explainer, input_string, pipeline, num_features=15, num_samples=LIME_SAMPLES, top_labels=1):
    results = explainer.explain_instance(input_string, pipeline, num_features=num_features, num_samples=num_samples, top_labels=top_labels)
    results.show_in_notebook(text=input_string)
    
    return results.as_list(label=results.top_labels[0])
    

lime_explainer = LimeTextExplainer(class_names=class_names)
lime_words = []
predictor = LIME_predictor(model, tokenizer)
for input_string in inputs_to_explain:
    lime_words.append(LIME_explain_single(lime_explainer, input_string, predictor.prediction))
[print(f"Important words for input {i}: ",x) for i,x in enumerate(lime_words)]
#%%
# SHAP
import shap

explainer = shap.Explainer(pipeline)
shap_values = explainer(inputs_to_explain)
shap.plots.text(shap_values)
if len(shap_values)>1:
    for i,l in enumerate(class_names):
        print(f"{l} feature importance")
        shap.plots.bar(shap_values[:,:,i].mean(0), order=shap.Explanation.argsort.flip)
#%%
# Preprocessing shap values to represent correctly words
import numpy as np

# print(shap_values[0])
new_values = np.empty(shap_values.shape[0], dtype=object)
new_base_values = np.empty(shap_values.shape[0], dtype=object)
new_data = np.empty(shap_values.shape[0], dtype=object)
for i, x in enumerate(inputs_to_explain):
    tokens = tokenizer(x)
    word_ids = tokens.word_ids()
    print(word_ids[-2])
    tmp_data = np.empty(word_ids[-2]+1, dtype=object)
    tmp_values = np.zeros((word_ids[-2]+1, len(class_names)), dtype=np.float64)
    tmp_id = 0
    string = ""
    for index, id in enumerate(word_ids):
        if id is not None:
            if id != tmp_id:
                tmp_data[tmp_id] = string
                string = ""
                tmp_id = id
            string += shap_values[i].data[index]
            tmp_values[id] = np.sum([tmp_values[id], shap_values[i].values[index]], axis=0)
    tmp_data[-1] = string
    print(tmp_values.shape)
    new_values[i] = tmp_values
    new_base_values[i] = shap_values[i].base_values
    new_data[i] = tmp_data

new_shap_values = shap.Explanation(new_values, base_values=new_base_values, data=new_data, output_names=class_names, clustering=shap_values.clustering, feature_names=new_data)
print(new_shap_values.shape, shap_values.shape)
#%%
shap.plots.text(new_shap_values)
print(shap_values.clustering)
print(new_shap_values.hierarchical_values)
if len(shap_values)>1:
    for i,l in enumerate(class_names):
        print(f"{l} feature importance")
        # shap.plots.bar(shap_values[:,:,i].mean(0), order=shap.Explanation.argsort.flip)
        shap.plots.bar(new_shap_values[:,:,i].mean(0), order=shap.Explanation.argsort.flip)