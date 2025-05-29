import transformers
MODEL_PATH = "huggingface/CodeBERTa-language-id"
#%%
model = transformers.RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = transformers.RobertaTokenizer.from_pretrained(MODEL_PATH)
#%%
input_text = "func (r *Resource) GetAction(name string) *Action {\n\tfor _, a := range r.Actions {\n\t\tif a.Name == name {\n\t\t\treturn a\n\t\t}\n\t}\n\treturn nil\n}"
output = model(tokenizer.encode(input_text))
print(output)

import shap
shap.plots.bar()
# %%
