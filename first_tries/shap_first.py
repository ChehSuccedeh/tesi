#%%
# Constants
MODEL_PATH = "huggingface/CodeBERTa-language-id"
js_code = """import mod189 from './mod189';
var value=mod189+1;
export default value;
"""
python_code = """
def ciao():
    end = idk() + random + "return_"
    return
"""
CODE_TO_TEST = [python_code, js_code]


#%% 
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from transformers import TextClassificationPipeline

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

pipeline_ = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)


#%% 
# Base Testing


# not working
# inputs = tokenizer(CODE_TO_TEST)
# print(inputs["input_ids"])
# output = model(inputs)[0]
# language_id = output.argmax()
# print(language_id)
for s in CODE_TO_TEST:
    print(pipeline_(s))

#%% 
import shap
import numpy as np

def score_and_visualize(text):
    # prediction = pipeline_(text)
    # print(prediction[0])
    masker = shap.maskers.Text(tokenizer=tokenizer)
    explainer = shap.explainers.Permutation(pipeline_, masker=masker)
    shap_values = explainer.shap_values(text, npermutations=100)
    
    # print("SHAP Values Shape : {}".format(shap_values.shape))
    # print("SHAP Base Values  : {}".format(shap_values.base_values))
    # print(shap_values[0,3])


    
    shap.plots.text(shap_values)
    # shap.plots.bar(shap_values[:,:, "python"].mean(0))
    # shap.plots.bar(shap_values[0].abs.sum(0))


score_and_visualize(np.array(CODE_TO_TEST))
# %%
