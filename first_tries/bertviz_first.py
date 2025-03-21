#%%
from bertviz import model_view
from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

sentence_a = "prima frase"
sentence_b = "seconda frase"
inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
attention = outputs[-1]
input_ids = inputs['input_ids']
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
model_view(attention, tokens)           
# %%
