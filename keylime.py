import keybert
from transformers import AutoModel

# Load the model
# Load your custom model
custom_model = AutoModel.from_pretrained('/path/to/your/custom/model')
model = KeyBERT(custom_model)

