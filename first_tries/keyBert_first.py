from keybert import KeyBERT
from transformers import AutoModel

# Load the model
# Load your custom model
# custom_model = AutoModel.from_pretrained('/path/to/your/custom/model')
base_model = AutoModel.from_pretrained("bert-base-uncased")
model = KeyBERT(base_model)

# Extract keywords
doc = """I live in a house near the mountains. I have two brothers and one sister, and I was born last. My father teaches mathematics, and my mother is a nurse at a big hospital. My brothers are very smart and work hard in school. My sister is a nervous girl, but she is very kind. My grandmother also lives with us. She came from Italy when I was two years old. She has grown old, but she is still very strong. She cooks the best food!
My family is very important to me. We do lots of things together. My brothers and I like to go on long walks in the mountains. My sister likes to cook with my grandmother. On the weekends we all play board games together. We laugh and always have a good time. I love my family very much.
"""

keywords = model.extract_keywords(doc)
print(keywords)

