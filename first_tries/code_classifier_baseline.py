import transformers
import torch
MODEL = "huggingface/CodeBERTa-language-id"
model = transformers.RobertaForSequenceClassification.from_pretrained(MODEL)
tokenizer = transformers.RobertaTokenizer.from_pretrained(MODEL)

model_input = tokenizer("def hello_world():\n    print('Hello, world!')", return_tensors="pt")
tokens = tokenizer.tokenize("def hello_world():\n    print('Hello, world!')")
print(tokenizer("def"))
print(f"{tokens=}")
print(f"{model_input=}")
outputs = model(**model_input)
print(outputs)
softmax = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(softmax)

pipe = transformers.TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
outputs = pipe("def hello_world():\n    print('Hello, world!')")
print(outputs)