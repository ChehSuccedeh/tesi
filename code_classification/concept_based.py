#%%
# pip install transformers
#%%
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from tqdm import tqdm
import torch
import os
import pickle
#%%
class Roberta_Modified(torch.nn.Module):
    def __init__(self, model):
        super(Roberta_Modified, self).__init__()
        self.roberta_classifier = RobertaForSequenceClassification.from_pretrained(model)
        self.grad_representation = None
        self.representation = None
        
        for name, module in self.roberta_classifier.named_modules():
            # print(f"Registering hooks for {name}")
            if name == "roberta.encoder.layer.5.output":
                # print(f"Registering hooks for {name}")
                module.register_forward_hook(self.forward_hook_fn)
                module.register_backward_hook(self.backward_hook_fn)
                
        self.roberta_classifier.requires_grad_(True)
    
    def forward_hook_fn(self, module, input, output):
        self.representation = output
    
    def backward_hook_fn(self, module, grad_input, grad_output):
        # print(grad_output[0])
        self.grad_representation = grad_output[0]

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            loss, logits = self.roberta_classifier(input_ids, attention_mask=attention_mask, labels=labels)
        else:
            out = self.roberta_classifier(input_ids, attention_mask=attention_mask)
            logits = out[0]
            
        preds = torch.argmax(logits, dim=-1)
        if labels is None:
            return logits, preds, self.representation
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return loss, logits, preds, self.representation

    def forward_from_representation(self, representation):
        logits = self.roberta_classifier.classifier(representation)
        preds = torch.argmax(logits, dim=-1)
        return logits, preds
#%%
MODEL_NAME = "huggingface/CodeBERTa-language-id"
NUM_RUNS = 10000
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#%%
def get_concept_representations(model, tokenizer, concept_examples):
    
    concept_representations = []
    with torch.no_grad():
        for example in tqdm(concept_examples, desc="Getting concept representations"):
            input = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True)
            input_ids = input['input_ids'].to(DEVICE)
            attention_mask = input['attention_mask'].to(DEVICE)
            _,_, representations = model(input_ids, attention_mask=attention_mask)
            # print(representations)
            concept_representations.append(representations[:,0,:].detach().numpy())
    
    # print(concept_representations)
    # concept_representations = torch.cat(concept_representations, dim=0).cpu().detach().numpy()
    
    return concept_representations
    
def get_concept_cavs(model, tokenizer, concept_examples, num_runs = 10):
    """
    Get the concept CAVs for a given set of concept examples.
    """
    # Get the concept CAVs
    concept_cavs = []
    concept_representations = get_concept_representations(model, tokenizer, concept_examples)
    
    for _ in tqdm(range(num_runs), desc="Generating CAVs"):
        # Get the CAV for the current run
        concept_ids = list(np.random.choice(range(len(concept_examples)),5))
        concept_rep = [concept_representations[i] for i in concept_ids]
        concept_cavs.append(np.mean(concept_rep, axis=0))  
          
    return concept_cavs

def get_grad_logit(model, tokenizer, sample, output_class):
    input = tokenizer(sample["text"], return_tensors="pt", padding=True, truncation=True)
    model.zero_grad()
    input_ids = input['input_ids'].to(DEVICE)
    attention_mask = input['attention_mask'].to(DEVICE)
    
    logits,_,representations = model(input_ids, attention_mask=attention_mask)
    
    logits = logits[0,output_class].backward()
    grad = model.grad_representation
    # print(grad)
    grad = grad[0][0].cpu().numpy()
    
    return grad, logits
#%%
def get_TCAVs(output_class, concept_examples, concept_name, random_samples = None, num_runs = 10):
    model = Roberta_Modified(MODEL_NAME)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    concept_cavs = get_concept_cavs(model, tokenizer, concept_examples, num_runs=num_runs)
    
    if os.path.exists(f"grad_logits/{concept_name}.pkl"):
        with open(f"grad_logits/{concept_name}.pkl", "rb") as f:
            data = pickle.load(f)
        grads = data['grads']
        logits = data['logits']
    else:
        if random_samples:
            grads = []
            logits = []
            for sample in tqdm(random_samples, desc="Processing random samples"):
                grad, logit = get_grad_logit(model, tokenizer, sample, output_class)
                grads.append(grad)
                logits.append(logit)
            
            with open(f"grad_logits/{concept_name}.pkl", "wb") as f:
                pickle.dump({'grads': grads, 'logits': logits}, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            raise ValueError("No random samples provided and grad_logits file does not exist.")
    
    sensitivities = []
    for grad in grads:
        sensitivities.append([np.dot(grad, cav.T) for cav in concept_cavs])
    sensitivities = np.array(sensitivities)
    
    tcavs = []
    for i in range(len(concept_cavs)):
        tcavs.append(len([s for s in sensitivities[:,i] if s > 0]) / len(random_samples))
        
    print(f"TCAV for {concept_name}: \n{np.mean(tcavs)}, {np.std(tcavs)}")
    
    return logits, sensitivities, tcavs
#%%
import json
concept_to_analyze = 'comments'
with open(f"./data/{concept_to_analyze}_dataset.jsonl", "r") as f:
    concept_examples = [json.loads(line) for line in f]
classes = {'go': 0, 'java': 1, 'javascript': 2, 'php': 3, 'python': 4, 'ruby': 5}
class_to_examine = 'go'

data = []
for x in concept_examples:
    data.append({'text': x['text'], 'label': classes[x['language']]})

with open("./data/test_dataset.jsonl", "r") as f:
    random_samples = [json.loads(line) for line in f]
random_data = []
for x in random_samples:
    random_data.append({'text': x['code'], 'label': classes[x['language']]})
    
#%%
# single Try
print(f"Calculating TCAVs for {concept_to_analyze} concept")
logits, sensitivities, tcavs = get_TCAVs(classes[class_to_examine], concept_examples, concept_to_analyze, random_samples=random_data, num_runs=NUM_RUNS)

# %%
def automate_TCAVs(random_data, concept_to_analyze, classes_to_examine):
    for concept in concepts: 
        with open(f"./data/{concept}_dataset.jsonl", "r") as f:
            concept_examples = [json.loads(line) for line in f]
            data = []
            for x in concept_examples:
                data.append({'text': x['text'], 'label': classes[x['language']]})

        for c in classes_to_examine:
           # calculate TCAVs
            print(f"Calculating TCAVs for {concept} concept, class {c}")
            logits, sensitivities, tcavs = get_TCAVs(classes[c], data, concept, random_samples=random_data, num_runs=NUM_RUNS)
        
        
concepts = ["comments", "function_declarations", "python_function_declarations", "go_function_declarations", "java_function_declarations", "javascript_function_declarations", "ruby_function_declarations", "php_function_declarations"]
classes_to_examine = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
automate_TCAVs(random_data, concepts, classes_to_examine)