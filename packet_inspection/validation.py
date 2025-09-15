import json
import os
import torch
CLASSES = {
    "Adware": 0,
    "Backdoor": 1,
    "Botnet": 2,
    "CGI": 3,
    "Code-execution": 4,
    "DDos": 5,
    "Dir-Traversal": 6,
    "Dos": 7,
    "Info-Disclosure": 8,
    "Injection": 9,
    "Other": 10,
    "Overflow": 11,
    "Ransomware": 12,
    "Remote-file-Inclusion": 13,
    "Scanner": 14,
    "Spyware": 15,
    "Trojan": 16,
    "Virus": 17,
    "Webshell": 18,
    "Worm": 19,
    "XSS": 20
}
INV_CLASSES = {v: k for k, v in CLASSES.items()}
CONCEPTS= ["ip"]
CLASSES_TO_EXAMINE = ["Adware", "Scanner", "Spyware", "Trojan", "XSS", "Remote-file-Inclusion", "Overflow", "Injection", "Info-Disclosure", "Dir-Traversal", "Code-execution", "CGI", "Ransomware", "Botnet", "Backdoor"]
MODEL_NAME = "./codebert-base-mlm"


from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAIN_DATASETS = {}
def load_dataset(name):
    with open(f"./data/packet_inspection/{name}.jsonl", "r") as f:
        return [json.loads(line) for line in f]
    

MAIN_DATASETS["complete"] = load_dataset("packets_dataset")
MAIN_DATASETS["ip"] = load_dataset("ip_dataset")


CONCEPT_TO_DATASET = {
    "ip": MAIN_DATASETS["ip"],
}

#-----------
#-----------
#-----------
#-------------------------------------------------------------------------------------
#-----------
#-----------
#-----------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import SGDClassifier
import random
import string

class TCAV_Avg:
    """ Class for concept activation vectors for PyTorch models.

    Attributes:
        model: a roBerta LLM loaded through Huggingface API
        tokenizer: a tokenizer for roBerta
        cavs: dict mapping concept names to their activation vectors
        sensitivities: dict mapping concept names to their sensitivities
        y_labels:  list that will contain labels of the values used to calculate sensitivities
        bottleneck: list of bottleneck layers to analyze
        model_activations: dict storing activations and gradients for hooks
        fix_length: optional fixed sequence length for tokenization
    """

    def __init__(self, model=None, tokenizer=None, fix_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.cavs = {}  # dict: concept_name -> cav
        self.sensitivities = {}  # dict: concept_name -> sensitivities
        self.y_labels = None  # list of testing labels -> y_labels
        self.current_concept = None
        self.bottleneck = [] # list of bottleneck layers to analyze
        self.model_activations = {}
        self.fix_length = fix_length
    
    def forward_hook_fn(self, name:str):
        if name not in self.model_activations.keys():
            self.model_activations["forward_"+name] = []
        def fn(module, input, output):
            # print("leaf",output.is_leaf)
            # x = output
            # print("leaf",output[0].is_leaf)
            x = output[0]
            x.requires_grad_(True)
            x.retain_grad()
            self.model_activations["forward_"+name].append(x)
            # print("extracted activations", output.shape)
            # print("extracted activations", output[0].shape)
            # print("----------------",output[0].shape[1],"---------------")
            # self.fix_length = output[0].shape[1]
            return
        return fn
    
    def backward_hook_fn(self, name:str):
        if name not in self.model_activations.keys():
            self.model_activations["backward_"+name] = []
            
        def fn(module, grad_input, grad_output):
            # print("leaf",grad_output[0].is_leaf)
            
            self.model_activations["backward_"+name] = grad_output[0]
            # print("extracted grads", grad_output[0].shape)
            return
        
        return fn

    def set_concept(self, concept):
        """ Set the concept name """
        self.current_concept = concept
        
    def set_model(self, model):
        """ Set the model """
        self.model = model
        return

    def set_tokenizer(self, tokenizer):
        """ Set the tokenizer """
        self.tokenizer = tokenizer
        return

    def split_model(self, bottleneck):
        """ Set the hook at the bottleneck layer """
        if bottleneck < 0 or bottleneck >= len(self.model.roberta.encoder.layer):
            raise ValueError("Invalid layer for sampling")
        
        # layers = list(self.model.children())
        # print(layers)
        self.bottleneck.append(str(bottleneck))   
        
        # self.model.classifier.dense.register_forward_hook(self.hook_fn(str(bottleneck)))
        self.model.roberta.encoder.layer[bottleneck].register_forward_hook(self.forward_hook_fn(str(bottleneck)))
        self.model.roberta.encoder.layer[bottleneck].register_full_backward_hook(self.backward_hook_fn(str(bottleneck)))
        return


    def _create_counterexamples(self, x_concept):
        """ Creates random counterexamples to a series of concept inputs """
        n = len(x_concept)
    
        counterexamples = []
        for i in range(n):
            l = len(x_concept[i])
            counterexamples.append(''.join(random.choices(string.printable, k=l)))
        return counterexamples

    def _tokenize(self, inputs, return_words=False):
        """ Tokenize the inputs (if tokenizer is provided) """
        # print(len(inputs))
        if self.tokenizer is not None:
            x = self.tokenizer(inputs, return_tensors="pt", padding="max_length", max_length=self.fix_length, truncation=True)
            # print("tokenizer", x["input_ids"].shape)
            if return_words:
                strings = []
                for i in range(len(inputs)):
                    # Ottieni gli ID dei token (inclusi speciali) per ogni input
                    input_ids = x["input_ids"][i]
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                    strings.append(tokens)
                return x, strings
            return x
        return inputs

    def train_cav(self, x_concept):
        """ Train and extract the Concept Activation Vector """
        
        counterexamples = self._create_counterexamples(x_concept)
        tmp = x_concept + counterexamples
        x_train_concept = self._tokenize(tmp)
        y_train_concept = torch.cat((torch.ones(len(x_concept)), torch.zeros(len(counterexamples))))
        
        # print("calculating cavs")
        # Obtain activations of concept and counterexamples
        with torch.no_grad():
            _ = self.model(**x_train_concept)
            
            concept_activations = {}
            for bottleneck in self.bottleneck:
                if(len(self.model_activations["forward_"+bottleneck][0].shape)>2):
                    # Instead of flattening (n, m, z) to (n, m*z), take the mean over m to get (n, z)
                    # print("attentions dimensions:", len(self.model_activations["forward_"+bottleneck][0].shape))
                    activations = self.model_activations["forward_"+bottleneck][0]  # (batch, seq_len, hidden)
                    attention_mask = x_train_concept["attention_mask"]  # (batch, seq_len)
                    # print("attention mask", attention_mask.shape, attention_mask)
                    mask = attention_mask.unsqueeze(-1).expand(activations.size())  # (batch, seq_len, hidden)
                    activations_masked = activations * mask  # maschera i padding
                    lengths = attention_mask.sum(dim=1).unsqueeze(-1)  # (batch, 1)
                    # Evita divisione per zero
                    lengths = lengths.clamp(min=1)
                    concept_activations[bottleneck] = activations_masked.sum(dim=1) / lengths
                else:
                    concept_activations[bottleneck] = self.model_activations["forward_"+bottleneck][0]
                # print("concept activations shape", concept_activations.shape)
        # print(concept_activations.shape)
        
        # Iterate over all bottlenecks
        # print("bottlenecks", self.bottleneck)
        self.cavs[self.current_concept] = {}

        for b in concept_activations.keys():
            # Train linear classifier
            lm = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
            lm.fit(concept_activations[b].detach().numpy(), y_train_concept.numpy())
            cav = -lm.coef_.T
            
            self.cavs[self.current_concept][b] = cav
            # print("cav", len(cav))
        
            self.model_activations["forward_"+b] = [] #once calculated all results, reset for next operations                           
            self.model_activations["backward_"+b] = []
            
        del tmp
        del x_train_concept
        del y_train_concept
        del concept_activations
        return
        
    def calculate_sensitivity(self, x_train, y_train, device="cpu"):
        """
        This function calculates the TCAV sensitivity scores for a given set of inputs and labels.
        It computes the gradient of the loss with respect to the activations at the bottleneck layer,
        projects these gradients onto the concept activation vector (CAV), and measures how sensitive
        the model's predictions are to the concept for each class. The results are stored for later analysis.
        """
        
        # print("calculating sensitivity")
        x_train, x_train_words = self._tokenize(x_train, return_words=True)
        # print(x_train_words[0]) # x_train_words list of n_sentences x n_tokens
        # print(x_train)
        
        # Calculate the output and obtain activations
        x_train = x_train.to(device)
        output = self.model(**x_train)

        # Control the format
        if isinstance(y_train, list):
            y_train = np.array(y_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.from_numpy(y_train)
        y_labels = y_train.view(-1).to(device)

        # Define and compute the loss
        loss = F.cross_entropy(output.logits, y_labels)
        
        loss.backward()
        
        grads = {}
        avg_grads = {}
        for bottleneck in self.bottleneck:
            grads[bottleneck] = self.model_activations["backward_"+bottleneck]
            
            attention_mask = x_train["attention_mask"]  # (batch, seq_len)
            mask = attention_mask.unsqueeze(-1).expand(grads[bottleneck].size())  # (batch, seq_len, hidden)
            grads_masked = grads[bottleneck] * mask  # maschera i padding
            
            lengths = attention_mask.sum(dim=1).unsqueeze(-1)  # (batch, 1)
            lengths = lengths.clamp(min=1)
            
            grads[bottleneck] = grads_masked  # NON facciamo la media, lasciamo (batch, seq_len, hidden)
            
            avg_grads[bottleneck] = grads_masked.sum(dim=1) / lengths  # (batch, hidden)

        cavs = self.cavs

        # print("grads", grads)
        
        sensitivities = {}
        sensitivities_per_sample = {}
        token_sensitivities = {}
        for concept in cavs.keys():
            sensitivities[concept] = {}
            sensitivities_per_sample[concept] = {}
            token_sensitivities[concept] = {}
            cav_tensor = cavs[concept]
            for b in self.bottleneck:
                # Sensibility for concept
                sensitivities[concept][b] = np.dot(avg_grads[b], cav_tensor[b])
                
                # Sensibilities for each token: (batch, seq_len, hidden) x (hidden, 1) -> (batch, seq_len)
                grad_b = grads[b].detach().cpu().numpy()  # (batch, seq_len, hidden)
                # print("grad_b", grad_b.shape)
                # print("x_train_words", len(x_train_words), len(x_train_words[0]))
                cav_b = cav_tensor[b]
                # Prodotto scalare per ogni token
                sens_per_token = []
                for i in range(grad_b.shape[0]):
                    sens_per_token.append([])
                    for j in range(grad_b.shape[1]):
                        g = grad_b[i, j, :]
                        sens_per_token[i].append((x_train_words[i][j], np.dot(g, cav_b).to_list()[0]))  # (token, sensitivity)
                token_sensitivities[concept][b] = sens_per_token


        # Salva anche le altre info come prima
        self.sensitivities = sensitivities
        self.sensitivities_per_sample = token_sensitivities
        self.y_labels = y_train.detach().cpu().numpy().reshape(-1)
        
        del grads
        del cavs
        del x_train
        del output

        return
        
    def print_all_sensitivities(self, id_to_labels):
        for concept, sensitivities_dict in self.sensitivities.items():
            print(f"Sensitivities for concept '{concept}':")
            for bottleneck, sensitivity in sensitivities_dict.items():
                print(f"  Bottleneck {bottleneck}:")
                num_labels = len(np.unique(self.y_labels))
                for label_idx in range(num_labels):
                    idxs = np.where(self.y_labels == label_idx)[0]
                    value = np.sum(sensitivity[idxs] > 0) / idxs.shape[0]
                    print(f"    Class {id_to_labels[label_idx]}: {value:.2f}")
                print("-" * 5)

        for concept, token_sens_dict in self.sensitivities_per_sample.items():
            print(f"Token Sensitivities for concept '{concept}':")
            for bottleneck, sens_per_sample in token_sens_dict.items():
                print(f"  Bottleneck {bottleneck}:")
                for i, token_sens in enumerate(sens_per_sample):
                    print(f"    Sample {i}: {token_sens}")

    def get_sensitivity_results(self, id_to_labels, include_per_sample=False):
        """
        Returns a list of dicts with the sensitivity results for all concepts, bottlenecks and classes.

        Args:
            id_to_labels (dict): Mapping from class indices to class names.

        Returns:
        [
            {"concept": "...", "bottleneck": "...", "class": "...", "sensitivity": 0.xx},
            ...
        ]
        """
        results = []
        if not include_per_sample:
            for concept, sensitivities_dict in self.sensitivities.items():
                for bottleneck, sensitivity in sensitivities_dict.items():
                    num_labels = len(np.unique(self.y_labels))
                    for label_idx in range(num_labels):
                        idxs = np.where(self.y_labels == label_idx)[0]
                        value = np.sum(sensitivity[idxs] > 0) / idxs.shape[0]
                        results.append({
                            "concept": concept,
                            "bottleneck": bottleneck,
                            "class": id_to_labels[label_idx],
                            "sensitivity": value
                        })
                        
        else:
            for concept, token_sens_dict in self.sensitivities_per_sample.items():
                for bottleneck, sens_per_sample in token_sens_dict.items():
                    for i, token_sens in enumerate(sens_per_sample):
                        results.append({
                            "concept": concept,
                            "bottleneck": bottleneck,
                            "sample_index": i,
                            "token_sensitivities": token_sens  # List of (token, sensitivity) tuples
                        })
        return results
    
        
#--------------------------------------------------------------------------------------------  

from datetime import datetime
import gc
import time
import sys
# Ensure the results directory exists
if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./results/packet_inspection"):
    os.makedirs("./results/packet_inspection")

f_avg = open("./results/packet_inspection/validation.txt", "a")
f_avg.write(datetime.now().isoformat() + "\n")

f_avg.close()

import json
import random

print(f"------------------- Executing Run -------------------")
# Initialize TCAV instances for different configurations
tcav_avg = TCAV_Avg(model=model, tokenizer=tokenizer)

# Random extract samples from main datasets
samples = {}

samples["random"] = random.sample(MAIN_DATASETS["complete"], len(MAIN_DATASETS["complete"]))
samples["test"] = random.sample(MAIN_DATASETS["complete"], len(MAIN_DATASETS["complete"]))

# Extract samples for each concept
for concept in CONCEPTS:
    samples[concept] = random.sample(CONCEPT_TO_DATASET[concept], len(CONCEPT_TO_DATASET[concept]))

for dataset_name, dataset in MAIN_DATASETS.items():
    samples[dataset_name] = random.sample(dataset, len(dataset))

# Set the bottleneck layer
for layer in range (6):
    tcav_avg.split_model(layer)

# -------------------------------- Training TCAVs -------------------------------- 
# Train CAVs for random baseline
tcav_avg.set_concept("random")

# Train CAVs for auto and fixed length
# tcav_auto.train_cav([s["text"] for s in samples["random"]])
# tcav_fixed.train_cav([s["text"] for s in samples["random"]])

# Train CAV for average pooling
tcav_avg.train_cav([s["text"] for s in samples["random"]])

# Train CAVs for each concept
for concept in CONCEPTS:
    tcav_avg.set_concept(concept)

    # Train CAVs for auto and fixed length
    print(f"Training CAV for concept: {concept}")

    # Train CAV for average pooling
    # print("Avg TCAV training...")
    tcav_avg.train_cav([s["text"] for s in samples[concept]])

# print("finished training")
# -------------------------------- Calculating Sensitivities --------------------------------

print("--------- Calculating sensitivities --------")

tcav_avg.calculate_sensitivity([s["text"] for s in samples["test"]], [CLASSES[s["class"]] for s in samples["test"]])

f_avg = open("./results/packet_inspection/validation.txt", "a")
f_avg.write(json.dumps(tcav_avg.get_sensitivity_results(INV_CLASSES, include_per_sample=True), indent=4)+ "\n")
f_avg.close()

print("Avg TCAV Sensitivities:")
tcav_avg.print_all_sensitivities(INV_CLASSES)
