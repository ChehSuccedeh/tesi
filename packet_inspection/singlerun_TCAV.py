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
    "Ransonware": 12,
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
CLASSES_TO_EXAMINE = ["adware", "scanner", "spyware", "trojan", "xss", "remotefileinclosure", "overflow", "injection", "infodisclosure", "directorytraversal", "codeexecution", "cgi", "ransomware", "botnet", "backdoor"]
MODEL_NAME = "./codebert-base-mlm"


from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(CLASSES_TO_EXAMINE))
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

class TCAV:
    """ Class for concept activation vectors for PyTorch models.

    Attributes:
        model: a roBerta LLM loaded through Huggingface API
        tokenizer: a tokenizer for roBerta
        cavs: dict mapping concept names to their activation vectors
        sensitivities: dict mapping concept names to their sensitivities
        y_labels: list that will contain labels of the values used to calculate sensitivities
        bottleneck: list of bottleneck layers to analyze
        model_activations: dict storing activations and gradients for hooks
        fix_length: optional fixed sequence length for tokenization
    """

    def __init__(self, model=None, tokenizer=None, fix_length=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cavs = {}  # dict: concept_name -> cav
        self.sensitivities = {}  # dict: concept_name -> sensitivities
        self.y_labels = None # list of testing labels concept_name -> y_labels
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

    def _tokenize(self, inputs):
        """ Tokenize the inputs (if tokenizer is provided) """
        # print(len(inputs))
        if self.tokenizer is not None:
            if self.fix_length:
                x = self.tokenizer(inputs, return_tensors="pt", padding="max_length", max_length=self.fix_length, truncation=True)
            else:
                x = self.tokenizer(inputs, return_tensors="pt", padding="longest", truncation=True)
                self.fix_length = x["input_ids"].shape[1]
            # print("tokenizer", x["input_ids"].shape)
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
            # print("attentions dimensions:", len(self.model_activations["forward_"+self.bottleneck][0].shape))
            concept_activations = {}
            for bottleneck in self.bottleneck:
                if(len(self.model_activations["forward_"+bottleneck][0].shape)>2):
                    # Flatten the activations if they are not linear: (n, m, z) to (n, m*z)
                    concept_activations[bottleneck] = self.model_activations["forward_"+bottleneck][0].reshape(self.model_activations["forward_"+bottleneck][0].shape[0],-1)
                else:
                    # If activations are linear, no flattening need: (n, m)
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
        return
        
    def calculate_sensitivity(self, x_train, y_train, device="cpu"):
        """
        This function calculates the TCAV sensitivity scores for a given set of inputs and labels.
        It computes the gradient of the loss with respect to the activations at the bottleneck layer,
        projects these gradients onto the concept activation vector (CAV), and measures how sensitive
        the model's predictions are to the concept for each class. The results are stored for later analysis.
        """
        
        # print("calculating sensitivity")
        x_train = self._tokenize(x_train)
        # print(x_train)
        
        # Calculate the output and obtain activations
        x_train = x_train.to(device)
        output = self.model(**x_train)
        
        #activations = self.model_activations["forward_"+self.bottleneck][0].reshape(self.model_activations["forward_"+self.bottleneck][0].shape[0],-1) # Prendi l'ultima attivazione del bottleneck
        # print("output logits", output.logits.shape)
        # print("activation shape", activations.shape)
        
        # Control the format
        if isinstance(y_train, list):
            y_train = np.array(y_train)
            # print(y_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.from_numpy(y_train)
        y_labels = y_train.view(-1).to(device)
        # print(y_labels)

        # Define and compute the loss
        loss = F.cross_entropy(output.logits, y_labels)
        # print("loss", loss)
        # print("activations",activations.is_leaf, activations)
        # print("activations requires grad", activations.requires_grad)

        # Calculate the gradient
        loss.backward()
        
        grads = {}
        for bottleneck in self.bottleneck:
            grads[bottleneck] = self.model_activations["backward_"+bottleneck]
    
        # grads = torch.autograd.grad(loss, activations, allow_unused=True)
        # print("grads", grads)
        # concatenate grads
        grads = {k: v.reshape(v.shape[0], -1) for k, v in grads.items()}

        # Scalar product
        cavs = self.cavs
        
        # print("shapes", cavs.shape, grads.shape)
        sensitivities = {}
        for concept in cavs.keys():
            sensitivities[concept] = {}
            cav_tensor = cavs[concept]
            for b in self.bottleneck:
            # print("cav_tensor", cav_tensor.shape)
                sensitivities[concept][b] = (np.dot(grads[b], cav_tensor[b]))
        
        
        # print("sensitivity", sensitivity)

        # Saving sensitivity
        del grads, cavs
        self.sensitivities = sensitivities
        self.y_labels = y_train.detach().cpu().numpy().reshape(-1)

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

    def get_sensitivity_results(self, id_to_labels):
        """
        Returns a list of dicts with the sensitivity results for all concepts, bottlenecks and classes.
        Format:
        [
            {"concept": "...", "bottleneck": "...", "class": "...", "sensitivity": 0.xx},
            ...
        ]
        """
        results = []
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
        return results
    


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

    def _tokenize(self, inputs):
        """ Tokenize the inputs (if tokenizer is provided) """
        # print(len(inputs))
        if self.tokenizer is not None:
            x = self.tokenizer(inputs, return_tensors="pt", padding="max_length", max_length=self.fix_length, truncation=True)
            # print("tokenizer", x["input_ids"].shape)
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
        x_train = self._tokenize(x_train)
        # print(x_train)
        
        # Calculate the output and obtain activations
        x_train = x_train.to(device)
        output = self.model(**x_train)

        # Take the mean over the sequence dimension (m) to get (n, z)
        # activations = self.model_activations["forward_"+self.bottleneck][0].mean(dim=1)
        # print("output logits", output.logits.shape)
        # print("activation shape", activations.shape)
        
        # Control the format
        if isinstance(y_train, list):
            y_train = np.array(y_train)
            # print(y_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.from_numpy(y_train)
        y_labels = y_train.view(-1).to(device)
        # print(y_labels)

        # Define and compute the loss
        loss = F.cross_entropy(output.logits, y_labels)
        # print("loss", loss)
        # print("activations",activations.is_leaf, activations)
        # print("activations requires grad", activations.requires_grad)

        # Calculate the gradient
        loss.backward()
        
        grads = {}
        for bottleneck in self.bottleneck:
            grads[bottleneck] = self.model_activations["backward_"+bottleneck]
            if grads[bottleneck].dim() > 2:
                attention_mask = x_train["attention_mask"]  # (batch, seq_len)
                mask = attention_mask.unsqueeze(-1).expand(grads[bottleneck].size())  # (batch, seq_len, hidden)
                grads_masked = grads[bottleneck] * mask  # maschera i padding
                lengths = attention_mask.sum(dim=1).unsqueeze(-1)  # (batch, 1)
                lengths = lengths.clamp(min=1)
                grads[bottleneck] = grads_masked.sum(dim=1) / lengths
    
        # grads = torch.autograd.grad(loss, activations, allow_unused=True)
        # print("grads", grads)
        # concatenate grads
        # grads = {k: v.reshape(v.shape[0], -1) for k, v in grads.items()}

        # Scalar product
        cavs = self.cavs
        
        # print("shapes", cavs.shape, grads.shape)
        sensitivities = {}
        for concept in cavs.keys():
            sensitivities[concept] = {}
            cav_tensor = cavs[concept]
            for b in self.bottleneck:
            # print("cav_tensor", cav_tensor.shape)
                sensitivities[concept][b] = (np.dot(grads[b], cav_tensor[b]))
        
        
        # print("sensitivity", sensitivity)

        # Saving sensitivity
        self.sensitivities = sensitivities
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

    def get_sensitivity_results(self, id_to_labels):
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
        return results

#--------------------------------------------------------------------------------------------  
    
from datetime import datetime
import gc
import time
import sys
# Ensure the results directory exists
if not os.path.exists("./results"):
    os.makedirs("./results")

f_auto = open("./results/tcav_auto.txt", "a")
f_auto.write(datetime.now().isoformat() + "\n")
f_fixed = open("./results/tcav_fixed.txt", "a")
f_fixed.write(datetime.now().isoformat() + "\n")
f_avg = open("./results/tcav_avg.txt", "a")
f_avg.write(datetime.now().isoformat() + "\n")
f_auto.close()
f_fixed.close()
f_avg.close()
import json
import random

print(f"------------------- Executing Run -------------------")
# Initialize TCAV instances for different configurations
tcav_auto = TCAV(model=model, tokenizer=tokenizer)
tcav_fixed = TCAV(model=model, tokenizer=tokenizer, fix_length=512)
tcav_avg = TCAV_Avg(model=model, tokenizer=tokenizer)

# Random extract samples from main datasets
samples = {}

# samples["random"] = random.sample(MAIN_DATASETS["complete"], 500)
# samples["test"] = random.sample(MAIN_DATASETS["complete"], 500)
# yet to have enough samples
samples["random"] = random.shuffle(MAIN_DATASETS["complete"])
samples["test"] = random.shuffle(MAIN_DATASETS["complete"])

# Extract samples for each concept
for concept in CONCEPTS:
    # samples[concept] = random.sample(CONCEPT_TO_DATASET[concept], 500)
    samples[concept] = random.shuffle(CONCEPT_TO_DATASET[concept])

for dataset_name, dataset in MAIN_DATASETS.items():
    # samples[dataset_name] = random.sample(dataset, 500)
    samples[dataset_name] = random.shuffle(dataset)

# Set the bottleneck layer
for layer in range (6):
    tcav_auto.split_model(layer)
    tcav_fixed.split_model(layer)
    tcav_avg.split_model(layer)

# -------------------------------- Training TCAVs -------------------------------- 
# Train CAVs for random baseline
tcav_auto.set_concept("random")
tcav_fixed.set_concept("random")
tcav_avg.set_concept("random")

# Train CAVs for auto and fixed length
tcav_auto.train_cav([s["text"] for s in samples["random"]])
tcav_fixed.train_cav([s["text"] for s in samples["random"]])

# Train CAV for average pooling
tcav_avg.train_cav([s["text"] for s in samples["random"]])

# Train CAVs for each concept
for concept in CONCEPTS:
    tcav_auto.set_concept(concept)
    tcav_fixed.set_concept(concept)
    tcav_avg.set_concept(concept)

    # Train CAVs for auto and fixed length
    print(f"Training CAV for concept: {concept}")
    print("Auto TCAV training...")
    tcav_auto.train_cav([s["text"] for s in samples[concept]])
    print("Fixed TCAV training...")
    tcav_fixed.train_cav([s["text"] for s in samples[concept]])

    # Train CAV for average pooling
    print("Avg TCAV training...")
    tcav_avg.train_cav([s["text"] for s in samples[concept]])

# print("finished training")
# -------------------------------- Calculating Sensitivities --------------------------------

print("--------- Calculating sensitivities --------")
tcav_auto.calculate_sensitivity([s["text"] for s in samples["test"]], [CLASSES[s["language"]] for s in samples["test"]])
tcav_fixed.calculate_sensitivity([s["text"] for s in samples["test"]], [CLASSES[s["language"]] for s in samples["test"]])
tcav_avg.calculate_sensitivity([s["text"] for s in samples["test"]], [CLASSES[s["language"]] for s in samples["test"]])

f_auto = open("./results/tcav_auto.txt", "a")
f_fixed = open("./results/tcav_fixed.txt", "a")
f_avg = open("./results/tcav_avg.txt", "a")

f_auto.write(json.dumps(tcav_auto.get_sensitivity_results(INV_CLASSES), indent=4)+ "\n")
f_fixed.write(json.dumps(tcav_fixed.get_sensitivity_results(INV_CLASSES), indent=4)+ "\n")
f_avg.write(json.dumps(tcav_avg.get_sensitivity_results(INV_CLASSES), indent=4)+ "\n")

f_auto.close()
f_fixed.close()
f_avg.close()

# Print sensitivities
print("Auto TCAV Sensitivities:")
tcav_auto.print_all_sensitivities(INV_CLASSES)

print("Fixed TCAV Sensitivities:")
tcav_fixed.print_all_sensitivities(INV_CLASSES)

print("Avg TCAV Sensitivities:")
tcav_avg.print_all_sensitivities(INV_CLASSES)

del tcav_avg, tcav_auto, tcav_fixed, samples
