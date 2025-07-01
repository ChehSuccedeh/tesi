import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import SGDClassifier
import random
import string

class TCAV:
    """ Classe per vettori di attivazione concettuale per modelli PyTorch """

    def __init__(self, model=None, tokenizer=None):
        """ Inizializza la classe con variabili vuote """
        self.model = model
        self.tokenizer = tokenizer
        self.model_f = None
        self.model_h = None
        self.cav = None
        self.sensitivity = None
        self.tcav_score = []
        self.y_labels = None

    def set_model(self, model):
        """ Imposta il modello PyTorch """
        self.model = model

    def set_tokenizer(self, tokenizer):
        """ Imposta il tokenizer """
        self.tokenizer = tokenizer

    def split_model(self, bottleneck):
        """ Divide il modello su un dato layer di bottleneck """
        if bottleneck < 0 or bottleneck >= len(list(self.model.children())):
            raise ValueError("Il layer di bottleneck deve essere valido!")
        
        layers = list(self.model.children())
        self.model_f = nn.Sequential(*layers[:bottleneck+1])
        self.model_h = nn.Sequential(*layers[bottleneck+1:])

    def _create_counterexamples(self, x_concept):
        """ Crea esempi casuali come controesempi """
        n = x_concept.shape[0]
    
        counterexamples = []
        for i in range(n):
            length = x_concept[i].shape[0]
            counterexamples.append(random.choices(string.printable, k=length))
        return np.array(counterexamples)

    def _tokenize(self, inputs):
        """ Tokenizza gli input se il tokenizer è fornito """
        if self.tokenizer is not None:
            # Assumiamo che il tokenizer restituisca tensori PyTorch
            return self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.model_f.device)
        return inputs

    def train_cav(self, x_concept):
        """ Calcola il vettore di attivazione concettuale """
        counterexamples = self._create_counterexamples(x_concept)
        x_train_concept = torch.cat((x_concept, counterexamples), dim=0)
        y_train_concept = torch.cat((torch.ones(x_concept.shape[0]), torch.zeros(counterexamples.shape[0])))

        # Tokenizza gli input se necessario
        x_train_concept = self._tokenize(x_train_concept)

        with torch.no_grad():
            concept_activations = self.model_f(x_train_concept).cpu().numpy()
        
        lm = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
        lm.fit(concept_activations, y_train_concept.numpy())
        self.cav = -lm.coef_.T

    def calculate_sensitivity(self, x_train, y_train):
        """ Calcola e restituisce la sensibilità per ogni label (multi-label) """
        # Tokenizza gli input se necessario
        x_train = self._tokenize(x_train)
        x_train.requires_grad = True
        model_f_activations = self.model_f(x_train)

        criterion = nn.BCEWithLogitsLoss()
        output = self.model_h(model_f_activations).squeeze()

        if output.dim() == 1:
            output = output.unsqueeze(1)
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(1)

        grads = []

        for i in range(y_train.shape[1]):
            self.model.zero_grad()
            if x_train.grad is not None:
                x_train.grad.zero_()
            loss = criterion(output[:, i], y_train[:, i].float())
            loss.backward(retain_graph=True)
            grad = x_train.grad.detach().clone()
            grads.append(grad.unsqueeze(2))  # (batch, features, 1)
            x_train.grad.zero_()

        grads = torch.cat(grads, dim=2)  # (batch, features, num_labels)
        cav_tensor = torch.tensor(self.cav, dtype=torch.float)
        if cav_tensor.dim() == 1:
            cav_tensor = cav_tensor.unsqueeze(1)
        self.sensitivity = torch.einsum('bfi,fi->bi', grads, cav_tensor)  # (batch, num_labels)
        self.y_labels = y_train.cpu().numpy()

    def print_sensitivity(self):
        """ Stampa le sensibilità per tutte le label in modo leggibile """
        if isinstance(self.y_labels, list):
            self.y_labels = np.array(self.y_labels)

        num_labels = self.y_labels.shape[1] if len(self.y_labels.shape) > 1 else 1

        for label_idx in range(num_labels):
            y_label = self.y_labels[:, label_idx] if num_labels > 1 else self.y_labels
            sensitivity_label = self.sensitivity[:, label_idx] if num_labels > 1 else self.sensitivity

            for class_value in np.unique(y_label):
                mask = y_label == class_value
                if np.sum(mask) > 0:
                    perc = np.sum(sensitivity_label[mask] > 0) / np.sum(mask)
                    print(f"Sensibilità label {label_idx} classe {class_value}: {perc:.3f}")