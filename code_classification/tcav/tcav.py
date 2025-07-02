import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        self.cav = None
        self.sensitivity = None
        self.tcav_score = []
        self.y_labels = None
        self.bottleneck = None
        self.model_activations = {}
    
    def hook_fn(self, name):
        if name not in self.model_activations.keys():
            self.model_activations[name] = []
        def fn(module, input, output):
            self.model_activations[name].append(output)
        return fn

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
        print(layers)
        self.bottleneck = str(bottleneck)
        self.model.roberta.encoder.layer[bottleneck].register_forward_hook(self.hook_fn(str(bottleneck)))


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
            return self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def train_cav(self, x_concept):
        """ Calcola il vettore di attivazione concettuale """
        counterexamples = self._create_counterexamples(x_concept)
        x_train_concept = torch.cat((x_concept, counterexamples), dim=0)
        y_train_concept = torch.cat((torch.ones(x_concept.shape[0]), torch.zeros(counterexamples.shape[0])))

        # Tokenizza gli input se necessario
        
        x_train_concept = self._tokenize(x_train_concept)

        with torch.no_grad():
            _ = self.model(**x_train_concept)
            concept_activations = self.model_activations[self.bottleneck]
        
        lm = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
        lm.fit(concept_activations, y_train_concept.numpy())
        self.cav = -lm.coef_.T
        self.model_activations[self.bottleneck] = [] #once calculated all results, reset for next operations                           


    def calculate_sensitivity(self, x_train, y_train, device="cpu"):
        """
        Versione PyTorch della funzione, con commenti che rimandano
        ai passaggi originali in Keras.
        """
        x_train = self._tokenize(x_train)
        print(x_train)
        # --- (1) Predict di model_f, equivalente a model_f.predict(x_train)
        x_train = x_train.to(device)
        
        output = self.model(**x_train)

        activations = self.model_activations[self.bottleneck][0][0]  # Prendi l'ultima attivazione del bottleneck
        print(output.logits.shape)
        print(activations.shape)
        
        # --- (2) Reshape delle label, come reshape + tf.convert_to_tensor
        if isinstance(y_train, list):
            y_train = np.array(y_train)
            print(y_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.from_numpy(y_train)
        y_labels = y_train.view(-1).to(device)
        print(y_labels)

        # --- (3) Abilita il tracking dei gradienti su activations
        #     corrisponde in Keras a usare self.model_h.input con grafo attivo
        activations.requires_grad_(True)

        # --- (4) Forward pass in model_h e calcolo della loss
        #     k.binary_crossentropy → F.binary_cross_entropy
        
        loss = F.cross_entropy(output.logits, y_labels)

        # --- (5) Calcolo del gradiente di loss rispetto ad activations
        #     k.gradients + k.function + chiamata → torch.autograd.grad
        grads = torch.autograd.grad(loss, activations)[0]

        # --- (6) Prodotto scalare con CAV, equivalente a np.dot(calc_grad, self.cav)
        cav_tensor = torch.from_numpy(self.cav).float().to(device)
        sensitivity = torch.matmul(grads, cav_tensor)

        # --- (7) Salvataggio dei risultati come NumPy array
        self.sensitivity = sensitivity.detach().cpu().numpy()
        self.y_labels    = y_train.detach().cpu().numpy().reshape(-1)


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