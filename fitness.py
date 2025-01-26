import encoding
import qsvm
import numpy as np
import os
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

def eval_acc(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    return(accuracy)

def eval_recall(y_true, y_pred):
    from sklearn.metrics import recall_score
    recall = recall_score(y_true, y_pred)
    return(recall)

def Dataset(X, y, test_size_split=0.2):
    class_labels = [r'0', r'1']

    n_samples = np.shape(X)[0]
    training_size = int(n_samples-(n_samples*test_size_split))
    test_size =int(n_samples-training_size)
    train_sample, test_sample, train_label, test_label = \
        train_test_split(X, y, stratify=y, test_size=test_size_split, random_state=12)

    std_scale = StandardScaler().fit(train_sample)
    train_sample = std_scale.transform(train_sample)
    test_sample = std_scale.transform(test_sample)

    samples = np.append(train_sample, test_sample, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    train_sample = minmax_scale.transform(train_sample)
    test_sample = minmax_scale.transform(test_sample)

    return train_sample, train_label, test_sample, test_label

class Fitness_acc:

    def __init__(self, nqubits, nparameters, X, y, debug=False):
        self.nqubits = nqubits
        self.nparameters = nparameters
        self.cc = encoding.CircuitConversor(nqubits, nparameters)
        self.X = X
        self.y = y
        self.debug = debug

    def __call__(self, POP):
        return self.fitness_acc(POP)

    def fitness_acc(self, POP):
        # Fitness function evaluating the fitness score for multiobjective - maximizing the accuracy
        #  and minimizing the number of gates 
        training_features, training_labels, test_features, test_labels = \
            Dataset(self.X, self.y)
        model = qsvm.QSVM(lambda parameters: self.cc(POP, parameters)[0],
                          training_features, training_labels)
        y_pred = model.predict(test_features) 
        acc = eval_acc(test_labels, y_pred)
        POP=''.join(str(i) for i in POP)
        _, gates = self.cc(POP, training_features[:,:])
        if self.debug:
            print(f'String: {POP}\n -> accuracy = {acc}, gates = {gates}')
        gate = gates/self.nqubits
        wc = gate + (gate*(acc**2))
        return wc, acc #
    

class Fitness_recall:

    def __init__(self, nqubits, nparameters, X, y, debug=False):
        self.nqubits = nqubits
        self.nparameters = nparameters
        self.cc = encoding.CircuitConversor(nqubits, nparameters)
        self.X = X
        self.y = y
        self.debug = debug

    def __call__(self, POP):
        return self.fitness_recall(POP)


    def fitness_recall(self, POP):
        # Fitness function evaluating the fitness score for multiobjective - maximizing the recall
        #  and minimizing the number of gates 
        training_features, training_labels, test_features, test_labels = \
            Dataset(self.X, self.y)
        model = qsvm.QSVM(lambda parameters: self.cc(POP, parameters)[0],
                        training_features, training_labels)
        y_pred = model.predict(test_features) # 22% del computo (ver abajo line-profiler)
        rec = eval_recall(test_labels, y_pred) # sklearn
        POP=''.join(str(i) for i in POP)
        _, gates = self.cc(POP, training_features[:,:])
        if self.debug:
            print(f'String: {POP}\n -> recall = {rec}, gates = {gates}')
        gate = gates/self.nqubits
        wc = gate + (gate*(rec**2))
        return wc, rec #

class Fitness_recall_single:

    def __init__(self, nqubits, nparameters, X, y, debug=False):
        self.nqubits = nqubits
        self.nparameters = nparameters
        self.cc = encoding.CircuitConversor(nqubits, nparameters)
        self.X = X
        self.y = y
        self.debug = debug

    def __call__(self, POP):
        return self.fitness_recall_single(POP)

    def fitness_recall_single(self, POP):
        # Fitness function evaluating the fitness score for singleobjective - maximizing the recall 
        training_features, training_labels, test_features, test_labels = \
            Dataset(self.X, self.y)
        model = qsvm.QSVM(lambda parameters: self.cc(POP, parameters)[0],
                        training_features, training_labels)
        try:
           y_pred = model.predict(test_features)
           rec = eval_recall(test_labels, y_pred)
        except Exception as e:
           if self.debug:
              print(f"Error during fitness evaluation: {e}")
              rec = 0.0
        return rec,
