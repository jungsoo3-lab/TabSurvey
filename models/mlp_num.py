import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel_torch import BaseModelTorch

'''
    Custom implementation for the standard multi-layer perceptron
'''


class MLPNUM(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.model = MLPNUM_Model(n_layers=self.params["n_layers"], input_dim=self.args.num_features,
                               bins = self.params["bins"],
                               hidden_dim=self.params["hidden_dim"], output_dim=self.args.num_classes,
                               task=self.args.objective)

        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float64)
        X_val = np.array(X_val, dtype=np.float64)

        return super().fit(X, y, X_val, y_val)

    def predict_helper(self, X):
        X = np.array(X, dtype=np.float64)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "bins": trial.suggest_int("bins", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.001)
        }
        return params

# numerical data transformation module
"""
code obtained from discussion https://stackoverflow.com/questions/65831101/is-there-a-split-equivalent-to-torch-nn-sequential
"""
class Split(torch.nn.Module):
    """
    https://stackoverflow.com/questions/65831101/is-there-a-parallel-equivalent-to-toech-nn-sequencial#65831101

    models a split in the network. works with convolutional models (not FC).
    specify out channels for the model to divide by n_parts.
    """
    def __init__(self, module, n_parts: int, dim=1):
        super().__init__()
        self._n_parts = n_parts
        self._dim = dim
        self._module = module

    def forward(self, inputs):
        output = self._module(inputs)
        chunk_size = output.shape[self._dim] // self._n_parts
        return torch.split(output, chunk_size, dim=self._dim)


class Unite(torch.nn.Module):
    """
    put this between two Splits to allow them to coexist in sequence.
    """
    def __init__(self):
        super(Unite, self).__init__()

    def forward(self, inputs):
        return torch.cat(inputs, dim=1)

class DataTransformer(torch.nn.Module):
    """
    Data transformer for a single feature
    """
    def __init__(self, input_dim, bins):
        super().__init__()
        self.bins = bins
        self.layer = nn.Linear(input_dim, bins)

    def forward(self, x):
        x = F.hardtanh(self.layer(x))
        return x



class MLPNUM_Model(nn.Module):

    def __init__(self, n_layers, input_dim, bins, hidden_dim, output_dim, task):
        super().__init__()
        self.input_dim = input_dim
        self.task = task
        self.layers = nn.ModuleList()
        #self.data_transform_layer = nn.ModuleList([])
        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim*bins, hidden_dim)
        self.data_transfomer_layer = [DataTransformer(1,bins) for _ in range(input_dim)]
        #self.split = Split(nparts = input_dim)
        self.unite = Unite()
        # Hidden Layers (number specified by n_layers)
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #x = self.split(x)
        x = [self.data_transfomer_layer[i](x[:,i]) for i in range(self.input_dim)]
        x = torch.flatten(x)
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x