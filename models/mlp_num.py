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
                               bins = [self.params[f"bins_{i}"] for i in range(self.args.num_features) ],
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
            #"bins": trial.suggest_int("bins", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.001)
        }
        for i in range(args.num_features):
            params[f"bins_{i}"] = trial.suggest_int("bins", 2, 10)
        return params

# numerical data transformation module
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
        self.input_layer = nn.Linear(input_dim*sum(bins), hidden_dim)
        self.data_transformer_layer = [DataTransformer(1, bins[i]) for i in range(input_dim)]
        # Hidden Layers (number specified by n_layers)
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # data transformation layer
        x = [self.data_transformer_layer[i](x[:,i].reshape((x.shape[0], 1))) for i in range(len(self.data_transformer_layer))]
        x = torch.cat(x, dim = 1)

        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x