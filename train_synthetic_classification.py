import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, FunctionTransformer, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from numpy import percentile
from torch.utils.data import dataset
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import random


def accuracy(x, y):
    x = np.concatenate((1-x, x), 1)
    x = np.argmax(x, axis=-1)
    return accuracy_score(y, x)

def generate_res_label(x, y):
    x = np.concatenate((1-x, x), 1)
    x = np.argmax(x, axis=-1).reshape(-1)
    y = y.reshape(-1)
    return (x==y).astype(int)

class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.layers(x)

# set seed
seed=42
random_state = np.random.RandomState(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

X = np.load(f'./X.npy', allow_pickle=True)
ground_truth = np.load(f'./ground_truth.npy', allow_pickle=True).reshape(-1,1)

# Define the number of inliers and outliers
# n_samples = 1000
# outliers_fraction = 0.2

# xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))

# n_inliers = int((1. - outliers_fraction) * n_samples)
# n_outliers = int(outliers_fraction * n_samples)
# ground_truth = np.zeros(n_samples, dtype=int)
# ground_truth[-n_outliers:] = 1
# random_state = np.random.RandomState(42)

# X = 0.3*np.random.randn(n_inliers, 2)
# X1 = 0.3*np.random.randn(n_outliers//4, 2)
# X1[:,0] = X1[:,0]-3
# X1[:,1] = X1[:,1]+3
# X2 = 0.3*np.random.randn(n_outliers//4, 2)
# X2[:,0] = X2[:,0]+3
# X2[:,1] = X2[:,1]+3
# X3 = 0.3*np.random.randn(n_outliers//4, 2)
# X3[:,0] = X3[:,0]-3
# X3[:,1] = X3[:,1]-3
# X4 = 0.3*np.random.randn(n_outliers//4, 2)
# X4[:,0] = X4[:,0]+3
# X4[:,1] = X4[:,1]-3


# X = np.r_[X, X1, X2, X3, X4]

# sample =np.hstack([X,ground_truth.reshape(-1,1)])
print(X.shape, ground_truth.shape)
sample =np.concatenate([X,ground_truth.reshape(-1,1)],axis=-1)
n_samples = sample.shape[0]

normalizers = {'z_score':StandardScaler(), 'quantile':QuantileTransformer(), 'power':PowerTransformer(), 'function':FunctionTransformer(), 'robust':RobustScaler(), 'minmax':MinMaxScaler()}
inputs = dict()
for norm in normalizers.keys():
    X_ = sample[:,:-1].copy()
    normalizer = normalizers[norm]
    X_0 = normalizer.fit_transform(X_[:,0].reshape(-1,1))
    normalizer = normalizers[norm]
    X_1 = normalizer.fit_transform(X_[:,1].reshape(-1,1))
    X_ = np.concatenate([X_0, X_1], axis=-1)
    inputs[norm] = torch.from_numpy(X_)

inputs_ = inputs.copy()

mlp = Mlp(X.shape[1], 32)

max_epoch = 1

best_keys = []
best_models = []

ground_truth = torch.from_numpy(ground_truth)


dataloaders = dict()
for norm in normalizers.keys():
    dataset = TensorDataset(inputs[norm].float(), ground_truth)
    # dataloaders[norm] = DataLoader(dataset, batch_size = n_samples, shuffle = True)
    dataloaders[norm] = DataLoader(dataset, batch_size = 128, shuffle = True)

# train
models = dict()
for norm in normalizers.keys():
    models[norm] = copy.deepcopy(mlp)
evaluations = dict()
for norm in normalizers.keys():
    training_loss = 0.0
    model = models[norm]
    model.cuda()
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criteria = nn.BCEWithLogitsLoss()
    dataloader = dataloaders[norm]
    for epoch in range(max_epoch):
        running_loss = 0.0
        for bid, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda().reshape(-1,1)
            outputs = model(inputs)
            loss = criteria(outputs,labels.float())
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss = running_loss / len(dataloader)
        training_loss = training_loss + running_loss
    training_loss = training_loss / max_epoch
    evaluations[norm] = training_loss
    models[norm] = model
    ####  start evaluation
    model.eval()
    eval_output = model(inputs_[norm].float().cuda())
    eval_accuracy = accuracy(eval_output.detach().cpu().numpy(),ground_truth.numpy())
    res_label = generate_res_label(eval_output.detach().cpu().numpy(),ground_truth.numpy())
    np.save(f'./res_continues_{norm}', ground_truth.numpy() - eval_output.detach().cpu().numpy())
    np.save(f'./res_label_{norm}', res_label)
    np.save(f'./input_{norm}', inputs_['function'].numpy())
    print(f'{norm}`s accuray:{eval_accuracy}')
