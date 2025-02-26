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

# synthetic data generation
seed=42
random_state = np.random.RandomState(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

X = np.load(f'./X.npy', allow_pickle=True)
ground_truth = np.load(f'./ground_truth.npy', allow_pickle=True).reshape(-1,1)
# train norm enhancer
normalizers = {'z_score':StandardScaler(), 'quantile':QuantileTransformer(), 'power':PowerTransformer(), 'robust':RobustScaler(), 'minmax':MinMaxScaler()}
inputs = dict()
for norm in normalizers.keys():
    X_ = X.copy()
    normalizer = normalizers[norm]
    X_0 = normalizer.fit_transform(X_[:,0].reshape(-1,1))
    normalizer = normalizers[norm]
    X_1 = normalizer.fit_transform(X_[:,1].reshape(-1,1))
    X_ = np.concatenate([X_0, X_1], axis=-1)
    inputs[norm] = torch.from_numpy(X_)

inputs_ = inputs.copy()

mlp = Mlp(X.shape[1], 128)
iters = 3

max_epoch = 10

best_keys = []
best_models = []

ground_truth = torch.from_numpy(ground_truth)
ground_truth_eval = ground_truth.copy()
ground_truth_eval = torch.from_numpy(ground_truth_eval)

dataloaders = dict()
for norm in normalizers.keys():
    dataset = TensorDataset(inputs[norm].float(), ground_truth.float())
    dataloaders[norm] = DataLoader(dataset, batch_size=inputs[norm].shape[0], shuffle=False)

for i in range(iters):
    models = dict()
    for norm in normalizers.keys():
        models[norm] = copy.deepcopy(mlp)
    np.save(f'./ground_truth_{i}', ground_truth.numpy())
    evaluations = dict()
    for norm in normalizers.keys():
        running_loss = 0.0
        model = models[norm]
        model.cuda()
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        dataloader = dataloaders[norm]
        for epoch in range(max_epoch):
            for bid, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                loss = F.mse_loss(outputs, labels)
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        running_loss = running_loss / max_epoch
        evaluations[norm] = running_loss
        models[norm] = model
        
    best_key = min(evaluations, key=evaluations.get)
    best_keys.append(best_key)
    best_models.append(models[best_key])
    
    dataloader = dataloaders[best_key]
    loss_eval = 0.0
    
    ground_truth_ = []
    for bid, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = models[best_key](inputs)
        loss = F.mse_loss(outputs, labels)
        loss_eval += loss
        ground_truth_.append(labels - outputs)
        
    print(f'iter {i} best_norm:{best_key} mse_loss:{loss_eval / len(dataloader)}')
    ground_truth = torch.cat(ground_truth_, dim=0).data.cpu()

    for norm in normalizers.keys():
        dataset = TensorDataset(inputs_[norm].float(), ground_truth.float())
        dataloaders[norm] = DataLoader(dataset, batch_size=inputs_[norm].shape[0], shuffle=False)


final_pred = 0.0    
for i in range(iters):
    best_key = best_keys[i]
    dataloader = dataloaders[best_key]
    model = best_models[i]
    outputs_ = []
    for bid, (inputs,_) in enumerate(dataloader):
        inputs = inputs.cuda()
        outputs = model(inputs)
        outputs_.append(outputs)
    outputs = torch.cat(outputs_, dim=0)
    final_pred += outputs
ground_truth_eval = ground_truth_eval.cuda().float()
print(f'final mse_loss:{F.mse_loss(final_pred, ground_truth_eval)}')
