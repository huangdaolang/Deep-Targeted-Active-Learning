import numpy as np
import pandas as pd
import config
from torch.utils.data import Dataset
import torch


def normalize(data):
    return (data-np.mean(data))/np.std(data)


def categorical2indicator(data, key):
    val = data[key].values
    uni_val = np.unique(val)

    for v in uni_val:
        data[key + '.' + str(v)] = np.array(val == v, dtype = int)
    
    data = data.drop(key, axis=1)
    
    return data


def get_covariates():
    data = pd.read_csv('inputs.csv', sep=',')

    data['bw'] = normalize(data['bw'])
    data['b.head'] = normalize(data['b.head'])
    data['preterm'] = normalize(data['preterm'])
    data['nnhealth'] = normalize(data['nnhealth'])
    data['momage'] = normalize(data['momage'])
    
    data['first'] = data['first'] - 1

    data = categorical2indicator(data, 'birth.o')
    X = data[data.keys()[1:]].values
    
    return X


def get_IHDP_data(Ntrain, Nquery, Ntest, seed_split):
    data = pd.read_csv('inputs.csv', sep=',')
    X = get_covariates()
    N, dim = X.shape

    D = data['treat'].values

    Y = pd.read_csv('counterfactual_outcomes.csv', sep=',').values
    Y = Y[:, [2, 3]]
    
    np.random.seed(config.seed)
    # p = 0.5*np.ones(3,)/3 + 0.5*np.random.dirichlet(np.ones(3,))
    # D = np.random.randint(0, 3, size = (N,))
    
    np.random.seed(seed_split)
    
    idx = np.random.permutation(range(0, N))

    idx_tr = idx[Ntest:Ntrain+Ntest]
    idx_te = idx[0:Ntest]

    idx_qu = idx[Ntrain+Ntest:Ntrain+Nquery+Ntest]
    
    train = {
        'x': X[idx_tr, :],
        'y': np.reshape(Y[idx_tr, D[idx_tr]], (Ntrain, 1)),
        # 'y_c': np.reshape(Y[idx_tr, 1- D[idx_tr]], (Ntrain, 1)),  # counterfactual outcome
        'd': D[idx_tr]
    }

    query = {
        'x': X[idx_qu, :],
        'y': np.reshape(Y[idx_qu, D[idx_qu]], (Nquery, 1)),
        'd': D[idx_qu]
    }

    test = {
        'x': X[idx_te, :],
        'y': np.reshape(Y[idx_te, :], (Ntest, 2)),
        'd': np.argmax(np.reshape(Y[idx_te, :], (Ntest, 2)), axis=1)
    }

    return train, query, test


class IHDP_dataset(Dataset):
    def __init__(self, x, y, d):
        self.feature = torch.from_numpy(x)
        self.label = torch.from_numpy(y)
        self.decision = d

    def __getitem__(self, index):
        entry = {"feature": self.feature[index], "label": self.label[index], "decision": self.decision[index]}
        return entry

    def __len__(self):
        return len(self.feature)


if __name__ == "__main__":
    Ntrain = config.Ntrain
    Ntest = config.Ntest
    print(Ntest)
    Nquery = config.Nquery
    seed_split = 1

    train, query, test = get_IHDP_data(Ntrain, Nquery, Ntest, seed_split)
    print(train['x'].shape)
    data_train = IHDP_dataset(train['x'], train['y'], train['d'])

    print(len(data_train))