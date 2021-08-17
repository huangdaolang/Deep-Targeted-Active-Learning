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
        data[key + '.' + str(v)] = np.array(val == v, dtype=int)

    data = data.drop(key, axis=1)

    return data


def get_covariates():
    data = pd.read_csv('../ACIC_dataset/x.csv', sep=',')
    data = categorical2indicator(data, 'x_2')
    data = categorical2indicator(data, 'x_21')
    data = categorical2indicator(data, 'x_24')
    for column in data.columns:
        data[column] = normalize(data[column])
    X = data[data.keys()[1:]].values
    return X


def get_ACIC_data(Ntrain, Nquery, Ntest, seed_split):
    X = get_covariates()
    N, dim = X.shape

    data = pd.read_csv('../ACIC_dataset/1/zymu_13.csv')
    Y = data.values[:, [1, 2]]
    D = data['z'].values

    np.random.seed(seed_split)

    idx = np.random.permutation(range(0, N))

    idx_tr = idx[Ntest:Ntrain + Ntest]
    idx_te = idx[0:Ntest]
    idx_qu = idx[Ntrain + Ntest:Ntrain + Nquery + Ntest]

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


class ACIC_dataset(Dataset):
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
    Nquery = config.Nquery
    seed_split = 1

    train, query, test = get_ACIC_data(Ntrain, Nquery, Ntest, seed_split)
    print(train['x'].shape)
    data_train = ACIC_dataset(train['x'], train['y'], train['d'])

    print(len(data_train))
