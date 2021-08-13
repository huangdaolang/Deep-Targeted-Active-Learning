import numpy as np
import pandas as pd
import config
from torch.utils.data import Dataset
import torch


def categorical2indicator(data, key):
    val = data[key].values
    uni_val = np.unique(val)

    for v in uni_val:
        data[key + '.' + str(v)] = np.array(val == v, dtype=int)

    data = data.drop(key, axis=1)

    return data


def get_OAI_data(Ntrain, Ntest, imag, seed_split):
    # Read
    data = pd.read_csv('dataset.csv', sep=',')
    # Check imaging
    if imag is False:
        print('Not using imaging data')
        data = data.drop('BL_JSW', axis=1)
        data = data.drop('BL_OA', axis=1)
    else:
        print('Using imaging data')
        data = categorical2indicator(data, 'BL_OA')

    data = categorical2indicator(data, 'VARVAL')

    L = list(data.keys())
    L.remove('ID')
    L.remove('D')
    L.remove('Y')

    X = data[L].values
    N, dim = X.shape

    D = data['D'].values - 1
    Y = data['Y'].values

    np.random.seed(seed_split)
    idx = np.random.permutation(range(0, N))

    idx_tr = idx[Ntest:Ntrain + Ntest]
    idx_qu = idx[Ntrain + Ntest:]
    idx_te = idx[0: Ntest]

    train = {
        'x': X[idx_tr, :],
        'y': np.reshape(Y[idx_tr], (Ntrain, 1)),
        'd': D[idx_tr]
    }

    query = {
        'x': X[idx_qu, :],
        'y': np.reshape(Y[idx_qu], (N - Ntrain - Ntest, 1)),
        'd': D[idx_qu]
    }

    test = {
        'x': X[idx_te, :],
        'y': np.reshape(Y[idx_te], (Ntest, 1)),
        'd': D[idx_te]
    }

    return train, query, test


class OAI_dataset(Dataset):
    def __init__(self, data):
        self.feature = torch.from_numpy(data['x'])
        self.decision = data['d']
        self.label = torch.from_numpy(data['y'])

    def __getitem__(self, index):
        entry = {"feature": self.feature[index], "label": self.label[index], "decision": self.decision[index]}
        return entry

    def __len__(self):
        return len(self.feature)


if __name__ == "__main__":
    Ntrain = config.Ntrain
    Ntest = config.Ntest
    imag = config.imag
    seed_split = 1

    train, query, test = get_OAI_data(Ntrain, Ntest, imag, seed_split)

    data_train = OAI_dataset(train)

    print(len(data_train))
