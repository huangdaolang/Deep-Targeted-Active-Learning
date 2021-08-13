import numpy as np
import GPy
import config


def gen_decision(Ntrain, Nquery, Ntest, Ndec):
    np.random.seed(config.seed)
    Ntot = Ntrain + Nquery + Ntest
    p = (0.65/Ndec)*np.ones(Ndec,) + 0.35*np.random.dirichlet(np.ones(Ndec,))
    D = np.random.choice(range(0, Ndec), Ntot, replace=True, p=p)

    return D


def generation(dim, std, Ndec, Ntrain, Nquery, Ntest, seed_split):
    np.random.seed(config.seed)
    Ntot = Ntrain + Nquery + Ntest
    
    X = np.random.randn(Ntot, dim)
    D = gen_decision(Ntrain, Nquery, Ntest, Ndec)
    Y = np.zeros((Ntot, Ndec))

    # length and variance of RBF kernel
    l = np.sqrt(dim) * (0.25 + 0.75*np.random.rand(Ndec, dim))
    v = 0.5 + 2*np.random.rand(Ndec,)

    for i in range(0, Ndec):
        kernel = GPy.kern.RBF(input_dim=dim, lengthscale=l[i, :], variance=v[i], ARD=True)
        C = kernel.K(X, X)
        Y[:, i] = np.random.multivariate_normal(np.zeros(Ntot), C, size=1) + std*np.random.randn(Ntot)

    np.random.seed(seed_split)
    
    idx_tot = range(0, Ntot)
    
    idx_tr = np.zeros(Ntrain, dtype='int64')
    # ensure at least one decision included in train?
    for i in range(0, Ndec):
        idx_d = np.where(D == i)[0]
        idx_tr[i] = np.random.choice(idx_d, size=1, replace=False)
    idx_tot = np.delete(idx_tot, idx_tr[:Ndec])

    idx_tmp = np.random.choice(range(0, len(idx_tot)), size=Ntrain - Ndec, replace=False)
    idx_tr[Ndec:] = idx_tot[idx_tmp]
    idx_tot = np.delete(idx_tot, idx_tmp)
    
    idx_tot = np.random.permutation(idx_tot)
    idx_qu = idx_tot[:Nquery]
    idx_te = idx_tot[Nquery:]

    train_x = X[idx_tr, :]
    train_d = D[idx_tr]
    train_y = np.reshape(Y[idx_tr, train_d], (Ntrain, 1))

    query_x = X[idx_qu, :]
    query_d = D[idx_qu]
    query_y = np.reshape(Y[idx_qu, query_d], (Nquery, 1))

    test_x = X[idx_te, :]
    test_d = np.argmax(Y[idx_te, :], axis=1)
    test_y = np.reshape(Y[idx_te, test_d], (Ntest, 1))

    train = {
    'x': train_x,
    'y': train_y,
    'd': train_d
        }
    query = {
        'x': query_x, 
        'y': query_y,
        'd': query_d
    }
    test = {
        'x': test_x,
        'd': test_d,
        'y': test_y
    }

    return train, query, test


if __name__ == "__main__":
    dim = 5
    std = 0.25
    Nacq = config.Nacq
    Ndec = 2
    Ntrain = config.Ntrain
    Nquery = config.Nquery
    Ntest = config.Ntest
    seed = 1

    train, query, test = generation(dim, std, Ndec, Ntrain, Nquery, Ntest, seed)