import copy
import numpy as np
import torch
import config
import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_criterion(criterion):
    if criterion == "random":
        return random_sampling
    elif criterion == "uncertainty":
        return uncertainty_sampling
    elif criterion == "decision_uncertainty":
        return decision_uncertainty_sampling
    elif criterion == "targeted_eig":
        return targeted_eig
    elif criterion == "decision_eig":
        return decision_eig
    else:
        print("Active learning criterion not specified correctly")
        return


def estimate_entropy(M, X):
    # N = X['x'].shape[0]
    # Ns = config.Ns
    # eps = np.spacing(1)
    # K = 2
    # samples = np.zeros((Ns, N, K))
    # x = torch.from_numpy(X['x']).to(device)
    # M.train()
    # with torch.no_grad():
    #     for i in range(Ns):
    #         for k in range(0, K):
    #             samples[i, :, k] = M(x)[k].reshape(-1)
    #
    # dbest = np.argmax(samples, axis=2)
    # H = np.zeros(N, )
    # for i in range(0, N):
    #     p = np.bincount(dbest[:, i], minlength=K) / Ns
    #     p[p == 0] = eps
    #     H[i] = np.sum(-p * np.log(p))
    # return np.sum(H)
    M.eval()
    N = X['x'].shape[0]
    Ns = config.Ns
    eps = np.spacing(1)
    K = 2
    mu = np.zeros((N, K))
    var = np.zeros((N, K))
    x = torch.from_numpy(X['x']).to(device)
    init_var = torch.zeros_like(x)
    [m0, m1], [v0, v1] = M((x, init_var))
    mu[:, [0]] = m0.detach().numpy()
    mu[:, [1]] = m1.detach().numpy()
    var[:, [0]] = v0.detach().numpy()
    var[:, [1]] = v1.detach().numpy()
    samples = mu + np.sqrt(var) * np.random.randn(Ns, N, K)
    dbest = np.argmax(samples, axis=2)

    H = np.zeros(N, )
    for i in range(0, N):
        p = np.bincount(dbest[:, i], minlength=K) / Ns
        p[p == 0] = eps
        H[i] = np.sum(-p * np.log(p))

    return np.sum(H)


def random_sampling(M, train, query, test):
    n = len(query['x'])
    return np.random.randint(0, n)


def uncertainty_sampling(M, train, query, test):
    # n = len(query['x'])
    # M.train()
    # with torch.no_grad():
    #     out = np.zeros((n, config.Ns))
    #     for i in range(config.Ns):
    #         for j in range(0, n):
    #             x = torch.from_numpy(query['x'][[j], :]).to(device)
    #             d = query['d'][j]
    #             out[j, i] = M(x)[d].reshape(-1)
    #     query_var = np.var(out, axis=1)
    #
    # return np.argmax(query_var)
    M.eval()
    n = len(query['x'])
    with torch.no_grad():
        query_var = np.zeros(n)
        for i in range(0, n):
            x = torch.from_numpy(query['x'][[i], :]).to(device)
            init_var = torch.zeros_like(x)
            query_var[i] = M((x, init_var))[1][query['d'][i]].reshape(-1).detach().numpy()
    return np.argmax(query_var)


def decision_uncertainty_sampling(M, train, query, test):
    return np.argmax(estimate_entropy(M, query))


def targeted_eig(M, train, query, test):
    # Nq_loc = len(query['x'])
    # Ns = config.Ns
    # exp_h = np.zeros(Nq_loc, )
    # N = test['x'].shape[0]
    # test_x = torch.from_numpy(test['x']).to(device)
    # M.eval()
    # for i in range(0, Nq_loc):
    #     xq = query['x'][[i], :]
    #     dq = query['d'][i]
    #
    #     x_plus = np.vstack((train['x'], xq))
    #     d_plus = np.append(train['d'], dq)
    #
    #     y_gh_t = []
    #     xq = torch.from_numpy(xq).to(device)
    #     for j in range(10):
    #         y_gh_t.append(M(xq)[dq].reshape(-1))
    #
    #     for y in y_gh_t:
    #         m = copy.deepcopy(M)
    #         y = y.detach().numpy()
    #         y_plus = np.vstack((train['y'], y))
    #         m = util.update_nn_s(m, x_plus, y_plus, d_plus)
    #         m.train()
    #         with torch.no_grad():
    #             out = np.zeros((N, Ns))
    #             for a in range(Ns):
    #                 out[:, [a]] = m(test_x)[dq].cpu().numpy()
    #             v = np.var(out, axis=1)
    #         exp_h[i] += np.sum(np.log(np.sqrt(2 * np.pi * np.e * v)))
    #
    # return np.argmin(exp_h)
    M.eval()
    Nq_loc = len(query['x'])
    exp_h = np.zeros(Nq_loc, )

    N_GH = 10
    y_gh, w_gh = np.polynomial.hermite.hermgauss(N_GH)
    test_x = torch.from_numpy(test['x']).to(device)
    for i in range(0, Nq_loc):
        xq = query['x'][[i], :]
        dq = query['d'][i]

        idx = np.where(train['d'] == dq)[0]
        x_plus = np.vstack((train['x'][idx, :], xq))
        d_plus = np.append(train['d'], dq)
        xq = torch.from_numpy(xq)
        vq = torch.zeros_like(xq)
        mu, var = M((xq, vq))
        y_gh_t = np.sqrt(2 * var[dq].detach().numpy()).reshape(-1) * y_gh + mu[dq].detach().numpy().reshape(-1)

        for j, y in enumerate(y_gh_t):
            m = copy.deepcopy(M)
            y_plus = np.vstack((train['y'], y))
            m = util.update_nn_s(m, x_plus, y_plus, d_plus)
            test_var = torch.zeros_like(test_x)
            _, v = m((test_x, test_var))

            exp_h[i] = exp_h[i] + np.sum(w_gh[j] * np.log(np.sqrt(2 * np.pi * np.e * v[dq].detach().numpy())))

    return np.argmin(exp_h)


def decision_eig(M, train, query, test):
    # Nq_loc = len(query['x'])
    # exp_h = np.zeros(Nq_loc, )
    #
    # for i in range(0, Nq_loc):
    #     xq = query['x'][[i], :]
    #     dq = query['d'][i]
    #
    #     x_plus = np.vstack((train['x'], xq))
    #     d_plus = np.append(train['d'], dq)
    #
    #     y_gh_t = []
    #     xq = torch.from_numpy(xq).to(device)
    #     M.train()
    #     for j in range(10):
    #         y_gh_t.append(M(xq)[dq].reshape(-1))
    #
    #     for y in y_gh_t:
    #         m = copy.deepcopy(M)
    #         y = y.detach().numpy()
    #         y_plus = np.vstack((train['y'], y))
    #         m = util.update_nn_s(m, x_plus, y_plus, d_plus)
    #         exp_h[i] += estimate_entropy(m, test)
    #
    # return np.argmin(exp_h)
    M.eval()
    Nq_loc = len(query['x'])
    exp_h = np.zeros(Nq_loc, )

    N_GH = 10
    y_gh, w_gh = np.polynomial.hermite.hermgauss(N_GH)
    for i in range(0, Nq_loc):
        xq = query['x'][[i], :]
        dq = query['d'][i]

        idx = np.where(train['d'] == dq)[0]
        x_plus = np.vstack((train['x'][idx, :], xq))
        d_plus = np.append(train['d'], dq)
        xq = torch.from_numpy(xq)
        vq = torch.zeros_like(xq)
        mu, var = M((xq, vq))
        y_gh_t = np.sqrt(2 * var[dq].detach().numpy()).reshape(-1) * y_gh + mu[dq].detach().numpy().reshape(-1)

        for j, y in enumerate(y_gh_t):
            m = copy.deepcopy(M)
            y_plus = np.vstack((train['y'], y))
            m = util.update_nn_s(m, x_plus, y_plus, d_plus)

            exp_h[i] = exp_h[i] + w_gh[j] * estimate_entropy(m, test)

    return np.argmin(exp_h)