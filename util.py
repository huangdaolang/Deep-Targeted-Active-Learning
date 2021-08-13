import copy
import numpy as np
import time
import config
from activelearning import choose_criterion, estimate_entropy
import warnings
import torch
from torch.utils.data import DataLoader
from model import Single_Model, UNet
import IHDP_preprocess

warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_nn_s(x, y, d):
    train_set = IHDP_preprocess.IHDP_dataset(x, y, d)
    train_loader = DataLoader(train_set, batch_size=5, shuffle=False, drop_last=False)
    model = UNet().to(device)
    model.double()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    model.train()
    for epoch in range(100):
        train_loss = 0
        for idx, data in enumerate(train_loader):
            label = data['label']
            feature = data['feature']
            decision = data['decision']
            feature, label = feature.to(device), label.to(device)
            variance = torch.zeros_like(feature)
            optimizer.zero_grad()
            output, variance = model((feature, variance))
            loss = 0
            for i, d in enumerate(decision):
                if int(d) == 1:
                    loss += criterion(output[1][i], label[i])
                else:
                    loss += criterion(output[0][i], label[i])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # print('[Epoch : %d] loss: %.3f' % (epoch + 1, train_loss / len(train_loader)))

    return model


def update_nn_s(model, x, y, d):
    train_set = IHDP_preprocess.IHDP_dataset(x, y, d)
    train_loader = DataLoader(train_set, batch_size=5, shuffle=False, drop_last=False)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    model.train()
    for epoch in range(10):
        train_loss = 0
        for idx, data in enumerate(train_loader, 0):
            label = data['label']
            feature = data['feature']
            decision = data['decision']
            feature, label = feature.to(device), label.to(device)
            variance = torch.zeros_like(feature)
            optimizer.zero_grad()
            output, variance = model((feature, variance))
            loss = 0
            for i, d in enumerate(decision):
                if int(d) == 1:
                    loss += criterion(output[1][i], label[i])
                else:
                    loss += criterion(output[0][i], label[i])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # print('[Epoch : %d] loss: %.3f' % (epoch + 1, train_loss / len(train_loader)))

    return model


def compute_dec(M, x_test):
    acc = 0
    Ntest = len(x_test['x'])

    K = 2
    test_set = IHDP_preprocess.IHDP_dataset(x_test['x'], x_test['y'], x_test['d'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)
    M.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            U = np.zeros(K)
            label = int(torch.argmax(data['label'], dim=1))
            feature = data['feature'].to(device)
            variance = torch.zeros_like(feature)
            for k in range(0, K):
                U[k] = float(M((feature, variance))[0][k])
            predict = int(np.argmax(U, axis=0))
            if predict == label:
                acc += 1
    print(acc / Ntest)
    return acc / Ntest


def active_learning(M0, train0, query0, test, Ndec, Nacq, criterion):
    M = copy.deepcopy(M0)
    train = train0.copy()
    query = query0.copy()

    dec = np.zeros(Nacq + 1, )
    H = np.zeros(Nacq + 1, )

    # Compute initial accuracy
    dec[0] = compute_dec(M, test)
    H[0] = estimate_entropy(M, test)

    print('Beginning active learning with \'' + criterion + '\' criterion')
    al_function = choose_criterion(criterion)

    for i in range(0, Nacq):
        # Select index of next query
        t1 = time.time()
        i_star = al_function(M, train, query, test)
        t2 = time.time()
        print('Data point selected... Took ' + str(t2 - t1) + ' sec.')
        dq = query['d'][i_star]

        # Add query to training set
        train['x'] = np.vstack((train['x'], query['x'][[i_star], :]))
        train['d'] = np.hstack((train['d'], dq))
        train['y'] = np.vstack((train['y'], query['y'][[i_star], :]))

        # Remove query from query set
        query['x'] = np.delete(query['x'], i_star, axis=0)
        query['d'] = np.delete(query['d'], i_star)
        query['y'] = np.delete(query['y'], i_star, axis=0)

        # Retrain model associated with decision dq
        M = update_nn_s(M, train['x'], train['y'], train['d'])

        # Compute accuracy on test set
        dec[i + 1] = compute_dec(M, test)
        # Compute posterior probability of optimal decision
        H[i + 1] = estimate_entropy(M, test)

    print('Active learning over')
    print('---')

    return dec, H
