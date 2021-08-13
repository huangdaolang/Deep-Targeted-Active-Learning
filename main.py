import numpy as np
import sys
import util
import config
import IHDP_preprocess
import os


def main(Ntrain, Nquery, Ntest, Nacq, seed):
    print('Simulation with IHDP data - ' + str(config.Ntrain) + ' training points, ' + str(config.Nquery) +
          ' query points, ' + str(config.Ntest) + ' test point.')
    print('Seed split = ' + str(seed))
    
    print('-----')

    train, query, test = IHDP_preprocess.get_IHDP_data(Ntrain, Nquery, Ntest, seed)

    Ndec = 2
    model = util.initialize_nn_s(train['x'], train['y'], train['d'])

    acc = np.zeros((5, Nacq+1))
    H = np.zeros((5, Nacq+1))
    acc[0, :], H[0, :] = util.active_learning(model, train, query, test, Ndec, Nacq, 'random')
    acc[1, :], H[1, :] = util.active_learning(model, train, query, test, Ndec, Nacq, 'uncertainty')
    acc[2, :], H[2, :] = util.active_learning(model, train, query, test, Ndec, Nacq, 'decision_uncertainty')
    acc[3, :], H[3, :] = util.active_learning(model, train, query, test, Ndec, Nacq, 'targeted_eig')
    acc[4, :], H[4, :] = util.active_learning(model, train, query, test, Ndec, Nacq, 'decision_eig')

    print("acc", acc)
    print("entropy", H)

    print('Saving results...')
    root_name = 'Sim/Single_VP_dropout0.05_GH10'
    if not os.path.exists(root_name):
        os.mkdir(root_name)
    np.save(root_name + '/acc_' + str(seed) + '.npy', acc)
    np.save(root_name + '/H_' + str(seed) + '.npy', H)

    return None


if __name__ == "__main__":
    Nacq = config.Nacq
    Ntrain = config.Ntrain
    Nquery = config.Nquery
    Ntest = config.Ntest
    seed = int(sys.argv[1])

    main(Ntrain, Nquery, Ntest, Nacq, seed)
