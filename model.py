import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import math
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(28, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 1)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class Single_Model(nn.Module):
    def __init__(self):
        super(Single_Model, self).__init__()

        self.fc1 = nn.Linear(28, 30)
        self.fc2_1 = nn.Linear(30, 100)
        self.fc2_2 = nn.Linear(30, 100)
        self.fc3_1 = nn.Linear(100, 30)
        self.fc3_2 = nn.Linear(100, 30)
        self.fc4_1 = nn.Linear(30, 1)
        self.fc4_2 = nn.Linear(30, 1)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        d1 = self.fc2_1(x)
        d1 = F.relu(d1)
        d1 = self.dropout(d1)
        d1 = self.fc3_1(d1)
        d1 = F.relu(d1)
        d1 = self.dropout(d1)
        d1 = self.fc4_1(d1)

        d2 = self.fc2_2(x)
        d2 = F.relu(d2)
        d2 = self.dropout(d2)
        d2 = self.fc3_2(d2)
        d2 = F.relu(d2)
        d2 = self.dropout(d2)
        d2 = self.fc4_2(d2)
        return [d1, d2]


# variance propagation
def variance_product_rnd_vars(mean1, mean2, var1, var2):
    return mean1 ** 2 * var2 + mean2 ** 2 * var1 + var1 * var2


class UDropout(nn.Module):
    def __init__(self, rate, initial_noise=False):
        super(UDropout, self).__init__()
        self.initial_noise = initial_noise
        self.rate = rate
        self.dropout = nn.Dropout(rate)

    def _call_diag_cov(self, mean, var):
        if self.initial_noise:
            out = mean ** 2 * self.rate / (1 - self.rate)
        else:
            new_mean = 1 - self.rate
            new_var = self.rate * (1 - self.rate)
            out = variance_product_rnd_vars(mean, new_mean, var, new_var) / (1 - self.rate) ** 2
        return out

    def forward(self, inp):
        mean, var = inp
        return self.dropout(mean), self._call_diag_cov(mean, var)


class ULinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ULinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def _call_diag_cov(self, var):
        return F.linear(var, self.linear.weight ** 2)

    def forward(self, inp):
        mean, var = inp
        return self.linear(mean), self._call_diag_cov(var)


class UReLU(nn.Module):
    def __init__(self):
        super(UReLU, self).__init__()
        self.eps = 1e-8

    def _call_diag_cov(self, mean, var):
        std = torch.sqrt(var + self.eps)
        exp = mean / (np.sqrt(2.0) * std)
        erf_exp = torch.erf(exp)
        exp_exp2 = torch.exp(-1 * exp ** 2)
        term1 = 0.5 * (var + mean ** 2) * (erf_exp + 1)
        term2 = mean * std / (np.sqrt(2 * math.pi)) * exp_exp2
        term3 = mean / 2 * (1 + erf_exp)
        term4 = np.sqrt(1 / 2 / math.pi) * std * exp_exp2
        return F.relu(term1 + term2 - (term3 + term4) ** 2)

    def forward(self, inp):
        mean, var = inp
        return F.relu(mean), self._call_diag_cov(mean, var)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.fc1 = ULinear(28, 30)
        self.fc2_1 = ULinear(28, 100)
        self.fc2_2 = ULinear(28, 100)
        self.fc3_1 = ULinear(100, 30)
        self.fc3_2 = ULinear(100, 30)
        self.fc4_1 = ULinear(30, 1)
        self.fc4_2 = ULinear(30, 1)
        self.dropout = UDropout(config.dropout)
        self.relu = UReLU()

    def forward(self, inp):

        m1, v1 = self.fc2_1(inp)
        m1, v1 = self.relu((m1, v1))
        m1, v1 = self.dropout((m1, v1))
        m1, v1 = self.fc3_1((m1, v1))
        m1, v1 = self.relu((m1, v1))
        m1, v1 = self.dropout((m1, v1))
        m1, v1 = self.fc4_1((m1, v1))

        m2, v2 = self.fc2_2(inp)
        m2, v2 = self.relu((m2, v2))
        m2, v2 = self.dropout((m2, v2))
        m2, v2 = self.fc3_2((m2, v2))
        m2, v2 = self.relu((m2, v2))
        m2, v2 = self.dropout((m2, v2))
        m2, v2 = self.fc4_2((m2, v2))
        return [m1, m2], [v1, v2]
