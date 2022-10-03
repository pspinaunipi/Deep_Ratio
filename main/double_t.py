from scipy.stats import t
from scipy.special import digamma
from scipy.optimize import least_squares
import numpy as np

def func(x1, x0, g, w):
    val = np.log(x1 / 2) - digamma(x1 / 2) + 1 - np.log((x0 + 1) / 2) \
            + digamma((x0 + 1) / 2) + (g * (np.log(w) - w)).sum() / g.sum()
    return val


def EM(p0, data, iterations=10):
    df1, df2, mu1, mu2, std1, std2, p = p0
    N = data.shape[0]
    for _ in range(iterations):
        w_1 =  (df1 +1) * std1**2 / (df1 * std1**2 +(data - mu1)**2)
        w_2 =  (df2 +1) * std2**2 / (df2 * std2**2 +(data - mu2)**2)

        gamma = p * t.pdf(data, df2, mu2, std2) / \
                    ((1 - p) * t.pdf(data, df1, mu1, std1) + p * t.pdf(data, df2, mu2, std2))

        p = gamma.sum() / N
        mu1 = ((1 - gamma) * w_1 * data).sum() / ((1 - gamma) * w_1).sum()
        std1 = (((1 - gamma) * w_1 * (data - mu1)**2).sum() / ((1 - gamma).sum()))**.5

        df1 = least_squares(func, df1, args = (df1, 1 - gamma, w_1), bounds =(0, np.inf)).x[0]

        mu2 = (gamma * w_2 * data).sum() / (gamma * w_2).sum()
        std2 = ((gamma * w_2 * (data - mu2)**2).sum() / (gamma.sum()))**.5

        df2 = least_squares(func, df2, args = (df1, gamma, w_2), bounds =(0, np.inf)).x[0]

    return df1, df2, mu1, mu2, std1, std2, p
