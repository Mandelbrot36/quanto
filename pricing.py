import numpy as np
from scipy.stats import norm


def option_price(S, K, t, r, sigma, gold_0, option_type, D=0):

    K = K * gold_0

    if t <= 0:
        # Dla wygasłej opcji
        if option_type == 'call':
            return (100 / gold_0) * np.maximum(S - K, 0)
        else:  # put
            return (100 / gold_0) * np.maximum(K - S, 0)

    d1 = (np.log(S / K) + (r - D + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type == 'call':
        return (100 / gold_0) * (S * np.exp(-D*t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2))
    else:  # put
        return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_delta(S, S2, K, t, r, sigma, gold_0, option_type, D=0):

    K = K * gold_0

    d1 = (np.log(S / K) + (r - D + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    if option_type == 'call':
        return (100 / gold_0) * np.exp(-D * t) * norm.cdf(d1) / S2
    elif option_type == 'put':
        return (100 / gold_0) * np.exp(-D * t) * (norm.cdf(d1) - 1.0) / S2



