from math import gamma
import numpy as np
import pandas as pd
from pricing import *
from scipy.stats import norm


def adjust_vanilla_delta(vanilla_delta, delta_margin):
    adjusted_delta = vanilla_delta.copy()
    mask_near_1 = np.abs(1 - vanilla_delta) < delta_margin
    adjusted_delta[mask_near_1] = 1
    mask_near_0 = np.abs(vanilla_delta) < delta_margin
    adjusted_delta[mask_near_0] = 0

    return adjusted_delta


def adjust_binary_delta(binary_delta, vanilla_delta, delta_margin):
    cutoff_binary_delta = np.where(
        (np.abs(vanilla_delta) > delta_margin) |
        (np.abs(1 - vanilla_delta) > delta_margin),
        binary_delta,
        0
    )

    return cutoff_binary_delta


def wallet_delta(gold_paths, usdpln_paths, T_years, params, K, option_type='call', n_hedge=252):
    """
    Zakładamy że w momencie zero zaczynamy trading. Wykonanie akcji odbywa się w ostatnim kroku.
    (ostatnim dniu, bądź na zamknięcie giełdy ostatniego dnia)

    Parameters:
    -----------
    paths: ndarray
        Macierz ścieżek ceny aktywa kształtu (N, steps), gdzie N to liczba ścieżek,
        a steps to liczba punktów czasowych (włączając t=0)
    T_years: float
        Liczba lat do wygaśnięcia opcji
    sigma: float
        Zmienność roczna
    r: float
        Stopa procentowa wolna od ryzyka (ciągła, roczna)
    K: float
        Cena wykonania opcji (strike)
    option_type: str
        Typ opcji: 'call' lub 'put'
    n_hedge: int
        Liczba rehedgingów w ciągu roku.

    Returns:
    --------
    pnl: ndarray
        Wektor P&L w momencie wykonania opcji
    portfolio_composition: dict
        Słownik zawierający skład portfela w czasie:
        - 'gold': pozycja złota w indeksie
        - 'usd': pozycja usd w indeksie
        - 'cash': gotówka (instrument wolny od ryzyka)
        - 'option_value': teoretyczna wartośc opcji w czasie
        - 'portfolio_value': całkowita wartość portfela
        -'portfolio_delta': wartość delty portfela w czasie
    """

    N, steps = gold_paths.shape  # N — liczba ścieżek, steps — liczba punktów czasowych
    dt = T_years / steps  # Obliczanie kroku czasowego (dt)

    if n_hedge > steps: raise ValueError('Paths should be longer')

    hedge_frequency = steps // n_hedge + 1 if steps % n_hedge != 0 else steps / n_hedge
    hedge_frequency = max(1, hedge_frequency)
    hedge_indices = np.arange(0, steps, hedge_frequency)


    '''print(f'Liczba hedgy: {n_hedge}')
    print(f'steps: {steps}')
    print(f'hedge frequency: {hedge_frequency}')
    print(f'hedge_indices')
    print(hedge_indices)
    print('stop')'''

    # Inicjalizacja output
    gold_position = np.zeros((N, steps))
    usd_position = np.zeros((N, steps))
    usd_value = np.zeros((N, steps))
    cash_position = np.zeros((N, steps))
    option_value = np.zeros((N, steps))
    portfolio_value = np.zeros((N, steps))
    portfolio_delta = np.zeros((N, steps))

    # Inicjalizacja portfela
    gold_0 = gold_paths[:, 0]
    usdpln_0 = usdpln_paths[:, 0]
    t_to_maturity = T_years

    D = params['r'] - params['r_f'] + params['rho'] * params['gold']['std'] * params['usdpln']['std']
    option_value[:, 0] = option_price(gold_0,
                                      K,
                                      t_to_maturity, params['r'],
                                      params['gold']['std'],
                                      gold_0,
                                      'call',
                                      D=D)

    delta = option_delta(gold_0, usdpln_0, K, t_to_maturity, params['r'], params['gold']['std'], gold_0,
                         option_type, D)

    # Początkowy skład portfela
    gold_position[:, 0] = delta
    usd_position[:, 0] = -gold_0 * gold_position[:, 0]
    #usd_value[:, 0] = usd_position[:, 0] * usdpln_0
    cash_position[:, 0] = (option_value[:, 0] - gold_position[:, 0] * gold_0 * usdpln_0 - usd_position[:, 0] * usdpln_0)
    portfolio_value[:, 0] = -option_value[:, 0] + gold_position[:, 0] * gold_0 * usdpln_0 + usd_position[:, 0] * usdpln_0 + cash_position[:, 0]
    portfolio_delta[:, 0] = -delta + gold_position[:, 0] + usd_position[:, 0]

    for i in range(1, steps - 1):
        gold_current = gold_paths[:, i]  # Obecna cena złota
        usdpln_current = usdpln_paths[:, i]  # Obecny kurs USD/PLN
        t_to_maturity = T_years - i * dt  # Czas do wygaśnięcia

        option_value[:, i] = option_price(gold_current,
                                          K,
                                          t_to_maturity, params['r'],
                                          params['gold']['std'],
                                          gold_0,
                                          'call',
                                          D=D)  # Aktualizacja wartości opcji

        cash_position[:, i] = cash_position[:, i - 1] * np.exp(params['r'] * dt)  # Aktualizacja wartości gotówki (uwzględniamy oprocentowanie)
        #usd_value[:, i] = usd_value[:, i - 1] * np.exp(params['r_f'] * dt)  # Aktualizacja wartości dolara (oprocentowanie ze stopą zagraniczną)

        new_delta_1 = option_delta(gold_current,
                                   usdpln_current,
                                   K,
                                   t_to_maturity,
                                   params['r'],
                                   params['gold']['std'],
                                   gold_0, option_type, D)

        new_delta_2 = -gold_current * new_delta_1

        # Sprawdzenie, czy jest to punkt rehedgingu
        if i in hedge_indices:
            gold_position[:, i] = new_delta_1
            usd_position[:, i] = new_delta_2
            cash_position[:, i] -= ((gold_position[:, i] - (gold_position[:, i - 1])) * gold_current * usdpln_current +
                                    (usd_position[:, i] - usd_position[:, i-1] * np.exp(params['r_f'] * dt)) * usdpln_current)
            #usd_value[:, i] = usd_position[:, i] * usdpln_current
        else:
            gold_position[:, i] = gold_position[:, i - 1]
            usd_position[:, i] = usd_position[:, i - 1] * np.exp(params['r_f'] * dt)

        portfolio_value[:, i] = gold_position[:, i] * gold_current * usdpln_current + usd_position[:, i] * usdpln_current + cash_position[:, i]
        portfolio_delta[:, i] = -new_delta_1 + gold_position[:, i] + usd_position[:, i]

    # Moment wykonania opcji T = T_years
    gold_T = gold_paths[:, -1]
    usdpln_T = usdpln_paths[:, -1]

    option_value[:, -1] = option_price(gold_T,
                                       K,
                                       0, params['r'],
                                       params['gold']['std'],
                                       gold_0,
                                       'call',
                                       D=D)

    cash_position[:, -1] = cash_position[:, -2] * np.exp(params['r'] * dt)
    gold_position[:, -1] = gold_position[:, -2]
    usd_position[:, -1] = usd_position[:, -2] * np.exp(params['r_f'] * dt)
    #usd_value[:, -1] = usd_position[:, -1] * np.exp(params['r_f'] * dt) * usdpln_T
    portfolio_value[:, -1] = gold_position[:, -1] * gold_T * usdpln_T + usd_position[:, -1] * usdpln_T + cash_position[:, -1] - option_value[:, -1]
    portfolio_delta[:, -1] = 0

    portfolio_composition = {
        'gold': gold_position,
        'usdpln': usd_position,
        'cash': cash_position,
        'option_value': option_value,
        'portfolio_value': portfolio_value,
        'portfolio_delta': portfolio_delta
    }

    return portfolio_value[:, -1], portfolio_composition

