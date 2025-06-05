import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation import *
from visualization import *
from pricing import *
from data_loader import *
from hedging import *


def run_part0():
    """Wczytanie danych"""
    print("Wczytanie danych i estymacja")

    ### Wczytanie danych
    xauusd_path = os.path.join('data', 'xauusd_d.csv')
    usdpln_path = os.path.join('data', 'usdpln_d.csv')
    start_date = '01-01-2010'
    end_date = '30-06-2023'
    data_gold, stats = load_data(xauusd_path, start_date=start_date, end_date=end_date)
    data_usdpln, stats = load_data(usdpln_path, start_date=start_date, end_date=end_date)

    ### Wykresy danych historycznych
    #plot_price_history(data_gold, label='XAU_USD', save=True)
    #plot_price_history(data_usdpln, label='USD_PLN', save=True)
    #plot_price_history_dual(data_gold, data_usdpln, title='Porównanie trajektorii cen historycznych XAU/USD i USD/PLN',
    # label1='XAU_USD', label2='USD_PLN', normalize=True, save=False)

    n_years_list = [1, 2, 3, 4]
    ### Wykresy volatility dla XAU/USD
    data_with_volatility_gold = plot_historical_volatility(data_gold, n_years_list, plot=False, label='XAU_USD',
                                                           save=False)
    latest_data_vol_gold = data_with_volatility_gold.iloc[-1]

    ### Wykresy volatility dla USD/PLN
    data_with_volatility_usdpln = plot_historical_volatility(data_usdpln, n_years_list, plot=False, label='USD_PLN',
                                                             save=False)
    latest_data_vol_usdpln = data_with_volatility_usdpln.iloc[-1]

    data_with_corr = plot_historical_correlation(data_gold, data_usdpln, n_years_list, label1='XAU_USD',
                                                 label2='USD_PLN', save=False, plot=False)
    latest_data_corr = data_with_corr.iloc[-1]

    '''print("\nNajnowsze wartości zmienności dla złota:")
    for n in n_years_list:
        if f'volatility_{n}y' in data_with_volatility_gold.columns:
            print(f"Zmienność w ujęciu rocznym zebrana z ostatnich {n}-lat: {latest_data_vol_gold[f'volatility_{n}y']:.2f}%")

    print("\nNajnowsze wartości zmienności dla dolara:")
    for n in n_years_list:
        if f'volatility_{n}y' in data_with_volatility_usdpln.columns:
            print(f"Zmienność w ujęciu rocznym zebrana z ostatnich {n}-lat: {latest_data_vol_usdpln[f'volatility_{n}y']:.2f}%")

    print("\nNajnowsze wartości korelacji dla złoto / dolar:")
    for n in n_years_list:
        if f'correlation_{n}y' in data_with_corr.columns:
            print(f"Korelacja w ujęciu rocznym zebrana z ostatnich {n}-lat: {latest_data_corr[f'correlation_{n}y']:.4f}%")'''

    ### Wykres mean
    data_with_means_gold = plot_historical_mean_returns(data_gold, n_years_list, plot=False)
    data_with_means_usdpln = plot_historical_mean_returns(data_usdpln, n_years_list, plot=False)
    latest_data_mean_gold = data_with_means_gold.iloc[-1]
    latest_data_mean_usdpln = data_with_means_usdpln.iloc[-1]

    '''# Print ostatnich wartości
    print("\nNajnowsze wartości średnich zwrotów dla złota:")
    for n in n_years_list:
        if f'mean_return_{n}y' in data_with_means_gold.columns:
            print(f"Średni {n}-letni zwrot: {latest_data_mean_gold[f'mean_return_{n}y']:.2f}%")
    print('\n')

    print("\nNajnowsze wartości średnich zwrotów dla USD/PLN:")
    for n in n_years_list:
        if f'mean_return_{n}y' in data_with_means_usdpln.columns:
            print(f"Średni {n}-letni zwrot: {latest_data_mean_usdpln[f'mean_return_{n}y']:.2f}%")
    print('\n')'''

    '''################################## Symulacja ścieżek
    N = 10000
    T = 1 # wykres będzie miał t = 252 (wliczając s0)
    s0 = data1.iloc[-1, 4]  # data z 30-12-2021
    std = latest_data_vol[f'volatility_{2}y'] / 100 # Ustawiam zmienność roczną zebraną z ostatnich 2 lat
    mean = latest_data_mean[f'mean_return_{2}y'] / 100 # i średnią

    paths_gbm, time_gbm = generate_gbm_paths(
        N=N,
        T=T,
        s0=s0,
        mean=mean,
        volatility=std,
        h=1,
        plot=False
    )


    # Porównanie kwantyli z danymi historycznymi
    data2, _ = load_data(data_path, '30-12-2021', '01-01-2023')
'''

    '''# Quantiles vs historical data
    plot_quantiles_vs_historical(
        paths_list=[paths_gbm, paths_bootstrap],
        historical_data=data2,
        labels=['GBM', 'Bootstrap'],
        colors=['blue', 'green'],
        plot_together=True,
        title="Kwantyle symulowanych ścieżek vs rzeczywista trajektoria"
    )'''

    gold_0 = 1920.185  # data z 01-06-2023
    usdpln_0 = 4.06505  # data z 01-06-2023

    std_gold = latest_data_vol_gold[f'volatility_{3}y'] / 100
    std_usdpln = latest_data_vol_usdpln[f'volatility_{3}y'] / 100

    mean_gold = latest_data_mean_gold[f'mean_return_{3}y'] / 100
    mean_usdpln = latest_data_mean_usdpln[f'mean_return_{3}y'] / 100

    rho = latest_data_corr[f'correlation_{3}y']
    r_f = 0.05
    r = 0.06


def run_part1():
    """Inicjalizacja parametrów, symulacja ścieżek, walidacja"""
    print("Inicjalizacja parametrów, symulacja ścieżek, walidacja")

    ### Inicjalizacja parametrów dla symulacji ścieżek
    global params
    params = {'gold':
                  {'s0': 1920.185, 'std': 0.15048488481135047, 'mean': -0.0034530722372250137},
              'usdpln':
                  {'s0': 4.06505, 'std': 0.1157034074932167, 'mean': 0.027959336414910278},
              'r_f': 0.05, 'r': 0.06, 'rho': -0.327604639460133}

    ### Cena opcji w t=0
    dividend_rate = params['r'] - params['r_f'] + params['rho'] * params['gold']['std'] * params['usdpln']['std']
    price0 = option_price(S=params['gold']['s0'],
                          K=1,
                          t=1,
                          r=params['r'],
                          sigma=params['gold']['std'],
                          gold_0=params['gold']['s0'],
                          option_type='call',
                          D=dividend_rate)

    print(price0)

    ### Symulacja ścieżek
    global N, T, h
    N = 1000
    T = 1
    h = 1

    s0_vec = [params['gold']['s0'], params['usdpln']['s0']]
    mean_vec = [params['gold']['mean'], params['usdpln']['mean']]
    vol_vec = [params['gold']['std'], params['usdpln']['std']]
    corr_matrix = np.array([[1, params['rho']],
                            [params['rho'], 1]])

    global paths
    paths, time = generate_correlated_gbm_paths(N=N,
                                                T=T,
                                                s0_vec=s0_vec,
                                                mean_vec=mean_vec,
                                                vol_vec=vol_vec,
                                                corr_matrix=corr_matrix,
                                                h=h)
    #validate_paths_correlation(paths, corr_matrix)


def run_part2():
    """Delta Hedge i analiza"""
    print("Delta Hedge i analiza")

    K = 1
    n_hedge = 252

    pnl, portfolio = wallet_delta(gold_paths=paths['asset_0'],
                                  usdpln_paths=paths['asset_1'],
                                  T_years=T,
                                  params=params,
                                  K=K,
                                  option_type='call',
                                  n_hedge=n_hedge)

    fig, stats = plot_pnl(pnl, title='Delta Hedge dla opcji quanto', save=False)
    #fig, stats = plot_qq_pnl(pnl, title="Delta Hedge dla opcji quanto", save=False)
    #perform_hedging_frequency_test(paths, params, K, option_type='call', save=False, savename='hedge_freq_test')

    plot_portfolio_composition(portfolio_composition=portfolio, gold_paths=paths['asset_0'], usdpln_paths=paths['asset_1'], path_indices=[0,1,2])
    plt.show()


if __name__ == "__main__":
    setup_plotting_style()

    #run_part0()
    run_part1()
    run_part2()
