import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation import *
from visualization import *
from pricing import *
from data_loader import *
from hedging import *
import random


def run_part0():
    """Wczytanie danych"""
    print("Wczytanie danych i estymacja")
    pd.set_option('display.max_rows', None)

    ### Wczytanie danych

    xauusd_path = os.path.join('data', 'xauusd_d.csv')
    usdpln_path = os.path.join('data', 'usdpln_d.csv')
    start_date = '01-01-2010'
    end_date = '30-06-2023'
    data_gold, stats = load_data(xauusd_path, start_date=start_date, end_date=end_date)
    data_usdpln, stats = load_data(usdpln_path, start_date=start_date, end_date=end_date)


    data_gold['Date'] = pd.to_datetime(data_gold['Date'])
    data_usdpln['Date'] = pd.to_datetime(data_usdpln['Date'])

    data_Y = data_gold.copy()
    data_Y['Close'] = data_gold['Close'] * data_usdpln['Close']


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

    ### Wykresy volatility dla XAU/PLN
    data_with_volatility_goldpln = plot_historical_volatility(data_Y, n_years_list, plot=False, label='XAU_USD',
                                                           save=False)
    latest_data_vol_goldpln = data_with_volatility_goldpln.iloc[-1]

    ### Corr XAU/USD i USD/PLN
    data_with_corr = plot_historical_correlation(data_gold, data_usdpln, n_years_list, label1='XAU_USD',
                                                 label2='USD_PLN', save=False, plot=False)
    latest_data_corr = data_with_corr.iloc[-1]

    ### Corr XAU/PLN i USD/PLN
    data_with_corr_2 = plot_historical_correlation(data_Y, data_usdpln, n_years_list, label1='XAU_PLN',
                                                 label2='USD_PLN', save=False, plot=False)
    latest_data_corr_2 = data_with_corr_2.iloc[-1]

    '''print("\nNajnowsze wartości zmienności dla złota:")
    for n in n_years_list:
        if f'volatility_{n}y' in data_with_volatility_gold.columns:
            print(f"Zmienność w ujęciu rocznym zebrana z ostatnich {n}-lat: {latest_data_vol_gold[f'volatility_{n}y']:.2f}%")

    print("\nNajnowsze wartości zmienności dla dolara:")
    for n in n_years_list:
        if f'volatility_{n}y' in data_with_volatility_usdpln.columns:
            print(f"Zmienność w ujęciu rocznym zebrana z ostatnich {n}-lat: {latest_data_vol_usdpln[f'volatility_{n}y']:.2f}%")
    
    '''

    #print("\nNajnowsze wartości korelacji dla złoto / złotówka:")
    #for n in n_years_list:
    #    if f'correlation_{n}y' in data_with_corr.columns:
    #        print(f"Korelacja w ujęciu rocznym zebrana z ostatnich {n}-lat: {latest_data_corr_2[f'correlation_{n}y']:.4f}%")

    ### Wykres mean
    data_with_means_gold = plot_historical_mean_returns(data_gold, n_years_list, plot=False)
    data_with_means_usdpln = plot_historical_mean_returns(data_usdpln, n_years_list, plot=False)
    data_with_means_Y = plot_historical_mean_returns(data_Y, n_years_list, plot=False)
    latest_data_mean_gold = data_with_means_gold.iloc[-1]
    latest_data_mean_usdpln = data_with_means_usdpln.iloc[-1]
    latest_data_mean_Y = data_with_means_Y.iloc[-1]

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



    gold_0 = 1920.185  # data z 30-06-2023
    usdpln_0 = 4.06505  # data z 30-06-2023
    Y_0 = 7825.560353

    std_gold = latest_data_vol_gold[f'volatility_{3}y'] / 100
    std_usdpln = latest_data_vol_usdpln[f'volatility_{3}y'] / 100
    std_Y = latest_data_vol_goldpln[f'volatility_{3}y'] / 100

    mean_gold = latest_data_mean_gold[f'mean_return_{3}y'] / 100
    mean_usdpln = latest_data_mean_usdpln[f'mean_return_{3}y'] / 100
    mean_Y = latest_data_mean_Y[f'mean_return_{3}y'] / 100

    rho = latest_data_corr[f'correlation_{3}y']
    rho_2 = latest_data_corr_2[f'correlation_{3}y']
    r_f = 0.05
    r = 0.06

    mean_2_theoretic = mean_usdpln + mean_gold + rho * std_gold * std_usdpln
    std_2_theoretic = np.sqrt(std_gold ** 2 + std_usdpln ** 2 + 2 * rho * std_gold * std_usdpln)
    rho_2_theoretic = ((std_usdpln + rho * std_gold) / std_2_theoretic)

    #print(mean_2_theoretic)
    #print(mean_usdpln)

    ################################## Symulacja ścieżek
    N = 10000

    s0_vec = [gold_0, usdpln_0]
    mean_vec = [mean_gold, mean_usdpln]
    vol_vec = [std_gold, std_usdpln]
    corr_matrix = np.array([[1, rho],
                            [rho, 1]])

    paths, time = generate_correlated_gbm_paths(N=N,
                                                T=1,
                                                s0_vec=s0_vec,
                                                mean_vec=mean_vec,
                                                vol_vec=vol_vec,
                                                corr_matrix=corr_matrix,
                                                h=1)



    # Porównanie kwantyli z danymi historycznymi ( złoto )
    global data2
    data2, _ = load_data(xauusd_path, '30-06-2023', '30-06-2024')

    '''
    # Quantiles vs historical data
    plot_quantiles_vs_historical(
        paths_list=[paths['asset_0']],
        historical_data=data2,
        labels=['GBM XAU'],
        colors=['#FFA700'],
        plot_together=False,
        title="Kwantyle symulowanych ścieżek vs rzeczywista trajektoria"
    )'''

    # Porównanie kwantyli z danymi historycznymi ( złoto )
    global data2_usd
    data2_usd, _ = load_data(usdpln_path, '30-06-2023', '30-06-2024')

    '''
    # Quantiles vs historical data
    plot_quantiles_vs_historical(
        paths_list=[paths['asset_1']],
        historical_data=data2_usd,
        labels=['GBM USDPLN'],
        colors=['darkblue'],
        plot_together=False,
        title="Kwantyle symulowanych ścieżek vs rzeczywista trajektoria"
    )'''


def run_part1():
    """Inicjalizacja parametrów, symulacja ścieżek, walidacja"""
    print("Inicjalizacja parametrów, symulacja ścieżek, walidacja")

    ### Inicjalizacja parametrów dla symulacji ścieżek
    global params
    params = {'gold':
                  {'s0': 1920.185, 'std': 0.15048488481135047, 'mean': -0.0034530722372250137},
              'usdpln':
                  {'s0': 4.06505, 'std': 0.1157034074932167, 'mean': 0.027959336414910278},
              'Y':
                  {'s0': 7825.560353, 'std': 0.19426398582318347, 'mean': 0.01803850311842298},
              'Y_theoretic':
                  {'s0': 7825.560353, 'std': 0.1569226817319014, 'mean': 0.018802138667537688},
              'r_f': 0.05, 'r': 0.06, 'rho': -0.327604639460133, 'rho_y': -0.22315317327916478,
              'rho_y_theoretic': 0.42316292538158357}

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

    D = params['r'] - params['r_f'] - params['usdpln']['std']**2 + params['rho_y'] * params['Y']['std'] * params['usdpln']['std']
    sigma_y = np.sqrt(params['Y']['std']**2 + params['usdpln']['std']**2 - 2 * params['rho_y'] * params['Y']['std'] * params['usdpln']['std'])

    price0_2 = option_price(S=params['Y']['s0']/params['usdpln']['s0'],
                            K=1,
                            t=1,
                            r=params['r'],
                            sigma=sigma_y,
                            gold_0=params['gold']['s0'],
                            option_type='call',
                            D=D)

    D_theoretic = params['r'] - params['r_f'] - params['usdpln']['std']**2 + params['rho_y_theoretic'] * params['Y_theoretic']['std'] * params['usdpln']['std']
    sigma_y_theoretic = np.sqrt(params['Y_theoretic']['std']**2 + params['usdpln']['std']**2 - 2 * params['rho_y_theoretic'] * params['Y_theoretic']['std'] * params['usdpln']['std'])

    price0_2_theoretic = option_price(S=params['Y']['s0']/params['usdpln']['s0'],
                            K=1,
                            t=1,
                            r=params['r'],
                            sigma=sigma_y_theoretic,
                            gold_0=params['gold']['s0'],
                            option_type='call',
                            D=D_theoretic)



    print(f'Cena dla 1 modelu opcji w chwili 0: {price0}\n'
          f'Cena dla 2 modelu opcji w chwili 0: {price0_2}\n')

    ### Symulacja ścieżek
    global N, T, h
    N = 10000
    T = 1
    h = 1

    s0_vec = [params['gold']['s0'], params['usdpln']['s0']]
    mean_vec = [params['gold']['mean'], params['usdpln']['mean']]
    vol_vec = [params['gold']['std'], params['usdpln']['std']]
    corr_matrix = np.array([[1, params['rho']],
                            [params['rho'], 1]])

    random.seed(11)
    global paths
    paths, time = generate_correlated_gbm_paths(N=N,
                                                T=T,
                                                s0_vec=s0_vec,
                                                mean_vec=mean_vec,
                                                vol_vec=vol_vec,
                                                corr_matrix=corr_matrix,
                                                h=h)

    s0_vec2 = [params['Y']['s0'], params['usdpln']['s0']]
    mean_vec2 = [params['Y']['mean'], params['usdpln']['mean']]
    vol_vec2 = [params['Y']['std'], params['usdpln']['std']]
    corr_matrix2 = np.array([[1, params['rho_y']],
                            [params['rho_y'], 1]])

    global paths2
    paths2, time = generate_correlated_gbm_paths(N=N,
                                                T=T,
                                                s0_vec=s0_vec2,
                                                mean_vec=mean_vec2,
                                                vol_vec=vol_vec2,
                                                corr_matrix=corr_matrix2,
                                                h=h)

    #validate_paths_correlation(paths, corr_matrix)
    #test_gbm_properties(paths, time, mean_vec, vol_vec, s0_vec)


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



    #fig, stats = plot_pnl(pnl, title='Delta Hedge dla opcji quanto', save=False)
    #fig, stats = plot_qq_pnl(pnl, title="Delta Hedge dla opcji quanto", save=False)
    #perform_hedging_frequency_test(paths, params, K, option_type='call', save=False, savename='hedge_freq_test')
    #plot_portfolio_composition(portfolio_composition=portfolio, gold_paths=paths['asset_0'],
    #                           usdpln_paths=paths['asset_1'], path_indices=[0, 1], title='Skład portfela w czasie w modelu 1', save=True, minmax=True, pnl=pnl)
    #plt.show()


    # Przykład użycia:

    # Heatmapa ceny opcji względem zmienności złota i korelacji
    #fig = plot_option_heatmap('std_gold', 'rho', [0.05, 0.5], [-1, 1], params, 'call')
    #plt.savefig('heatmap_std_gold_rho.png')
    #plt.show()

    # Heatmapa ceny opcji względem stopy procentowej PLN i USD
    #fig = plot_option_heatmap('r', 'r_f', [0.01, 0.1], [0.01, 0.1], params, 'call')
    #plt.savefig('heatmap_r_r_f.png')
    #plt.show()

    # Heatmapa ceny opcji względem czasu do wygaśnięcia i ceny wykonania
    #fig = plot_option_heatmap('time', 'strike', [0.1, 1.0], [0.01, 2], params, 'call')
    #plt.savefig('heatmap_time_strike.png')
    #plt.show()

    # Heatmapa ceny opcji względem korelacji i ceny wykonania
    #fig = plot_option_heatmap('rho', 'strike', [-1, 1], [0.01, 2], params, 'call')
    #plt.savefig('heatmap_rho_strike.png')
    #plt.show()

    # Heatmapa ceny opcji względem czasu do wygaśnięcia i ceny wykonania
    #fig = plot_option_heatmap('std_gold', 'time', [0.05, 0.5], [0, 1], params, 'call')
    #plt.savefig('heatmap_std_gold_time.png')
    #plt.show()

    # Heatmapa ceny opcji względem czasu do wygaśnięcia i ceny wykonania
    #fig = plot_option_heatmap('std_gold', 'strike', [0.05, 0.5], [0.01, 2], params, 'call')
    #plt.savefig('heatmap_std_gold_strike.png')
    #plt.show()

    #plt.show()

    #### Drugi model

    pnl2, portfolio2 = wallet_delta2(gold_paths=paths2['asset_0'],
                                  usdpln_paths=paths2['asset_1'],
                                  T_years=T,
                                  params=params,
                                  K=K,
                                  option_type='call',
                                  n_hedge=n_hedge)


    #fig, stats = plot_pnl(pnl, title='Delta Hedge dla opcji quanto', save=False)
    #fig, stats = plot_qq_pnl(pnl, title="Delta Hedge dla opcji quanto", save=False)
    #perform_hedging_frequency_test(paths, params, K, option_type='call', save=False, savename='hedge_freq_test')
    #plot_portfolio_composition(portfolio_composition=portfolio, gold_paths=paths['asset_0'], usdpln_paths=paths['asset_1'], path_indices=[0,1,2])

    ### Porównanie 2 modelu
    # Hedge
    '''fig, stats = perform_dual_model_hedging_frequency_test(
        paths1=paths,
        paths2=paths2,
        params=params,
        K=1,
        option_type='call',
        save=True,
        savename='hedge_dual_model_comparison_1_2'
    )'''

    # STD
    '''fig, stats = perform_dual_model_gold_std_test(paths,
                                                  paths2,
                                                  params,
                                                  K=1,
                                                  option_type='call',
                                                  save=True,
                                                  savename='dual_gold_std_test_1_2')'''

    # Strike
    #fig, stats = perform_dual_model_K_test(paths, paths2, params,
    #                          option_type='call', save=True, savename='dual_K_test')


    # Rho
    #fig, test = perform_dual_model_rho_test(paths, paths2, params,
    #                            option_type='call', save=False, savename='dual_rho_test')

    #plt.show()

    # usdpln
    fig, test = perform_dual_model_usdpln_std_test(paths, paths2, params, K=1,
                                option_type='call', save=True, savename='dual_usdpln_test')

    plt.show()


    # Porównanie składu portfela dla rzeczywistości, modelu 1 i modelu 2

    '''real_gold = np.array([data2['Close'][:252]])
    real_usdpln = np.array([data2_usd['Close'][:252]])
    real_goldpln = real_gold * real_usdpln

    print(real_gold.shape)

    pnl3, portfolio3 = wallet_delta(gold_paths=real_gold,
                                     usdpln_paths=real_usdpln,
                                     T_years=T,
                                     params=params,
                                     K=K,
                                     option_type='call',
                                     n_hedge=n_hedge)

    pnl4, portfolio4 = wallet_delta2(gold_paths=real_goldpln,
                                    usdpln_paths=real_usdpln,
                                    T_years=T,
                                    params=params,
                                    K=K,
                                    option_type='call',
                                    n_hedge=n_hedge)


    # Dla modelu 1
    plot_portfolio_composition2(
        portfolio_composition_model=portfolio,
        portfolio_composition_real=portfolio3,
        real_paths=[real_gold, real_usdpln],
        gold_paths=paths['asset_0'],
        usdpln_paths=paths['asset_1'],
        model_number=1,
        save=True
    )

    # Dla modelu 2
    plot_portfolio_composition2(
        portfolio_composition_model=portfolio2,
        portfolio_composition_real=portfolio4,
        real_paths=[real_goldpln, real_usdpln],
        gold_paths=paths2['asset_0'],
        usdpln_paths=paths2['asset_1'],
        model_number=2,
        save=True
    )'''

if __name__ == "__main__":
    setup_plotting_style()

    #run_part0()
    run_part1()
    run_part2()
