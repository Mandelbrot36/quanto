import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats as scipy_stats
from hedging import wallet_delta
import os
from IPython.display import HTML
import gc
from simulation import *
import weakref

def setup_plotting_style():
    """Konfiguruje styl wykresów dla całego projektu."""
    plt.rc("figure",
           autolayout=True,
           figsize=(10, 6),
           titlesize=13,
           titleweight='bold')
    plt.rc("axes",
           labelweight="bold",
           labelsize="large",
           titleweight="bold",
           titlesize=16,
           titlepad=10,
           titlelocation='center')

    sns.set_theme(style='whitegrid', context='paper', palette='dark')

setup_plotting_style()

# Globalna lista do przechowywania referencji do animacji
_animations = []


def create_animated_histogram(paths, params, param_to_animate, param_range, option_type='call',
                              base_n_hedge=252, base_K=1, fps=3, save=False, filename='pnl_animation.gif'):
    """
    Tworzy animowany histogram rozkładu P&L w zależności od wybranego parametru.
    Animacja zatrzymuje się na ostatniej klatce poprzez dodanie dodatkowej klatki końcowej.

    Parameters:
    -----------
    paths : dict
        Słownik ze ścieżkami cenowymi aktywów (taki jak zwracany przez generate_correlated_gbm_paths)
    params : dict
        Słownik z parametrami symulacji (sigma, r, itp.)
    param_to_animate : str
        Parametr do animacji, jeden z: 'n_hedge', 'K', 'gold_std', 'usdpln_std', 'rho'
    param_range : list
        Lista wartości parametru do animacji
    option_type : str, optional
        Typ opcji ('call' lub 'put'), domyślnie 'call'
    base_n_hedge : int, optional
        Bazowa liczba rehedgingów (używana gdy animowany jest inny parametr), domyślnie 252
    base_K : float, optional
        Bazowa cena wykonania (używana gdy animowany jest inny parametr), domyślnie 1
    fps : int, optional
        Liczba klatek na sekundę w animacji, domyślnie 2
    save : bool, optional
        Czy zapisać animację do pliku, domyślnie False
    filename : str, optional
        Nazwa pliku do zapisu animacji, domyślnie 'pnl_animation.gif'

    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        Obiekt animacji
    stats_df : pandas.DataFrame
        Ramka danych ze statystykami dla każdej klatki animacji
    """
    import pandas as pd

    # Sprawdzenie poprawności parametru do animacji
    valid_params = ['n_hedge', 'K', 'gold_std', 'usdpln_std', 'rho', 'r_f']
    if param_to_animate not in valid_params:
        raise ValueError(f"Parametr do animacji musi być jednym z: {valid_params}")

    # Inicjalizacja statystyk
    stats_data = []

    # Przygotowanie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dodajemy dodatkową klatkę na końcu, która będzie identyczna z ostatnią klatką
    # To sprawi, że animacja będzie zatrzymywać się na ostatniej klatce
    extended_param_range = param_range + [param_range[-1]]

    # Zmienna do śledzenia, czy jesteśmy na ostatniej klatce
    is_last_frame = [False]

    # Funkcja aktualizująca histogram dla każdej klatki
    def update_histogram(frame_idx):
        ax.clear()

        # Sprawdzenie, czy to ostatnia klatka (dodatkowa)
        if frame_idx == len(param_range):
            # Jeśli to ostatnia klatka (dodatkowa), użyj wartości z poprzedniej klatki
            frame_idx = len(param_range) - 1
            is_last_frame[0] = True

        # Aktualizacja parametrów w zależności od wybranego parametru do animacji
        current_params = params.copy()
        current_n_hedge = base_n_hedge
        current_K = base_K

        param_value = param_range[frame_idx]

        if param_to_animate == 'n_hedge':
            current_n_hedge = param_value
            param_label = f"Liczba rehedgingów: {param_value}"
        elif param_to_animate == 'K':
            current_K = param_value
            param_label = f"Cena wykonania (K): {param_value:.2f}"
        elif param_to_animate == 'gold_std':
            current_params['gold']['std'] = param_value
            param_label = f"Założone: {(param_value / 0.15048488481135047) * 100:.2f}% zrealizowanego volatility - Złoto"
        elif param_to_animate == 'usdpln_std':
            current_params['usdpln']['std'] = param_value
            param_label = f"Założone: {(param_value / 0.1157034074932167) * 100:.2f}% zrealizowanego volatility - USDPLN"
        elif param_to_animate == 'rho':
            current_params['rho'] = param_value
            param_label = f"Założona korelacja: {param_value: .2f}, zrealizowana: {-0.327604639460133: .2f}"
        elif param_to_animate == 'r_f':
            current_params['r_f'] = param_value
            param_label = f"Stopa zagraniczna (r_f): {param_value:.2f}"

        # Obliczenie P&L dla bieżących parametrów
        if not is_last_frame[0]:  # Obliczamy tylko jeśli to nie jest dodatkowa klatka
            print(f'klatka {frame_idx}')
            pnl, portfolio_composition = wallet_delta(
                gold_paths=paths['asset_0'],
                usdpln_paths=paths['asset_1'],
                T_years=1,  # Zakładamy T=1 rok
                params=current_params,
                K=current_K,
                option_type=option_type,
                n_hedge=current_n_hedge
            )

            # Przygotowanie danych do histogramu
            final_pnl = pnl

            # Obliczenie statystyk
            mask_nan = np.isnan(final_pnl)
            mu = np.nanmean(final_pnl)
            sigma = np.nanstd(final_pnl)
            median = np.nanmedian(final_pnl)
            skew = scipy_stats.skew(final_pnl[~mask_nan])
            kurtosis = scipy_stats.kurtosis(final_pnl[~mask_nan])
            positive_paths_pct = 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan])

            # Zapisanie statystyk
            stats_data.append({
                'Parametr': param_to_animate,
                'Wartość': param_value,
                'Średnia': mu,
                'Mediana': median,
                'Odchylenie std': sigma,
                'Skośność': skew,
                'Kurtoza': kurtosis,
                'Procent ścieżek z zyskiem': positive_paths_pct
            })

            # Zapisanie danych dla ostatniej klatki
            if frame_idx == len(param_range) - 1:
                update_histogram.last_frame_data = {
                    'final_pnl': final_pnl,
                    'mu': mu,
                    'sigma': sigma,
                    'median': median,
                    'skew': skew,
                    'kurtosis': kurtosis,
                    'positive_paths_pct': positive_paths_pct
                }
        else:
            # Użyj zapisanych danych dla ostatniej klatki
            final_pnl = update_histogram.last_frame_data['final_pnl']
            mu = update_histogram.last_frame_data['mu']
            sigma = update_histogram.last_frame_data['sigma']
            median = update_histogram.last_frame_data['median']
            skew = update_histogram.last_frame_data['skew']
            kurtosis = update_histogram.last_frame_data['kurtosis']
            positive_paths_pct = update_histogram.last_frame_data['positive_paths_pct']

        # Rysowanie histogramu
        sns.histplot(final_pnl, kde=False, ax=ax, stat='percent', color='blue')

        # Dodanie linii dla średniej i zera
        ax.axvline(mu, color='r', linestyle='--', label=f"Średnia: {mu:.2f}")
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, label="P&L = 0")

        # Dodanie statystyk do wykresu
        stats_text = (
            f"Średnia: {mu:.2f}\n"
            f"Mediana: {median:.2f}\n"
            f"Odchylenie std: {sigma:.2f}\n"
            f"Procent z zyskiem: {positive_paths_pct:.1f}%\n"
            f"Skośność: {skew:.2f}\n"
            f"Kurtoza: {kurtosis:.2f}\n"
        )
        ax.text(0.01, 0.75, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Formatowanie wykresu
        ax.set_xlabel('Końcowy P&L', loc='center')
        ax.set_ylabel('Procent ścieżek [%]')
        ax.set_title(f"Rozkład P&L - {param_label}", fontsize=14)
        ax.legend()
        plt.xlim(-2.5, 2.5)
        plt.ylim(0, 6)

        return ax,

    # Inicjalizacja atrybutu dla przechowywania danych ostatniej klatki
    update_histogram.last_frame_data = {}

    # Tworzenie animacji z dodatkową klatką na końcu
    anim = animation.FuncAnimation(
        fig, update_histogram, frames=len(extended_param_range), interval=1000/fps, blit=False
    )

    # Zapisanie animacji do pliku
    if save:
        try:
            anim.save(filename, writer='pillow', fps=fps)
            print(f"Animacja zapisana do pliku: {filename}")
        except Exception as e:
            print(f"Błąd przy zapisywaniu animacji: {e}")
            print("Próba zapisania z domyślnymi ustawieniami...")
            anim.save(filename)
            print(f"Animacja zapisana do pliku: {filename} (domyślne ustawienia)")

    # Konwersja statystyk do DataFrame
    stats_df = pd.DataFrame(stats_data)

    # Dodanie animacji do globalnej listy, aby zapobiec usunięciu przez garbage collector
    global _animations
    _animations.append(anim)

    return anim, stats_df


def display_animation(anim):
    """
    Wyświetla animację w notebooku Jupyter.

    Parameters:
    -----------
    anim : matplotlib.animation.FuncAnimation
        Obiekt animacji

    Returns:
    --------
    HTML : IPython.display.HTML
        Obiekt HTML do wyświetlenia w notebooku
    """
    return HTML(anim.to_jshtml())


def clear_animations():
    """
    Czyści wszystkie zapisane animacje i zwalnia pamięć.
    """
    global _animations
    _animations.clear()
    plt.close('all')
    gc.collect()
    print("Wszystkie animacje zostały wyczyszczone z pamięci.")

# Przykład użycia:

# Przykładowe parametry
params = {'gold':
              {'s0': 1920.185, 'std': 0.15048488481135047, 'mean': -0.0034530722372250137},
          'usdpln':
              {'s0': 4.06505, 'std': 0.1157034074932167, 'mean': 0.027959336414910278},
          'r_f': 0.05, 'r': 0.06, 'rho': -0.327604639460133}

# Generowanie ścieżek
N = 10000
T = 1
h = 1
s0_vec = [params['gold']['s0'], params['usdpln']['s0']]
mean_vec = [params['gold']['mean'], params['usdpln']['mean']]
vol_vec = [params['gold']['std'], params['usdpln']['std']]
corr_matrix = np.array([[1, params['rho']], [params['rho'], 1]])

paths, time = generate_correlated_gbm_paths(
    N=N, T=T, s0_vec=s0_vec, mean_vec=mean_vec,
    vol_vec=vol_vec, corr_matrix=corr_matrix, h=h
)

# Animacja dla różnych liczb rehedgingów
n_hedge_range = list(range(9, 126, 5)) + [252, 504, 1008]
K_range = np.arange(0.05, 2.05, 0.05)
gold_std_range = np.arange(0.5, 1.5, 0.01) * params['gold']['std']
usd_pln_range = np.arange(0.5, 1.5, 0.01) * params['usdpln']['std']
rho_range = np.arange(-1, 1, 0.05)
r_f_range = np.arange(0.01, 0.11, 0.005)
anim, stats = create_animated_histogram(
    paths=paths, params=params, param_to_animate='K',
    param_range=K_range, save=True, filename='K_animation.gif'
)

# Wyświetlenie statystyk
print(stats)

# Po zakończeniu pracy z animacjami, można zwolnić pamięć:
clear_animations()

