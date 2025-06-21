import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats as scipy_stats

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


def perform_dual_model_hedging_frequency_test(paths, params, K,
                                            option_type='call', save=False, savename='dual_hedge_freq_test'):
    """
    Zmodyfikowana wersja perform_hedging_frequency_test dla porównania dwóch modeli opcji quanto.
    
    Parameters:
    -----------
    paths: dict
        Ścieżki cenowe aktywów
    params: dict  
        Parametry dla Modelu 1
    K: float
        Cena wykonania opcji
    option_type: str
        Typ opcji ('call' lub 'put')
    wallet_delta_func: function
        Funkcja wallet_delta do symulacji portfela
    save: bool
        Czy zapisać wykres
    savename: str
        Nazwa pliku do zapisu

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej częstotliwości i modelu
    """

    # Konfiguracja testów - można dostosować
    frequencies = {
        'Co 2 tygodnie': 26,
        'Tygodniowo': 52,
        'Co 3 dni': 252 // 3,
    }

    # Parametry dla Modelu 2 (opcja quanto na Y)
    params_model2 = params.copy()
    # Tutaj można dostosować parametry dla Modelu 2 jeśli potrzeba
    
    # Dynamiczne określenie układu wykresów
    n_freq = len(frequencies)
    if n_freq == 2:
        nrows, ncols = 2, 2
        figsize = (14, 10)
    elif n_freq == 3:
        nrows, ncols = 3, 2
        figsize = (14, 14)
    else:
        nrows = n_freq
        ncols = 2
        figsize = (14, 4 * n_freq)

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle('Porównanie rozkładów P&L dla dwóch modeli opcji quanto', y=0.98, fontsize=16)

    # Upewnienie się, że axes jest zawsze 2D
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (freq_name, n_hedge) in enumerate(frequencies.items()):

        # Model 1: Symulacja z oryginalną funkcją wallet_delta
        pnl_model1, _ = wallet_delta(
            gold_paths=paths['asset_0'],
            usdpln_paths=paths['asset_1'],
            T_years=1,
            params=params,
            K=K,
            option_type=option_type,
            n_hedge=n_hedge
        )
        final_pnl_model1 = pnl_model1 if pnl_model1.ndim == 1 else pnl_model1[:, -1]

        # Model 2: Symulacja z parametrami dla drugiego modelu
        # Tutaj można użyć tej samej funkcji wallet_delta z innymi parametrami
        # lub wywołać inną wersję funkcji dla Modelu 2
        pnl_model2, _ = wallet_delta_func(
            gold_paths=paths['asset_0'],
            usdpln_paths=paths['asset_1'],
            T_years=1,
            params=params_model2,  # Można zmodyfikować parametry dla Modelu 2
            K=K,
            option_type=option_type,
            n_hedge=n_hedge
        )
        final_pnl_model2 = pnl_model2 if pnl_model2.ndim == 1 else pnl_model2[:, -1]

        # Filtrowanie NaN
        mask_nan1 = np.isnan(final_pnl_model1)
        mask_nan2 = np.isnan(final_pnl_model2)
        
        # Statystyki dla Modelu 1
        stats_model1 = {
            'Model': 'Model 1',
            'Częstotliwość': freq_name,
            'Liczba rehedge': n_hedge,
            'Średnia': np.nanmean(final_pnl_model1),
            'Mediana': np.nanmedian(final_pnl_model1),
            'Std': np.nanstd(final_pnl_model1),
            'Skewness': scipy_stats.skew(final_pnl_model1[~mask_nan1]) if np.sum(~mask_nan1) > 3 else 0,
            'Kurtosis': scipy_stats.kurtosis(final_pnl_model1[~mask_nan1]) if np.sum(~mask_nan1) > 3 else 0,
            'Min': np.nanmin(final_pnl_model1),
            'Max': np.nanmax(final_pnl_model1),
            'P(PL>0)': np.mean(final_pnl_model1 > 0) * 100
        }
        results.append(stats_model1)

        # Statystyki dla Modelu 2
        stats_model2 = {
            'Model': 'Model 2',
            'Częstotliwość': freq_name,
            'Liczba rehedge': n_hedge,
            'Średnia': np.nanmean(final_pnl_model2),
            'Mediana': np.nanmedian(final_pnl_model2),
            'Std': np.nanstd(final_pnl_model2),
            'Skewness': scipy_stats.skew(final_pnl_model2[~mask_nan2]) if np.sum(~mask_nan2) > 3 else 0,
            'Kurtosis': scipy_stats.kurtosis(final_pnl_model2[~mask_nan2]) if np.sum(~mask_nan2) > 3 else 0,
            'Min': np.nanmin(final_pnl_model2),
            'Max': np.nanmax(final_pnl_model2),
            'P(PL>0)': np.mean(final_pnl_model2 > 0) * 100
        }
        results.append(stats_model2)

        # Wykres histogramu dla Modelu 1 (pierwsza kolumna)
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl_model1, kde=False, ax=ax1, stat='percent', color='blue', alpha=0.7)
        ax1.set_title(f'Model 1: {freq_name} (n={n_hedge})', fontsize=12)
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax1.axvline(np.nanmean(final_pnl_model1), color='blue', linestyle='-', alpha=0.8)

        # Statystyki na wykresie dla Modelu 1
        stats_text1 = (f"Średnia: {stats_model1['Średnia']:.2f}\n"
                       f"Mediana: {stats_model1['Mediana']:.2f}\n"
                       f"Std: {stats_model1['Std']:.2f}\n"
                       f"Skośność: {stats_model1['Skewness']:.2f}\n"
                       f"Kurtoza: {stats_model1['Kurtosis']:.2f}")
        ax1.text(0.05, 0.95, stats_text1, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='lightblue', alpha=0.8))

        # Wykres histogramu dla Modelu 2 (druga kolumna)
        ax2 = axes[idx, 1]
        sns.histplot(final_pnl_model2, kde=True, ax=ax2, stat='percent', color='red', alpha=0.7)
        ax2.set_title(f'Model 2: {freq_name} (n={n_hedge})', fontsize=12)
        ax2.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax2.axvline(np.nanmean(final_pnl_model2), color='red', linestyle='-', alpha=0.8)

        # Statystyki na wykresie dla Modelu 2
        stats_text2 = (f"Średnia: {stats_model2['Średnia']:.2f}\n"
                       f"Mediana: {stats_model2['Mediana']:.2f}\n"
                       f"Std: {stats_model2['Std']:.2f}\n"
                       f"Skośność: {stats_model2['Skewness']:.2f}\n"
                       f"Kurtoza: {stats_model2['Kurtosis']:.2f}")
        ax2.text(0.05, 0.95, stats_text2, transform=ax2.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='lightcoral', alpha=0.8))

        # Etykiety osi
        if idx == nrows - 1:  # Ostatni wiersz
            ax1.set_xlabel('Końcowy P/L Model 1', fontsize=9)
            ax2.set_xlabel('Końcowy P/L Model 2', fontsize=9)
        else:
            ax1.set_xlabel('')
            ax2.set_xlabel('')

        ax1.set_ylabel('Procent ścieżek [%]', fontsize=9)
        ax2.set_ylabel('Procent ścieżek [%]', fontsize=9)

    plt.tight_layout()

    if save: 
        plt.savefig(f'{savename}.png', format='png', bbox_inches='tight', dpi=150)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[[
        'Model', 'Częstotliwość', 'Liczba rehedge', 'Średnia', 'Mediana', 'Std',
        'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)'
    ]]

    return fig, stats_df

