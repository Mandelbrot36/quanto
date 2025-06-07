import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib.dates import DateFormatter, MonthLocator
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.ticker import FuncFormatter
from scipy import stats as scipy_stats
from hedging import wallet_delta
from simulation import generate_gbm_paths



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


def plot_price_history(data, title=None, highlight_dates=None, save=False, label='Default'):
    """
    Wykreśla historyczne ceny z opcjonalnym zaznaczeniem ważnych dat.

    Parameters:
    -----------
    data : DataFrame
        Dane cenowe z kolumną 'Date' i 'Close'.
    title : str, optional
        Tytuł wykresu.
    highlight_dates : dict, optional
        Słownik z datami do zaznaczenia i ich opisami.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Konwersja dat
    data_dt = pd.to_datetime(data.Date)

    # Wykres cen
    ax.plot(data_dt, data.Close, label="WIG20")

    # Formatowanie osi X
    #ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    #ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    # Zaznaczenie ważnych dat
    if highlight_dates:
        for date_str, desc in highlight_dates.items():
            date = pd.to_datetime(date_str)
            ax.axvline(date, color='red', linestyle='dotted', linewidth=1.5)
            ax.text(date + pd.Timedelta(days=10), data.Close.min(), desc,
                    color="red", rotation=90, verticalalignment="bottom", fontsize=10)

    # Formatowanie
    plt.xticks(rotation=45)
    #ax.set_xlabel("Data")
    ax.set_ylabel(label)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{label} {data.Date.iloc[0].strftime('%Y-%m-%d')} do {data.Date.iloc[-1].strftime('%Y-%m-%d')}")

    #plt.legend()
    plt.tight_layout()

    if save: plt.savefig(f'historic_{label}.png', format='png', bbox_inches='tight', dpi=300)

    plt.show()


def plot_price_history_dual(data1, data2=None, title=None, highlight_dates=None, save=False, label1='Asset1', label2='Asset2', normalize=False, plot=True):
    """
    Wykreśla historyczne ceny jednej lub dwóch serii danych z opcjonalnym zaznaczeniem ważnych dat.

    Parameters:
    -----------
    data1 : DataFrame
        Pierwsze dane cenowe z kolumną 'Date' i 'Close'.
    data2 : DataFrame, optional
        Drugie dane cenowe z kolumną 'Date' i 'Close'.
    title : str, optional
        Tytuł wykresu.
    highlight_dates : dict, optional
        Słownik z datami do zaznaczenia i ich opisami.
    save : bool, optional
        Czy zapisać wykres do pliku.
    label1 : str, optional
        Etykieta dla pierwszej serii danych.
    label2 : str, optional
        Etykieta dla drugiej serii danych.
    normalize : bool, optional
        Czy normalizować dane do wartości początkowej (100).
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.dates import MonthLocator, DateFormatter
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))

    # Konwersja dat dla pierwszej serii
    data1 = data1.copy()
    data1['Date'] = pd.to_datetime(data1['Date'])

    # Normalizacja danych jeśli wymagana
    if normalize and len(data1) > 0:
        base_value1 = data1['Close'].iloc[0]
        data1['Close_normalized'] = data1['Close'] / base_value1 * 100
        y_data1 = data1['Close_normalized']
        y_label = "Znormalizowana wartość (100 = początek)"
    else:
        y_data1 = data1['Close']
        y_label = "Wartość"

    # Wykres pierwszej serii danych
    ax.plot(data1['Date'], y_data1, label=label1)

    # Jeśli podano drugą serię danych
    if data2 is not None:
        data2 = data2.copy()
        data2['Date'] = pd.to_datetime(data2['Date'])

        # Normalizacja drugiej serii jeśli wymagana
        if normalize and len(data2) > 0:
            base_value2 = data2['Close'].iloc[0]
            data2['Close_normalized'] = data2['Close'] / base_value2 * 100
            y_data2 = data2['Close_normalized']
        else:
            y_data2 = data2['Close']

        # Wykres drugiej serii danych
        ax.plot(data2['Date'], y_data2, label=label2)

    # Formatowanie osi X
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[6]))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    # Zaznaczenie ważnych dat
    if highlight_dates:
        # Określenie zakresu Y dla linii i tekstu
        y_min = min(data1['Close'].min(), data2['Close'].min() if data2 is not None else float('inf'))
        y_max = max(data1['Close'].max(), data2['Close'].max() if data2 is not None else float('-inf'))

        for date_str, desc in highlight_dates.items():
            date = pd.to_datetime(date_str)
            ax.axvline(date, color='red', linestyle='dotted', linewidth=1.5)
            ax.text(date + pd.Timedelta(days=10), y_min, desc,
                    color="red", rotation=90, verticalalignment="bottom", fontsize=10)

    # Formatowanie
    plt.xticks(rotation=45)
    ax.set_ylabel(y_label)

    # Dodanie legendy jeśli są dwie serie danych
    if data2 is not None:
        plt.legend()

    # Tytuł wykresu
    if title:
        ax.set_title(title)
    else:
        start_date = min(data1['Date'].iloc[0], data2['Date'].iloc[0] if data2 is not None else data1['Date'].iloc[0])
        end_date = max(data1['Date'].iloc[-1], data2['Date'].iloc[-1] if data2 is not None else data1['Date'].iloc[-1])
        ax.set_title(f"Ceny {start_date.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}")

    plt.tight_layout()

    # Zapisanie wykresu jeśli wymagane
    if save:
        if data2 is not None:
            plt.savefig(f'historic_{label1}_{label2}.png', format='png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'historic_{label1}.png', format='png', bbox_inches='tight', dpi=300)

    if plot is True:
        plt.show()


    return fig, ax



def compare_returns_distribution(returns, returns2, save=False):
    """
    Porównuje rozkłady zwrotów z rozkładem normalnym.

    Parameters:
    -----------
    returns : Series
        Zwroty 1.
    log_returns : Series
        Zwroty 2.
    """
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    fig.suptitle('Porównanie zwrotów dziennych \n '
                 'dla okresu 01.01.2022 - 31.12.2022')

    # Scatterploty
    sns.scatterplot(x=range(len(returns)), y=returns, ax=ax[0, 0], color='red', label="Zwroty rzeczywiste")
    sns.scatterplot(x=range(len(returns2)), y=returns2, ax=ax[1, 0], color='blue', label="Zwroty symulowane")

    # Porównanie z rozkładem normalnym
    x = np.linspace(-4, 4, 100)
    ax[0, 1].plot(x, norm.pdf(x, 0, 1), 'r', label="N(0,1) PDF")
    ax[1, 1].plot(x, norm.pdf(x, 0, 1), 'r', label="N(0,1) PDF")

    # Znormalizowane histogramy
    sns.histplot((returns - returns.mean()) / returns.std(), kde=True, color='red', ax=ax[0, 1], label="Zwroty rzeczywiste")
    sns.histplot((returns2 - returns2.mean()) / returns2.std(), kde=True, color='blue', ax=ax[1, 1],
                 label="Zwroty symulowane")

    # Tytuły i legendy
    ax[0, 0].set_title("Zwroty rzeczywiste")
    ax[1, 0].set_title("Zwroty symulowane za pomocą metody Bootstrap")
    ax[0, 1].set_title("Histogram zwrotów vs N(0,1)")
    ax[1, 1].set_title("Histogram zwrotów vs N(0,1)")

    for i in range(2):
        for j in range(2):
            ax[i, j].legend()
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')

    ax[0, 1].set_ylim(0, 50)
    ax[1, 1].set_ylim(0, 50)
    ax[0, 1].set_xlim(-6, 4.5)
    ax[1, 1].set_xlim(-6, 4.5)

    ax[0, 0].set_ylim(-0.15, 0.1)
    ax[1, 0].set_ylim(-0.15, 0.1)

    plt.tight_layout()
    if save: plt.savefig('return_compare_bootstrap.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_historical_volatility(data, n_years_list, window_size=252, plot=True, save=False, label='Default'):
    """
    Wykreśla zmienność historyczną dla różnych okresów czasu.

    Parameters:
    -----------
    data : DataFrame
        Dane cenowe z kolumnami 'Date' i 'Close'.
    n_years_list : list
        Lista liczb lat branych pod uwagę do obliczenia zmienności.
    window_size : int, optional
        Liczba dni handlowych w roku (domyślnie 252).
    highlight_2022 : bool, optional
        Czy dodać poziomą linię oznaczającą zmienność w 2022 roku.
    """
    # Konwersja dat i sortowanie danych od najstarszych do najnowszych
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Obliczanie zwrotów logarytmicznych
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()

    # Przygotowanie wykresu
    if plot is True: plt.figure(figsize=(8, 5))

    for n_years in n_years_list:
        # Obliczanie rozmiaru okna w dniach handlowych
        lookback_period = int(n_years * window_size)

        if lookback_period >= len(data):
            print(f"Ostrzeżenie: Okres {n_years} lat przekracza dostępne dane. Pomijam.")
            continue

        # Obliczanie rocznej zmienności za pomocą rolling window
        # Mnożymy przez sqrt(window_size) aby uzyskać roczną zmienność
        data[f'volatility_{n_years}y'] = data['log_return'].rolling(
            window=lookback_period
        ).std() * np.sqrt(window_size) * 100  # Wynik w procentach

        if plot is True:
            # Wykreślanie zmienności
            plt.plot(
                data['Date'],
                data[f'volatility_{n_years}y'],
                label=f'{n_years} {"rok" if n_years == 1 else "lata" if 1 < n_years < 5 else "lat"}'
        )

    if plot is True:
        plt.title(f'Wykres kroczącego odchylenia standardowego rocznego dla {label} dla różnych okresów')
        #plt.xlabel('Data')
        plt.ylabel('Historyczna zmienność roczna (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Okres obliczania:')

        # Formatowanie osi X
        plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save: plt.savefig(f'historic_{label}_volatility_2014.png', format='png', bbox_inches='tight', dpi=300)


        plt.show()

    return data  # Zwracamy dane z obliczonymi zmiennościami


def plot_historical_mean_returns(data, n_years_list, window_size=252, plot=True, save=False, label='Default'):
    """
    Wykreśla średnie zwroty dla różnych okresów czasu.

    Parameters:
    -----------
    data : DataFrame
        Dane cenowe z kolumnami 'Date' i 'Close'.
    n_years_list : list
        Lista liczb lat branych pod uwagę do obliczenia średnich zwrotów.
    window_size : int, optional
        Liczba dni handlowych w roku (domyślnie 252).
    highlight_2022 : bool, optional
        Czy dodać poziomą linię oznaczającą średni zwrot w 2022 roku.
    """
    # Konwersja dat i sortowanie danych od najstarszych do najnowszych
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Obliczanie zwrotów logarytmicznych
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()

    # Przygotowanie wykresu
    if plot is True: plt.figure(figsize=(8, 5))

    for n_years in n_years_list:
        # Obliczanie rozmiaru okna w dniach handlowych
        lookback_period = int(n_years * window_size)

        if lookback_period >= len(data):
            print(f"Ostrzeżenie: Okres {n_years} lat przekracza dostępne dane. Pomijam.")
            continue

        # Obliczanie średnich rocznych zwrotów za pomocą rolling window
        # Mnożymy przez window_size aby uzyskać roczny zwrot
        data[f'mean_return_{n_years}y'] = data['log_return'].rolling(
            window=lookback_period
        ).mean() * window_size * 100  # Wynik w procentach (annualizowany)

        if plot is True:
            # Wykreślanie średnich zwrotów
            plt.plot(
                data['Date'],
                data[f'mean_return_{n_years}y'],
                label=f'{n_years} {"rok" if n_years == 1 else "lata" if 1 < n_years < 5 else "lat"}',
                linewidth=0.7
            )


    if plot is True:
        plt.title(f'Wykres kroczącej średniej dla rocznego zwrotu WIG20 dla różnych okresów')
        #plt.xlabel('Data')
        plt.ylabel('Średni roczny zwrot (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Okres obliczania:')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Linia na poziomie 0%

        # Formatowanie osi X
        plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save: plt.savefig('historic_wig20_mean.png', format='png', bbox_inches='tight', dpi=300)

        plt.show()

    return data  # Zwracamy dane z obliczonymi średnimi zwrotami


def plot_historical_correlation(data1, data2, n_years_list, window_size=252, plot=True, save=False, label1='Asset1', label2='Asset2'):
    """
    Oblicza i wizualizuje kroczącą korelację między dwoma szeregami czasowymi.

    Parametry:
    data1, data2 - dataframe'y zawierające dane z kolumnami 'Date' i 'Close'
    n_years_list - lista okresów (w latach) do obliczenia korelacji
    window_size - liczba dni handlowych w roku (domyślnie 252)
    plot - czy generować wykres (domyślnie True)
    save - czy zapisać wykres do pliku (domyślnie False)
    label1, label2 - etykiety dla szeregów czasowych

    Zwraca:
    DataFrame z obliczonymi korelacjami
    """

    # Konwersja dat i sortowanie danych od najstarszych do najnowszych
    data1 = data1.copy()
    data1['Date'] = pd.to_datetime(data1['Date'])
    data1 = data1.sort_values('Date')

    data2 = data2.copy()
    data2['Date'] = pd.to_datetime(data2['Date'])
    data2 = data2.sort_values('Date')

    # Obliczanie zwrotów logarytmicznych
    data1['log_return'] = np.log(data1['Close'] / data1['Close'].shift(1))
    data1 = data1.dropna()

    data2['log_return'] = np.log(data2['Close'] / data2['Close'].shift(1))
    data2 = data2.dropna()

    # Łączenie danych na podstawie wspólnych dat
    merged_data = pd.merge(
        data1[['Date', 'log_return']],
        data2[['Date', 'log_return']],
        on='Date',
        suffixes=('_1', '_2')
    )

    # Przygotowanie wykresu
    if plot is True:
        plt.figure(figsize=(10, 6))

    for n_years in n_years_list:
        # Obliczanie rozmiaru okna w dniach handlowych
        lookback_period = int(n_years * window_size)

        if lookback_period >= len(merged_data):
            print(f"Ostrzeżenie: Okres {n_years} lat przekracza dostępne dane. Pomijam.")
            continue

        # Obliczanie kroczącej korelacji
        merged_data[f'correlation_{n_years}y'] = merged_data['log_return_1'].rolling(
            window=lookback_period
        ).corr(merged_data['log_return_2'])

        if plot is True:
            # Wykreślanie korelacji
            plt.plot(
                merged_data['Date'],
                merged_data[f'correlation_{n_years}y'],
                label=f'{n_years} {"rok" if n_years == 1 else "lata" if 1 < n_years < 5 else "lat"}'
            )

    if plot is True:
        plt.title(f'Wykres kroczącej korelacji między {label1} a {label2} dla różnych okresów')
        plt.ylabel('Historyczna korelacja')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Okres obliczania:')

        # Dodanie linii poziomych dla wartości referencyjnych
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        plt.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.3)

        # Formatowanie osi X
        plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        # Formatowanie osi Y
        plt.ylim(-1.05, 1.05)  # Korelacja jest w zakresie [-1, 1]

        plt.tight_layout()

        if save:
            plt.savefig(f'historic_correlation_{label1}_{label2}.png', format='png', bbox_inches='tight', dpi=300)

        plt.show()

    return merged_data  # Zwracamy dane z obliczonymi korelacjami



def plot_quantiles_vs_historical(paths_list, historical_data, labels, colors, plot_together=True,
                                 title="Kwantyle symulowanych ścieżek vs rzeczywista trajektoria", save=True):
    """
    Wykreśla kwantyle symulowanych ścieżek na tle historycznych danych.
    Pozwala na elastyczne wyświetlanie wielu metod symulacji razem lub osobno.

    Parameters:
    -----------
    paths_list : list of ndarray
        Lista macierzy symulowanych ścieżek (każda macierz reprezentuje inną metodę).
    historical_data : DataFrame
        Dane historyczne.
    labels : list of str
        Lista etykiet dla każdej metody.
    colors : list of str
        Lista kolorów dla każdej metody.
    plot_together : bool, optional
        Czy wykreślać wszystkie metody na jednym wykresie (True) czy osobno (False).
    title : str, optional
        Tytuł wykresu.
    """
    if not isinstance(paths_list, list):
        paths_list = [paths_list]
        labels = [labels]
        colors = [colors]

    if plot_together:
        # Wszystkie metody na jednym wykresie
        plt.figure(figsize=(8, 5))

        # Wykreślanie danych historycznych
        plt.plot(np.arange(0, len(historical_data)), historical_data.Close, label='Dane historyczne', color='red')

        # Wykreślanie każdej metody
        for paths, label, color in zip(paths_list, labels, colors):
            quantiles_paths = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            time = np.arange(paths.shape[1])

            # Mediana
            plt.plot(time, quantiles_paths[2], label=f"Mediana {label}", color=color)

            # Przedziały kwantylowe
            plt.fill_between(time, quantiles_paths[1], quantiles_paths[3], color=color, alpha=0.2,
                             label=f"IQR (25%-75%) {label}")

            # Ekstremalne kwantyle
            plt.plot(time, quantiles_paths[0], color=color, ls=':', label=f'5% / 95% {label}')
            plt.plot(time, quantiles_paths[4], color=color, ls=':')

        plt.legend()
        plt.xlabel("Czas (dni)")
        plt.ylabel("WIG20")
        plt.title(title)
        plt.tight_layout()
        if save: plt.savefig('compare_gbm_bootstrap_historic_T2.png', format='png', bbox_inches='tight', dpi=300)
        plt.show()

    else:
        # Każda metoda na osobnym wykresie
        for i, (paths, label, color) in enumerate(zip(paths_list, labels, colors)):
            quantiles_paths = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            time = np.arange(paths.shape[1])

            plt.figure(figsize=(8, 5))

            # Mediana
            plt.plot(time, quantiles_paths[2], label=f"Mediana {label}", color=color)

            # Dane historyczne
            plt.plot(np.arange(0, len(historical_data)), historical_data.Close, label='Dane historyczne', color='red')

            # Przedziały kwantylowe
            plt.fill_between(time, quantiles_paths[1], quantiles_paths[3], color=color, alpha=0.2,
                             label=f"IQR (25%-75%) {label}")

            # Ekstremalne kwantyle
            plt.plot(time, quantiles_paths[0], color=color, ls=':', label=f'5% {label}')
            plt.plot(time, quantiles_paths[4], color=color, ls=':', label=f'95% {label}')

            plt.legend()
            plt.xlabel("Czas (dni)")
            plt.ylabel("WIG20")
            plt.title(f"{title} - {label}")
            plt.tight_layout()
            if save: plt.savefig(f'quantile_{label}.png', format='png', bbox_inches='tight', dpi=300)
            plt.show()



def plot_pnl(pnl, title="Rozkład końcowego P/L", save=False, savename='Delta Hedge'):
    """
    Analizuje rozkład końcowego P/L dla wszystkich ścieżek.

    Parameters:
    -----------
    pnl: ndarray
        Macierz P/L dla wszystkich ścieżek
    title: str
        Tytuł wykresu

    Returns:
    --------
    fig: Figure
        Wykres z analizą P/L
    stats: dict
        Statystyki rozkładu P/L
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title)

    # Obliczenie końcowego P/L
    final_pnl = pnl[:, -1] if pnl.ndim > 1 else pnl

    # Statystyki
    mask_nan = np.isnan(final_pnl)
    mu = np.nanmean(final_pnl)
    sigma = np.nanstd(final_pnl)
    positive_paths_pct = 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan])
    median = np.nanmedian(final_pnl)
    skew = scipy_stats.skew(final_pnl[~mask_nan])
    curtosis = scipy_stats.kurtosis(final_pnl[~mask_nan])
    liczba_nan = np.isnan(final_pnl).sum()

    # Histogram z seaborn z normalizacją do procenta
    sns.histplot(final_pnl, kde=True, ax=ax, stat='percent')

    # Linie na histogramie
    ax.axvline(mu, color='r', linestyle='--', label=f"Średnia: ${mu:.2f}")
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label="P&L = 0")

    # Dodanie statystyk do wykresu
    stats_text = (
        f"Średnia: {mu:.2f}\n"
        f"Mediana: {median:.2f}\n"
        f"Odchylenie std: {sigma:.2f}\n"
        f"Procent z zyskiem: {positive_paths_pct:.1f}%\n"
        f"Skośność: {skew:.2f}\n"
        f"Kurtoza: {curtosis:.2f}\n"
    )
    ax.text(0.01, 0.75, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('Końcowy P/L', loc='center')
    ax.set_ylabel('Procent ścieżek [%]')
    ax.legend()

    # Pozostałe statystyki
    stats = {
        'mean': mu,
        'median': median,
        'std': sigma,
        'positive_paths_pct': positive_paths_pct,
        'skewness': skew,
        'kurtosis': curtosis
    }

    plt.tight_layout()
    if save: plt.savefig(savename, format='png', bbox_inches='tight', dpi=300)
    return fig, stats


def plot_qq_pnl(pnl, title="Wykres kwantylowo-kwantylowy (Q-Q) P/L", save=False, savename='delta_hedge_qq'):
    """
    Tworzy wykres kwantylowo-kwantylowy (Q-Q) dla rozkładu P/L w porównaniu do rozkładu normalnego.

    Parameters:
    -----------
    pnl: ndarray
        Macierz P/L dla wszystkich ścieżek (może być 1D lub 2D - w przypadku 2D brany jest ostatni okres)
    title: str
        Tytuł wykresu

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Obiekt wykresu
    stats: dict
        Statystyki opisowe rozkładu P/L
    """
    # Przygotowanie danych
    final_pnl = pnl[:, -1] if pnl.ndim > 1 else pnl

    # Obliczenie statystyk
    # Statystyki
    mask_nan = np.isnan(final_pnl)
    mu = np.nanmean(final_pnl)
    sigma = np.nanstd(final_pnl)
    positive_paths_pct = 100 * np.sum(final_pnl[~mask_nan] > 0) / len(final_pnl[~mask_nan])
    median = np.nanmedian(final_pnl)
    skew = scipy_stats.skew(final_pnl[~mask_nan])
    curtosis = scipy_stats.kurtosis(final_pnl[~mask_nan])

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(8, 5))

    # Wykres Q-Q
    scipy_stats.probplot((final_pnl - mu)/sigma, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor('steelblue')  # Punkty danych
    ax.get_lines()[0].set_markersize(5.0)
    ax.get_lines()[1].set_color('red')  # Linia referencyjna
    ax.get_lines()[1].set_linewidth(2.0)

    # Dodanie statystyk do wykresu
    stats_text = (
        f"Średnia: {mu:.2f}\n"
        f"Mediana: {median:.2f}\n"
        f"Odchylenie std: {sigma:.2f}\n"
        f"Procent z zyskiem: {positive_paths_pct:.1f}%\n"
        f"Skośność: {skew:.2f}\n"
        f"Kurtoza: {curtosis:.2f}\n"
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Formatowanie wykresu
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Kwantyle teoretyczne (rozkład normalny)')
    ax.set_ylabel('Kwantyle empiryczne (P/L)')
    ax.grid(True, alpha=0.3)

    # Statystyki do zwrócenia
    stats = {
        'mean': mu,
        'std': sigma,
        'skewness': skew,
        'curtosis': curtosis
    }
    plt.tight_layout()
    if save: plt.savefig('savename.png', format='png', bbox_inches='tight', dpi=300)
    return fig, stats


def plot_extreme_paths_separately(portfolio_composition, pnl, n_extreme=10,
                                 title="Ekstremalne ścieżki - zyski i straty", save=False):
    """
    Pokazuje osobne wykresy dla n ścieżek z największym zyskiem i n ścieżek z największą stratą.
    Pary (portfolio, opcja) dla tej samej ścieżki mają ten sam kolor.

    Parameters:
    -----------
    portfolio_composition: dict
        Słownik zwracany przez funkcję wallet_delta, zawierający:
        - 'portfolio_value': wartość portfela w czasie
        - 'option_value': wartość opcji w czasie
    pnl: ndarray
        Macierz P/L (używana do określenia których ścieżek użyć)
    n_extreme: int, optional (domyślnie 10)
        Liczba ścieżek do pokazania w każdej kategorii
    title: str, optional
        Tytuł główny wykresu

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Obiekt wykresu zawierający dwa podwykresy
    """
    # Pobranie danych
    portfolio = portfolio_composition['portfolio_value']
    option = portfolio_composition['option_value']
    time_points = portfolio.shape[1]
    time_axis = np.linspace(0, 1, time_points)

    # Identyfikacja ekstremalnych ścieżek
    final_pnl = pnl[:, -1] if pnl.ndim > 1 else pnl
    sorted_indices = np.argsort(final_pnl)

    # Największe straty i zyski
    worst_indices = sorted_indices[:n_extreme]
    best_indices = sorted_indices[-n_extreme:]

    # Generowanie unikalnych kolorów dla każdej ścieżki
    colors = plt.cm.Set1(np.linspace(0, 1, n_extreme))

    # Przygotowanie wykresu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle(title, y=1.05, fontsize=14)

    # Wykres dla największych strat
    for idx, i in enumerate(worst_indices):
        color = colors[idx]
        ax1.plot(time_axis, portfolio[i], color='red', alpha=0.8, linewidth=1,
                 label=f'Portfel {idx+1}' if idx == 0 else "")
        ax1.plot(time_axis, option[i], color='blue', alpha=0.8, linewidth=1, linestyle='-',
                 label=f'Opcja {idx+1}' if idx == 0 else "")

    ax1.set_title(f'Największa strata')
    ax1.set_xlabel('Czas')
    ax1.set_ylabel('Wartość')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Wykres dla największych zysków
    for idx, i in enumerate(best_indices):
        color = colors[idx]
        ax2.plot(time_axis, portfolio[i], color='red', alpha=0.8, linewidth=1,
                 label=f'Portfel {idx+1}' if idx == 0 else "")
        ax2.plot(time_axis, option[i], color='blue', alpha=0.8, linewidth=1, linestyle='-',
                 label=f'Opcja {idx+1}' if idx == 0 else "")

    ax2.set_title(f'Największy zysk')
    ax2.set_xlabel('Czas')
    ax2.set_ylabel('Wartość')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Dodanie informacji o wartościach P/L
    min_pnl = np.min(final_pnl[worst_indices])
    max_pnl = np.max(final_pnl[best_indices])
    fig.text(0.5, 0.93,
             f"Zakres P/L: od {min_pnl:.2f} (największa strata) do {max_pnl:.2f} (największy zysk)",
             ha='center', fontsize=10)

    plt.tight_layout()
    if save: plt.savefig('extreme_paths.png', format='png', bbox_inches='tight', dpi=300)
    return fig


### WALLET
def perform_hedging_frequency_test(paths, params, K, option_type='call', save=False, savename='hedge_freq_test'):
    """
    Przeprowadza test wpływu częstotliwości rehedgingu na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: float
        Cena wykonania opcji
    option_type: str
        Typ opcji ('call' lub 'put')

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej częstotliwości
    """


    # Konfiguracja testów
    frequencies = {
        'Co 2 tygodnie': 26,
        'Tygodniowo': 52,
        'Co 3 dni': 252 // 3,
        #'Dziennie': 252,
        #'2x dziennie': 252 * 2,
        #'5x dziennie': 252 * 5
    }

    # Przygotowanie wyników
    results = []
    figs = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(len(frequencies), 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu częstotliwości rehedgingu na rozkład P&L', y=1.02, fontsize=14)

    for idx, (freq_name, n_hedge) in enumerate(frequencies.items()):

        pnl, _ = wallet_delta(
            gold_paths=paths['asset_0'],
            usdpln_paths=paths['asset_1'],
            T_years=1,
            params=params,
            K=K,
            option_type=option_type,
            n_hedge=n_hedge
        )

        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        mask_nan = np.isnan(final_pnl)
        stats_data = {
            'Częstotliwość': freq_name,
            'Liczba rehedge': n_hedge,
            'Średnia': np.nanmean(final_pnl),
            'Mediana': np.nanmedian(final_pnl),
            'Std': np.nanstd(final_pnl),
            'Skewness':scipy_stats.skew(final_pnl[~mask_nan]),
            'Kurtosis': scipy_stats.kurtosis(final_pnl[~mask_nan]),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Wykres histogramu
        ax1 = axes[idx, 0] if len(frequencies) > 1 else axes[0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'{freq_name} (n={n_hedge}) - Histogram', fontsize=10)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.8)
                 )

        # Wykres Q-Q
        ax2 = axes[idx, 1] if len(frequencies) > 1 else axes[1]
        scipy_stats.probplot(final_pnl / final_pnl.std(), dist="norm", plot=ax2)
        ax2.set_title(f'{freq_name} (n={n_hedge}) - Wykres Q-Q', fontsize=10)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')

    axes[2, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[2, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)

    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)
    #plt.tight_layout()

    if save: plt.savefig(f'{savename}.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[[
        'Częstotliwość', 'Liczba rehedge', 'Średnia', 'Mediana', 'Std',
        'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)'
    ]]

    return fig, stats_df


def compare_pnl_by_strike(paths, params, strikes=None, option_type='call', n_hedge = 252, save=False, savename='strike_test'):
    """
    Analiza wpływu wartości Strike na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    strikes: list
        Lista cen wykonania opcji
    option_type: str
        Typ opcji ('call' lub 'put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike
    """

    if strikes is None or len(strikes) > 3:
        strikes = [1700, 1800, 1900]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu wartości Strike na rozkład P&L', y=1.02, fontsize=14)

    for idx, strike in enumerate(strikes):
        # Obliczenie P&L
        pnl, _ = wallet_delta(
            gold_paths=paths['asset_0'],
            usdpln_paths=paths['asset_1'],
            T_years=1,
            params=params,
            K=strike,
            option_type=option_type,
            n_hedge=n_hedge
        )
        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        stats_data = {
            'Strike': strike,
            'Średnia': np.mean(final_pnl),
            'Mediana': np.median(final_pnl),
            'Std': np.std(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl),
            'Kurtosis': scipy_stats.kurtosis(final_pnl),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Histogram
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'Strike {strike} - Histogram', fontsize=12)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.8))

        ax1.set_xlim(-60, 60)
        ax1.set_ylim(0, 6)

        # Wykres Q-Q
        ax2 = axes[idx, 1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'Strike {strike} - Wykres Q-Q', fontsize=12)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_ylim(-90, 90)

    # Etykiety osi
    axes[-1, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[-1, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)
    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)

    if save:
        plt.savefig(f'{savename}.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df


def compare_pnl_by_volatility(paths, params, K, volatility=None, option_type='call', n_hedge=252, save=False, savename='vol_test'):
    """
    Analiza wpływu wartości volatility na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: int
        Strike
    volatility: list
        Lista volatilites
    option_type: str
        Typ opcji ('call' lub 'put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike
    """

    if volatility is None or len(volatility) != 3:
        volatility = [params['volatility']*1.1, params['volatility'], 0.9*params['volatility']]

    vol_ratio = (np.array(volatility) / params['volatility']) * 100

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu wartości Volatiltiy na rozkład P&L', y=1.02, fontsize=14)

    for idx, volatility in enumerate(volatility):

        params_copy = params.copy()
        # !!!

        pnl, _ = wallet_delta(
            gold_paths=paths['asset_0'],
            usdpln_paths=paths['asset_1'],
            T_years=1,
            params=params,
            K=K,
            option_type=option_type,
            n_hedge=n_hedge
        )
        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        stats_data = {
            'Strike': K,
            'Średnia': np.mean(final_pnl),
            'Mediana': np.median(final_pnl),
            'Std': np.std(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl),
            'Kurtosis': scipy_stats.kurtosis(final_pnl),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Histogram
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'Założone {(volatility/params['volatility']) * 100:.2f}% zrealizowanego volatility - Histogram', fontsize=9)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xlim(-60, 100)
        ax1.set_ylim(0, 7)

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8))

        # Wykres Q-Q
        ax2 = axes[idx, 1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'Założone {(volatility/params['volatility']) * 100:.2f}% zrealizowanego volatility - Wykres Q-Q', fontsize=9)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        #ax1.set_xlim(-120, 50)
        ax2.set_ylim(-70, 120)

    # Etykiety osi
    axes[-1, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[-1, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)
    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)

    #plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        plt.savefig(f'{savename}.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df


def compare_pnl_by_mu(N, params, K, mu=None, option_type='call', save=False):
    """
    Analiza wpływu wartości volatility na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: int
        Strike
    volatility: list
        Lista volatilites
    option_type: str
        Typ opcji ('call' lub 'put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike
    """

    if mu is None or len(mu) != 3:
        mu = [params['mu']*1.1, params['mu'], 0.9*params['mu']]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu wartości Mu na rozkład P&L', y=1.02, fontsize=14)

    for idx, mu in enumerate(mu):
        paths_gbm, _ = generate_gbm_paths(
            N=N,
            T=1,
            s0=params['s0'],
            mean=mu,
            volatility=params['volatility'],
            h=1,
            plot=False
        )

        # Obliczenie P&L
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=params['r'],
            K=K,
            option_type=option_type,
            n_hedge=252
        )
        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        stats_data = {
            'Strike': K,
            'Średnia': np.mean(final_pnl),
            'Mediana': np.median(final_pnl),
            'Std': np.std(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl),
            'Kurtosis': scipy_stats.kurtosis(final_pnl),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Histogram
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'Założone {(mu/params['mu']) * 100:.2f}% zrealizowanej średniej - Histogram', fontsize=9)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xlim(-60, 70)
        ax1.set_ylim(0, 6)

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8))

        # Wykres Q-Q
        ax2 = axes[idx, 1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'Założone {(mu/params['mu']) * 100:.2f}% zrealizowanej średniej - Wykres Q-Q', fontsize=9)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        #ax1.set_xlim(-120, 50)
        ax2.set_ylim(-70, 70)

    # Etykiety osi
    axes[-1, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[-1, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)
    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)

    #plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        plt.savefig('compare_pnl_mu_1.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df

def compare_pnl_by_r(paths_gbm, params, K, r=None, option_type='call', save=False):
    """
    Analiza wpływu wartości volatility na rozkład P&L.

    Parameters:
    -----------
    paths_gbm: ndarray
        Wygenerowane ścieżki cenowe aktywa
    params: dict
        Parametry symulacji (sigma, r)
    K: int
        Strike
    volatility: list
        Lista volatilites
    option_type: str
        Typ opcji ('call' lub 'put')
    save: bool
        Czy zapisać wykres do pliku

    Returns:
    --------
    fig: matplotlib.figure.Figure
        Wykresy wyników
    stats_df: DataFrame
        Statystyki rozkładów P&L dla każdej wartości Strike
    """

    if r is None or len(r) != 3:
        r = [params['r']*1.1, params['r'], 0.9*params['r']]

    # Przygotowanie wyników
    results = []

    # Tworzenie wykresu głównego
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle('Analiza wpływu stopy zwrotu na rozkład P&L', y=1.02, fontsize=14)

    for idx, r in enumerate(r):
        # Obliczenie P&L
        pnl, _ = wallet_delta(
            paths=paths_gbm,
            T_years=1,
            sigma=params['volatility'],
            r=r,
            K=K,
            option_type=option_type,
            n_hedge=252
        )
        final_pnl = pnl if pnl.ndim == 1 else pnl[:, -1]

        # Obliczenie statystyk
        stats_data = {
            'Strike': K,
            'Średnia': np.mean(final_pnl),
            'Mediana': np.median(final_pnl),
            'Std': np.std(final_pnl),
            'Skewness': scipy_stats.skew(final_pnl),
            'Kurtosis': scipy_stats.kurtosis(final_pnl),
            'Min': np.min(final_pnl),
            'Max': np.max(final_pnl),
            'P(PL>0)': np.mean(final_pnl > 0) * 100
        }
        results.append(stats_data)

        # Histogram
        ax1 = axes[idx, 0]
        sns.histplot(final_pnl, kde=True, ax=ax1, stat='percent', color='blue')
        ax1.set_title(f'Założone {(r/params['r']) * 100:.2f}% r - Histogram', fontsize=9)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        #ax1.set_xlim(-60, 100)
        #ax1.set_ylim(0, 7)

        stats_text = (f"Średnia: {stats_data['Średnia']:.2f}\n"
                      f"Mediana: {stats_data['Mediana']:.2f}\n"
                      f"Std: {stats_data['Std']:.2f}\n"
                      f"Skośność: {stats_data['Skewness']:.2f}\n"
                      f"Kurtoza: {stats_data['Kurtosis']:.2f}")
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8))

        # Wykres Q-Q
        ax2 = axes[idx, 1]
        scipy_stats.probplot(final_pnl, dist="norm", plot=ax2)
        ax2.set_title(f'Założone {(r/params['r']) * 100:.2f}% zrealizowanego volatility - Wykres Q-Q', fontsize=9)
        ax2.get_lines()[0].set_markerfacecolor('orange')
        ax2.get_lines()[1].set_color('red')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        #ax1.set_xlim(-120, 50)
        #ax2.set_ylim(-70, 120)

    # Etykiety osi
    axes[-1, 0].set_xlabel('Końcowy P/L', fontsize=9)
    axes[-1, 1].set_xlabel('Teoretyczne kwantyle', fontsize=9)
    axes[1, 0].set_ylabel('Procent ścieżek [%]', fontsize=9)
    axes[1, 1].set_ylabel('Kwantyle P&L portfela', fontsize=9)

    #plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        plt.savefig('compare_pnl_r_1.png', format='png', bbox_inches='tight', dpi=300)

    # Przygotowanie statystyk
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['Strike', 'Średnia', 'Mediana', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'P(PL>0)']]

    return fig, stats_df


def plot_pnl_by_K(paths_gbm, params, n_hedge=252, K_values=None, save=False):

    if K_values is None:
        K_values = [1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
    results = np.zeros((len(K_values), 5))  # 5 kolumn: K, q25, q50, q75, mean

    for i, K in enumerate(K_values):
        pnl, _ = wallet_delta(paths=paths_gbm, T_years=1, sigma=params['volatility'] * 1.01, r=params['r'], K=K,
                              option_type='call', n_hedge=n_hedge)

        pnl_final = pnl

        # Zapisz K i statystyki
        results[i, 0] = K
        results[i, 1] = np.percentile(pnl_final, 25)  # q25
        results[i, 2] = np.percentile(pnl_final, 50)  # mediana (q50)
        results[i, 3] = np.percentile(pnl_final, 75)  # q75
        results[i, 4] = np.mean(pnl_final)  # średnia

    # Tworzenie wykresu
    plt.figure(figsize=(12, 7))

    # Rysowanie linii dla każdej statystyki
    plt.plot(results[:, 0], results[:, 1], 'b-', marker='o', label='Kwantyl 25%')
    plt.plot(results[:, 0], results[:, 2], 'g-', marker='s', label='Mediana')
    plt.plot(results[:, 0], results[:, 3], 'r-', marker='^', label='Kwantyl 75%')
    plt.plot(results[:, 0], results[:, 4], 'k--', marker='*', label='Średnia')

    # Dodawanie opisów
    plt.xlabel('Wartość K')
    plt.ylabel('PNL')
    plt.title('Statystyki PNL w zależności od K')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Wyświetlenie wykresu
    plt.tight_layout()

    if save:
        plt.savefig('pnl_by_K.png', format='png', bbox_inches='tight', dpi=300)

    plt.show()



def plot_deltas(paths, title, label, save, model):
    label = label
    color = 'red'
    quantiles_paths = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    time = np.arange(paths.shape[1])

    # Oblicz min i max dla każdego punktu czasowego
    min_per_time = np.min(paths, axis=0)  # Minimum dla każdego t
    max_per_time = np.max(paths, axis=0)  # Maximum dla każdego t

    fig, axes = plt.subplots(figsize=(8, 5))

    fig.suptitle(title)
    # Mediana
    plt.plot(time, quantiles_paths[2], label=f"Mediana {label}", color=color)

    # Przedziały kwantylowe
    plt.fill_between(time, quantiles_paths[1], quantiles_paths[3], color=color, alpha=0.2,
                     label=f"IQR (25%-75%) {label}")

    # Ekstremalne kwantyle
    plt.plot(time, quantiles_paths[0], color=color, ls=':', label=f'5% / 95% {label}')
    plt.plot(time, quantiles_paths[4], color=color, ls=':')

    # Dodane linie min i max
    plt.plot(time, min_per_time, color='blue', ls='--', alpha=0.7, label='Minimum')
    plt.plot(time, max_per_time, color='green', ls='--', alpha=0.7, label='Maximum')

    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f'{label}_{model}_plot.png', format='png', bbox_inches='tight', dpi=300)

    plt.show()


def plot_portfolio_composition(portfolio_composition, gold_paths, usdpln_paths, path_indices=None, title="Skład portfela w czasie", save=False):
    """
    Wizualizuje skład portfela w czasie dla wybranych ścieżek.

    Parameters:
    -----------
    portfolio_composition: dict
        Słownik zwracany przez funkcję wallet_delta
    path_indices: list, optional
        Lista indeksów ścieżek do pokazania (domyślnie pierwsze 5)
    title: str, optional
        Tytuł wykresu
    save: bool, optional
        Czy zapisać wykres do pliku
    """
    if path_indices is None:
        path_indices = list(range(5))

    fig, axes = plt.subplots(len(path_indices), 1, figsize=(10, 3*len(path_indices)))
    if len(path_indices) == 1:
        axes = [axes]

    time_points = portfolio_composition['gold'].shape[1]
    time_axis = np.linspace(0, 1, time_points)

    for i, path_idx in enumerate(path_indices):
        ax = axes[i]
        ax.plot(time_axis, portfolio_composition['gold'][path_idx] * gold_paths[path_idx] * usdpln_paths[path_idx], label='Złoto')
        ax.plot(time_axis, portfolio_composition['usdpln'][path_idx] * usdpln_paths[path_idx], label='USD/PLN')
        ax.plot(time_axis, portfolio_composition['cash'][path_idx], label='Gotówka')
        ax.plot(time_axis, portfolio_composition['portfolio_value'][path_idx], label='Wartość portfela', linestyle='--')
        ax.set_title(f'Ścieżka {path_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save: plt.savefig('portfolio_composition.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

    return fig


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def option_price(S, K, t, r, sigma, gold_0, option_type, D=0):
    K = K * gold_0
    if t <= 0:
        if option_type == 'call':
            return (100 / gold_0) * np.maximum(S - K, 0)
        else:
            return (100 / gold_0) * np.maximum(K - S, 0)
    d1 = (np.log(S / K) + (r - D + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == 'call':
        return (100 / gold_0) * (S * np.exp(-D*t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2))
    else:
        return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)



def plot_option_heatmap(param1, param2, param1_range, param2_range, fixed_params, option_type='call'):
    param_names = {
        'std_gold': 'Zmienność złota',
        'rho': 'Korelacja',
        'r': 'Stopa procentowa PLN',
        'r_f': 'Stopa procentowa USD',
        'time': 'Czas do wygaśnięcia',
        'strike': 'Cena wykonania'
    }

    # Domyślne parametry
    S = fixed_params.get('S', 1920.185)
    K = fixed_params.get('K', 1.0)
    t = fixed_params.get('t', 1.0)
    r = fixed_params.get('r', 0.06)
    r_f = fixed_params.get('r_f', 0.05)
    sigma = fixed_params.get('sigma', 0.15)
    rho = fixed_params.get('rho', -0.33)
    gold_0 = fixed_params.get('gold_0', S)

    # Przygotowanie siatki
    x = np.linspace(param1_range[0], param1_range[1], 100)
    y = np.linspace(param2_range[0], param2_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Obliczenie ceny opcji dla każdego punktu siatki
    for i in range(len(y)):
        for j in range(len(x)):
            params = {
                'S': S,
                'K': K,
                't': t,
                'r': r,
                'r_f': r_f,
                'sigma': sigma,
                'rho': rho
            }

            # Aktualizacja parametrów
            if param1 == 'std_gold':
                params['sigma'] = x[j]
            elif param1 == 'rho':
                params['rho'] = x[j]
            elif param1 == 'r':
                params['r'] = x[j]
            elif param1 == 'r_f':
                params['r_f'] = x[j]
            elif param1 == 'time':
                params['t'] = x[j]
            elif param1 == 'strike':
                params['K'] = x[j]

            if param2 == 'std_gold':
                params['sigma'] = y[i]
            elif param2 == 'rho':
                params['rho'] = y[i]
            elif param2 == 'r':
                params['r'] = y[i]
            elif param2 == 'r_f':
                params['r_f'] = y[i]
            elif param2 == 'time':
                params['t'] = y[i]
            elif param2 == 'strike':
                params['K'] = y[i]

            # Obliczenie stopy dywidendy
            D = params['r'] - params['r_f'] + params['rho'] * params['sigma'] * fixed_params.get('sigma_usdpln', 0.11)

            # Obliczenie ceny opcji
            Z[i, j] = option_price(params['S'], params['K'], params['t'], params['r'], params['sigma'], gold_0, option_type, D)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.colorbar(label='Cena opcji')
    plt.xlabel(param_names.get(param1, param1))
    plt.ylabel(param_names.get(param2, param2))
    plt.title(f'Heatmapa ceny opcji {option_type} względem {param_names.get(param1, param1)} i {param_names.get(param2, param2)}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()

