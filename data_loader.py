import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def load_data(file_path, start_date=None, end_date=None):
    """
    Wczytuje dane WIG20 i opcjonalnie filtruje po datach.

    Parameters:
    -----------
    file_path : str
        Ścieżka do pliku CSV z danymi.
    start_date : str, optional
        Data początkowa w formacie 'DD-MM-YYYY'.
    end_date : str, optional
        Data końcowa w formacie 'DD-MM-YYYY'.

    Returns:
    --------
    data_filtered : DataFrame
        Przefiltrowane dane.
    stats : dict
        Statystyki dla danych (zwroty, zmienność itp.)
    """
    # Wczytaj dane
    data_f = pd.read_csv(file_path)
    data_f.columns = ['Date', 'Open', 'Max', 'Min', 'Close']
    data_f.Date = pd.to_datetime(data_f.Date, format='%Y-%m-%d')

    # Filtruj daty
    if start_date and end_date:
        data_filtered = data_f[(data_f['Date'] >= start_date) & (data_f['Date'] <= end_date)]
    else:
        data_filtered = data_f

    n = len(data_filtered)

    # Oblicz zwroty
    returns = ((data_filtered.Close - data_filtered.Close.shift(1)) / data_filtered.Close.shift(1)).dropna()
    log_returns = np.log(data_filtered.Close / data_filtered.Close.shift(1)).dropna()

    # Statystyki
    stats = {
        'n': n,
        'returns': returns,
        'log_returns': log_returns,
        'mean': np.mean(log_returns),
        'std': np.std(log_returns),
        's0': data_filtered.Close.iloc[0]
    }

    return data_filtered, stats


def load_multiple_assets(assets1, assets2, start_date=None, end_date=None):
    """
    Parameters:
    -----------
    assets1 : str
        Ścieżka do pliku CSV z danymi assets1.
    assets2 : str
        Ścieżka do pliku CSV z danymi assets2.
    start_date : str, optional
        Data początkowa w formacie 'YYYY-MM-DD'.
    end_date : str, optional
        Data końcowa w formacie 'YYYY-MM-DD'.

    Returns:
    --------
    combined_data : DataFrame
        Połączone dane z cenami zamknięcia dla obu aktywów.
    stats : dict
        Statystyki dla obu aktywów (zwroty, zmienność, korelacja).
    """

    # Wczytaj dane assets1
    assets1_data = pd.read_csv(assets1)
    assets1_data.columns = ['Date', 'Open', 'Max', 'Min', 'Close', 'Volumen']
    assets1_data['Date'] = pd.to_datetime(assets1_data['Date'], format='%Y-%m-%d')
    assets1_data = assets1_data.rename(columns={'Close': 'WIG20'})

    # Wczytaj dane assets2
    assets2_data = pd.read_csv(assets2)
    assets2_data.columns = ['Date', 'Open', 'Max', 'Min', 'Close', 'Volumen']
    assets2_data['Date'] = pd.to_datetime(assets2_data['Date'], format='%Y-%m-%d')
    assets2_data = assets2_data.rename(columns={'Close': 'KGHM'})

    # Połącz dane po datach
    combined_data = pd.merge(assets1_data[['Date', 'WIG20']],
                             assets2_data[['Date', 'KGHM']],
                             on='Date',
                             how='inner')

    # Filtruj daty
    if start_date and end_date:
        combined_data = combined_data[(combined_data['Date'] >= start_date) &
                                      (combined_data['Date'] <= end_date)]

    # Oblicz logarytmiczne zwroty
    combined_data['WIG20_log_return'] = np.log(combined_data['WIG20'] / combined_data['WIG20'].shift(1))
    combined_data['KGHM_log_return'] = np.log(combined_data['KGHM'] / combined_data['KGHM'].shift(1))

    # Usuń wiersze z brakującymi danymi
    combined_data = combined_data.dropna()

    # Oblicz statystyki
    correlation = pearsonr(combined_data['WIG20_log_return'], combined_data['KGHM_log_return'])[0]

    stats = {
        'n': len(combined_data),
        'wig20_mean': np.mean(combined_data['WIG20_log_return']),
        'wig20_std': np.std(combined_data['WIG20_log_return']),
        'kghm_mean': np.mean(combined_data['KGHM_log_return']),
        'kghm_std': np.std(combined_data['KGHM_log_return']),
        'correlation': correlation,
        'covariance': np.cov(combined_data['WIG20_log_return'], combined_data['KGHM_log_return'])[0, 1]
    }

    return combined_data, stats