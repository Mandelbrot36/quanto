import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate_gbm_paths(N, T, s0, mean, volatility, h=1, plot=False):
    """
    Generuje ścieżki ceny przy użyciu geometrycznego ruchu Browna.

    Parameters:
    -----------
    N : int
        Liczba ścieżek.
    T : int
        Horyzont czasowy (liczba lat).
    s0 : float
        Początkowa cena aktywa.
    mean : float
        Średnia logarytmicznych zwrotów rocznych.
    volatility : float
        Odchylenie standardowe logarytmicznych zwrotów rocznych.
    h : float, optional
        Krok czasowy w dniach. Domyślnie 1 dzień.
    plot : bool, optional
        Czy wykreślić ścieżki. Domyślnie False.

    Returns:
    --------
    paths : ndarray
        Macierz ścieżek cenowych (N x (T / dt)).
    time : ndarray
        Macierz punktów czasowych.
    """
    dt = h / 252  # Konwersja kroku dziennego na część roku (252 dni handlowych)
    n = int(T / dt)

    # Przeskalowane parametry dla danego kroku czasowego
    mu_dt = mean * dt
    sigma_dt = volatility * np.sqrt(dt)

    # Generowanie przyrostów logarytmicznych dla GBM
    increments = np.exp((mu_dt - sigma_dt ** 2 / 2) +
                        sigma_dt * np.random.normal(0, 1, size=(N, n-1)))

    # Dodanie punktu początkowego i obliczenie ścieżek cenowych
    paths = np.hstack([np.ones((N, 1)), increments])
    paths = s0 * paths.cumprod(axis=1)

    time = np.linspace(0, T, n)
    matrixtime = np.full(shape=(N, n), fill_value=time)

    if plot is True:
        plt.figure(figsize=(12, 8))
        plt.plot(matrixtime.T, paths.T, alpha=0.3)
        plt.title(f'Symulowane ścieżki ({N}) - Geometryczny ruch Browna')
        plt.xlabel(f'Czas (lata)')
        plt.ylabel('Wartość aktywa')

    return paths, matrixtime


def generate_correlated_gbm_paths(N, T, s0_vec, mean_vec, vol_vec, corr_matrix, h=1):
    """
    Generuje ścieżki cen dla dwóch lub więcej aktywów przy użyciu skorelowanego geometrycznego ruchu Browna.

    Parameters:
    -----------
    N : int
        Liczba ścieżek.
    T : int
        Horyzont czasowy (liczba lat).
    s0_vec : list or ndarray
        Wektor początkowych cen aktywów.
    mean_vec : list or ndarray
        Wektor średnich ROCZNYCH stóp zwrotu (drift) dla każdego aktywa.
    vol_vec : list or ndarray
        Wektor ROCZNEJ zmienności (volatility) dla każdego aktywa.
    corr_matrix : ndarray
        Macierz korelacji między aktywami.
    h : float, optional
        Krok czasowy. Domyślnie 1 dzień.

    Returns:
    --------
    paths : dict
        Słownik zawierający macierze ścieżek cenowych dla każdego aktywa.
    time : ndarray
        Macierz punktów czasowych.

    Notes:
    ------
    Model GBM: dS_t = μ * S_t * dt + σ * S_t * dW_t
    Rozwiązanie: S_t = S_0 * exp((μ - σ²/2)*t + σ*W_t)
    """

    # Konwersja wektorów na tablice numpy
    s0_vec = np.array(s0_vec)
    mean_vec = np.array(mean_vec)
    vol_vec = np.array(vol_vec)

    dt = h / 252  # Konwersja kroku dziennego na część roku (252 dni handlowych)
    n = int(T / dt)
    num_assets = len(s0_vec)

    # Przeskalowane parametry dla danego kroku czasowego
    mu_dt = mean_vec * dt
    sigma_dt = vol_vec * np.sqrt(dt)

    # Dekompozycja Choleskiego macierzy korelacji
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Jeśli macierz nie jest dodatnio określona, dodaj małą wartość do diagonali
        epsilon = 1e-6
        corr_matrix_adjusted = corr_matrix + np.eye(num_assets) * epsilon
        L = np.linalg.cholesky(corr_matrix_adjusted)

    # Generowanie nieskorelowanych losowych liczb
    Z = np.random.normal(0, 1, size=(N, n-1, num_assets))

    # Korelacja liczb losowych - wektoryzacja operacji
    correlated_Z = np.matmul(Z, L.T)

    # POPRAWKA: Generowanie przyrostów logarytmicznych dla GBM
    # log_increments = (μ - σ²/2)*dt + σ*sqrt(dt)*Z
    drift_term = (mu_dt - 0.5 * sigma_dt**2).reshape(1, 1, -1)
    volatility_term = sigma_dt.reshape(1, 1, -1) * correlated_Z
    log_increments = drift_term + volatility_term

    # Inicjalizacja ścieżek
    paths = {}
    for i in range(num_assets):
        # Dodanie punktu początkowego (log_price = log(S0))
        log_price_increments = np.hstack([np.zeros((N, 1)), log_increments[:, :, i]])

        # Suma skumulowana przyrostów logarytmicznych
        log_prices = np.cumsum(log_price_increments, axis=1)

        # Przekształcenie z powrotem do cen: S_t = S_0 * exp(suma przyrostów log)
        paths[f'asset_{i}'] = s0_vec[i] * np.exp(log_prices)

    # Przygotowanie wektora czasu
    time = np.linspace(0, T, n)

    return paths, time


def test_gbm_properties(paths, time, mean_vec, vol_vec, s0_vec):
    """
    Testuje właściwości statystyczne wygenerowanych ścieżek GBM.
    """
    results = {}
    dt = time[1] - time[0]

    for i, asset_key in enumerate(paths.keys()):
        prices = paths[asset_key]

        # Oblicz logarytmiczne stopy zwrotu
        log_returns = np.diff(np.log(prices), axis=1)

        # Teoretyczne wartości
        theoretical_mean = (mean_vec[i] - 0.5 * vol_vec[i]**2) * dt
        theoretical_std = vol_vec[i] * np.sqrt(dt)

        # Empiryczne wartości
        empirical_mean = np.mean(log_returns)
        empirical_std = np.std(log_returns, ddof=1)

        results[asset_key] = {
            'theoretical_mean': theoretical_mean,
            'empirical_mean': empirical_mean,
            'theoretical_std': theoretical_std,
            'empirical_std': empirical_std,
            'mean_error': abs(empirical_mean - theoretical_mean),
            'std_error': abs(empirical_std - theoretical_std)
        }

    return print(results)


def validate_paths_correlation(paths_dict, expected_corr_matrix, time_window=None):
    """
    Waliduje czy wygenerowane ścieżki mają oczekiwaną korelację.

    Parameters:
    -----------
    paths_dict : dict
        Słownik zawierający ścieżki w formacie {'asset_0': array, 'asset_1': array, ...}
        gdzie każde array ma kształt [N, num_steps+1]
    expected_corr_matrix : ndarray
        Oczekiwana macierz korelacji
    time_window : tuple, optional
        (start_idx, end_idx) - okno czasowe do analizy

    Returns:
    --------
    empirical_corr : ndarray
        Empiryczna macierz korelacji log-zwrotów
    max_error : float
        Maksymalny błąd korelacji
    """
    # Konwersja słownika do macierzy 3D
    asset_keys = sorted([k for k in paths_dict.keys() if k.startswith('asset_')])
    num_assets = len(asset_keys)

    if num_assets == 0:
        raise ValueError("Brak ścieżek aktywów w słowniku (klucze 'asset_0', 'asset_1', ...)")

    # Sprawdź wymiary
    first_asset = paths_dict[asset_keys[0]]
    N, num_steps_plus_1 = first_asset.shape

    # Konwertuj do macierzy [N, num_steps+1, num_assets]
    paths_array = np.zeros((N, num_steps_plus_1, num_assets))
    for i, asset_key in enumerate(asset_keys):
        paths_array[:, :, i] = paths_dict[asset_key]

    if time_window is None:
        start_idx, end_idx = 1, num_steps_plus_1
    else:
        start_idx, end_idx = time_window

    # Oblicz log-zwroty dla każdego aktywa
    log_returns = np.log(paths_array[:, start_idx:end_idx, :] / paths_array[:, start_idx-1:end_idx-1, :])

    # Reshape: [N*(end_idx-start_idx), num_assets]
    log_returns_flat = log_returns.reshape(-1, num_assets)

    # Empiryczna korelacja
    empirical_corr = np.corrcoef(log_returns_flat.T)

    # Błąd względem oczekiwanej korelacji
    max_error = np.max(np.abs(empirical_corr - expected_corr_matrix))

    print(f"Znalezione aktywa: {asset_keys}")
    print(f"Kształt danych: {N} ścieżek, {num_steps_plus_1-1} kroków czasowych, {num_assets} aktywów")
    print(f"Oczekiwana korelacja:\n{expected_corr_matrix}")
    print(f"Empiryczna korelacja:\n{empirical_corr}")
    print(f"Maksymalny błąd korelacji: {max_error:.6f}")

    return empirical_corr, max_error
