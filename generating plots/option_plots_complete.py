import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import seaborn as sns


try:
    from IPython.display import HTML
except ImportError:
    HTML = None

def setup_plotting_style():
    """Konfiguruje styl wykresów dla całego projektu."""
    plt.rc("figure",
           autolayout=True,
           figsize=(12, 8),
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


def option_price_bs(S, K, t, r, sigma, option_type='call', D=0):
    """
    Wycena opcji metodą Black-Scholes z dywidendą.
    
    Parameters:
    -----------
    S : float or array
        Cena aktywa bazowego
    K : float
        Cena wykonania
    t : float
        Czas do wygaśnięcia (w latach)
    r : float
        Stopa procentowa wolna od ryzyka
    sigma : float
        Zmienność aktywa bazowego
    option_type : str
        Typ opcji ('call' lub 'put')
    D : float
        Stopa dywidendy ciągłej
        
    Returns:
    --------
    float or array
        Cena opcji
    """
    if t <= 0:
        if option_type == 'call':
            return np.maximum(S - K, 0)
        else:
            return np.maximum(K - S, 0)
    
    d1 = (np.log(S / K) + (r - D + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    if option_type == 'call':
        price = S * np.exp(-D * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-D * t) * norm.cdf(-d1)
    
    return price


def calculate_option_prices(S_values, K, t, params, option_type='call', theoretic=False):
    """
    Oblicza ceny opcji w obu modelach dla zakresu cen aktywa.
    
    Parameters:
    -----------
    S_values : array
        Zakres cen aktywa bazowego
    K : float
        Cena wykonania
    t : float
        Czas do wygaśnięcia
    params : dict
        Parametry modelu
    option_type : str
        Typ opcji ('call' lub 'put')
        
    Returns:
    --------
    tuple
        (prices_model1, prices_model2) - ceny opcji w obu modelach
    """
    # Model 1: Obliczenie dividend rate
    dividend_rate_1 = params['r'] - params['r_f'] + params['rho'] * params['gold']['std'] * params['usdpln']['std']


    if not theoretic:
        D_2 = params['r'] - params['r_f'] - params['usdpln']['std']**2 + params['rho_y'] * params['Y']['std'] * params['usdpln']['std']
        sigma_y = np.sqrt(params['Y']['std']**2 + params['usdpln']['std']**2 - 2 * params['rho_y'] * params['Y']['std'] * params['usdpln']['std'])
    else:
        D_2 = params['r'] - params['r_f'] - params['usdpln']['std']**2 + params['rho_y_theoretic'] * params['Y_theoretic']['std'] * params['usdpln']['std']
        sigma_y = np.sqrt(params['Y_theoretic']['std']**2 + params['usdpln']['std']**2 - 2 * params['rho_y_theoretic'] * params['Y_theoretic']['std'] * params['usdpln']['std'])
    
    prices_model1 = []
    prices_model2 = []
    
    for S in S_values:
        # Model 1: Opcja quanto na złoto
        price1 = option_price_bs(S=S,
                                K=K * params['gold']['s0'],
                                t=t,
                                r=params['r'],
                                sigma=params['gold']['std'],
                                option_type=option_type,
                                D=dividend_rate_1)
        # Normalizacja przez exchange rate i gold_0
        normalized_price1 = price1 * 100 / params['gold']['s0']
        prices_model1.append(normalized_price1)
        
        # Model 2: Opcja quanto na Y (złoto w PLN)
        # S w modelu 2 to Y/USD, więc musimy przekonwertować
        Y_value = S * params['usdpln']['s0']  # Konwersja z USD na PLN
        S_model2 = Y_value / params['usdpln']['s0']  # Y/USD
        
        price2 = option_price_bs(S=S_model2,
                                K=K * params['gold']['s0'],
                                t=t,
                                r=params['r'],
                                sigma=sigma_y,
                                option_type=option_type,
                                D=D_2)
        # Normalizacja
        normalized_price2 = price2 * 100 / params['gold']['s0']
        prices_model2.append(normalized_price2)
    
    return np.array(prices_model1), np.array(prices_model2)


def plot_option_prices(params, K=1, t=None, t_range=None, S_range=None, option_type='call', 
                      fps=6, save_animation=False, filename='option_prices_animation.gif', theoretic=False):
    """
    Generuje wykres ceny opcji w dwóch modelach w zależności od ceny aktywa.
    
    Parameters:
    -----------
    params : dict
        Słownik z parametrami modelu
    K : float, optional
        Cena wykonania, domyślnie 1
    t : float, optional
        Czas do wygaśnięcia (w latach) dla statycznego wykresu
    t_range : list or array, optional
        Zakres czasów do wygaśnięcia dla animowanego wykresu
    S_range : tuple, optional
        Zakres cen aktywa (min, max, liczba_punktów), domyślnie (0.7*S0, 1.3*S0, 100)
    option_type : str, optional
        Typ opcji ('call' lub 'put'), domyślnie 'call'
    fps : int, optional
        Liczba klatek na sekundę dla animacji, domyślnie 3
    save_animation : bool, optional
        Czy zapisać animację do pliku, domyślnie False
    filename : str, optional
        Nazwa pliku do zapisu animacji, domyślnie 'option_prices_animation.gif'
        
    Returns:
    --------
    matplotlib.figure.Figure or matplotlib.animation.FuncAnimation
        Obiekt figury (dla statycznego wykresu) lub animacji (dla animowanego wykresu)
    """
    
    # Sprawdzenie, czy generować statyczny wykres czy animację
    if t is not None and t_range is None:
        # Statyczny wykres
        return _plot_static_option_prices(params, K, t, S_range, option_type, theoretic=theoretic)
    elif t_range is not None and t is None:
        # Animowany wykres
        return _plot_animated_option_prices(params, K, t_range, S_range, option_type, fps, save_animation, filename, theoretic=theoretic)
    else:
        raise ValueError("Podaj albo 't' dla statycznego wykresu albo 't_range' dla animowanego wykresu")


def _plot_static_option_prices(params, K, t, S_range, option_type, theoretic=False):
    """
    Generuje statyczny wykres ceny opcji w dwóch modelach.
    """
    # Ustawienie zakresu cen aktywa
    if S_range is None:
        S_min = 0.7 * params['gold']['s0']
        S_max = 1.3 * params['gold']['s0']
        n_points = 100
    else:
        S_min, S_max, n_points = S_range
    
    S_values = np.linspace(S_min, S_max, n_points)
    
    # Obliczenie cen opcji w obu modelach
    if not theoretic:
        prices_model1, prices_model2 = calculate_option_prices(S_values, K, t, params, option_type)
    else:
        prices_model1, prices_model2 = calculate_option_prices(S_values, K, t, params, option_type, theoretic=True)

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(S_values, prices_model1, 'b-', linewidth=2, label='Model 1', alpha=0.8)
    ax.plot(S_values, prices_model2, 'r-', linewidth=2, label='Model 2', alpha=0.8)
    
    # Dodanie linii pionowej dla aktualnej ceny aktywa
    ax.axvline(params['gold']['s0'], color='gray', linestyle='--', alpha=0.7, 
               label=f"Aktualna cena: {params['gold']['s0']:.2f}")
    
    # Formatowanie wykresu
    ax.set_xlabel('Cena aktywa bazowego (USD)', fontsize=12)
    ax.set_ylabel('Cena opcji', fontsize=12)
    ax.set_title(f'Cena opcji {option_type} w modelu 1 i modelu 2 (teoretyczna estymacja parametrów) (t={t:.3f} lat do wykonania, K={K})', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_animated_option_prices(params, K, t_range, S_range, option_type, fps, save_animation, filename, theoretic=False):
    """
    Generuje animowany wykres ceny opcji w dwóch modelach.
    """
    # Ustawienie zakresu cen aktywa
    if S_range is None:
        S_min = 0.3 * params['gold']['s0']
        S_max = 1.7 * params['gold']['s0']
        n_points = 500
    else:
        S_min, S_max, n_points = S_range
    
    S_values = np.linspace(S_min, S_max, n_points)
    
    # Przygotowanie wykresu
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Inicjalizacja linii
    line1, = ax.plot([], [], 'b-', linewidth=2, label='Model 1', alpha=0.8)
    line2, = ax.plot([], [], 'r-', linewidth=2, label='Model 2', alpha=0.8)
    vline = ax.axvline(params['gold']['s0'], color='gray', linestyle='--', alpha=0.7, 
                       label=f"Aktualna cena: {params['gold']['s0']:.2f}")
    
    # Ustawienie stałych elementów wykresu
    ax.set_xlim(S_min, S_max)
    ax.set_ylim(0, 40)
    ax.set_xlabel('Cena aktywa bazowego (USD)', fontsize=12)
    ax.set_ylabel('Cena opcji', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Funkcja aktualizująca wykres dla każdej klatki
    def update_plot(frame_idx):
        t = t_range[frame_idx]

        # Obliczenie cen opcji w obu modelach
        if not theoretic:
            prices_model1, prices_model2 = calculate_option_prices(S_values, K, t, params, option_type)
        else:
            prices_model1, prices_model2 = calculate_option_prices(S_values, K, t, params, option_type, theoretic=True)
        
        # Aktualizacja linii
        line1.set_data(S_values, prices_model1)
        line2.set_data(S_values, prices_model2)
        
        # Aktualizacja tytułu
        ax.set_title(f'Cena opcji {option_type} w modelu 1 i modelu 2 (teoretyczna estymacja parametrów) (t={t:.3f} lat do wykonania, K={K})', fontsize=12)
        
        # Aktualizacja zakresu osi Y
        all_prices = np.concatenate([prices_model1, prices_model2])
        valid_prices = all_prices[np.isfinite(all_prices)]
        if len(valid_prices) > 0:
            y_min = np.min(valid_prices) * 0.95
            y_max = np.max(valid_prices) * 1.05
            if y_max > y_min:
                ax.set_ylim(y_min, y_max)
        
        return line1, line2
    
    # Tworzenie animacji
    anim = animation.FuncAnimation(
        fig, update_plot, frames=len(t_range), interval=1000/fps, blit=False, repeat=False
    )
    
    # Zapisanie animacji do pliku
    if save_animation:
        try:
            anim.save(filename, writer='pillow', fps=fps)
            print(f"Animacja zapisana do pliku: {filename}")
        except Exception as e:
            print(f"Błąd przy zapisywaniu animacji: {e}")
            print("Próba zapisania z domyślnymi ustawieniami...")
            anim.save(filename)
            print(f"Animacja zapisana do pliku: {filename} (domyślne ustawienia)")
    
    plt.tight_layout()
    return anim


def display_animation(anim):
    """
    Wyświetla animację w notebooku Jupyter.
    """
    if HTML is not None:
        return HTML(anim.to_jshtml())
    else:
        print("IPython nie jest dostępne. Animacja może być wyświetlona tylko w notebooku Jupyter.")
        return None


# Test funkcji
def test_functions():
    """
    Testuje funkcje generujące wykresy.
    """
    # Przykładowe parametry
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
    
    #print("Test 1: Statyczny wykres")
    fig = plot_option_prices(params, K=1, t=1.0, option_type='call')
    plt.savefig('static_plot_corrected.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Statyczny wykres zapisany jako static_plot_corrected.png")
    
    '''print("\nTest 2: Animowany wykres dla modelu 1 i 2 podejście naiwne")
    t_range = np.linspace(1.0, 0.01, 50)
    anim = plot_option_prices(params, K=1, t_range=t_range, option_type='call', 
                             save_animation=True, filename='option_prices_test.gif')
    print("Animacja zapisana jako option_prices_test.gif")
    
    return anim'''

    '''print("\nTest 2: Animowany wykres dla modelu 1 i 2 podejście teoretyczne")
    t_range = np.linspace(1.0, 0.01, 50)
    anim = plot_option_prices(params, K=1, t_range=t_range, option_type='call',
                              save_animation=True, filename='option_prices_test_theoretic.gif', theoretic=True)
    print("Animacja zapisana jako option_prices_test.gif")

    return anim'''


if __name__ == "__main__":
    anim = test_functions()

