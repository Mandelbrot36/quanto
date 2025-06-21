"""
Przykłady użycia funkcji do generowania wykresów opcji quanto.
Ten plik zawiera różne scenariusze testowe i przykłady użycia.
"""

import numpy as np
import matplotlib.pyplot as plt
from option_plots_complete import plot_option_prices, calculate_option_prices

def example_basic_usage():
    """
    Podstawowy przykład użycia - statyczny wykres opcji call.
    """
    print("=== Przykład 1: Podstawowy statyczny wykres ===")
    
    # Definicja parametrów
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
    
    # Generowanie statycznego wykresu
    fig = plot_option_prices(
        params=params,
        K=1,
        t=1.0,
        option_type='call',
        theoretic=True
    )
    
    plt.savefig('Option_price_theoretic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Wykres zapisany jako Option_price_theoretic.png")


def example_different_strike():
    """
    Przykład z różnymi cenami wykonania.
    """
    print("\n=== Przykład 3: Różne ceny wykonania ===")

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
    
    # Porównanie różnych cen wykonania
    strikes = [0.5, 1.0, 1.5]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plt.suptitle('Porównanie wrażliwości ceny opcji ze względu na strike', fontsize=18)
    
    for i, K in enumerate(strikes):
        # Obliczenie cen dla każdej ceny wykonania
        S_values = np.linspace(0.7 * params['gold']['s0'], 1.3 * params['gold']['s0'], 100)
        prices_model1, prices_model2 = calculate_option_prices(S_values, K, 1.0, params, 'call')
        
        axes[i].plot(S_values, prices_model1, 'b-', linewidth=2, label='Model 1', alpha=0.8)
        axes[i].plot(S_values, prices_model2, 'r-', linewidth=2, label='Model 2', alpha=0.8)
        axes[i].axvline(params['gold']['s0'], color='gray', linestyle='--', alpha=0.7)
        
        axes[i].set_xlabel('Cena aktywa bazowego (USD)', fontsize=12)
        axes[i].set_ylabel('Cena opcji', fontsize=12)
        axes[i].set_title(f'K = {K}', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('option_prices_strikes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Porównanie różnych cen wykonania zapisane jako example3_strikes.png")


def example_time_decay():
    """
    Przykład pokazujący wpływ czasu na cenę opcji.
    """
    print("\n=== Przykład 4: Wpływ czasu na cenę opcji ===")

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
    
    # Różne czasy do wygaśnięcia
    times = [1, 0.5, 0.1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle(f'Wpływ czasu na cenę opcji w modelu 1 i 2', fontsize=18)
    
    for i, t in enumerate(times):
        S_values = np.linspace(0.7 * params['gold']['s0'], 1.3 * params['gold']['s0'], 100)
        prices_model1, prices_model2 = calculate_option_prices(S_values, 1.0, t, params, 'call')
        
        axes[i].plot(S_values, prices_model1, 'b-', linewidth=2, label='Model 1', alpha=0.8)
        axes[i].plot(S_values, prices_model2, 'r-', linewidth=2, label='Model 2', alpha=0.8)
        axes[i].axvline(params['gold']['s0'], color='gray', linestyle='--', alpha=0.7)
        
        axes[i].set_xlabel('Cena aktywa bazowego (USD)', fontsize=12)
        axes[i].set_ylabel('Cena opcji', fontsize=12)
        axes[i].set_title(f't = {t:.1f} lat do wygaśnięcia opcji', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('option_prices_time_decay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Wpływ czasu na cenę opcji zapisany jako example4_time_decay.png")




def example_sensitivity_analysis():
    """
    Analiza wrażliwości na parametry.
    """
    print("\n=== Przykład 6: Analiza wrażliwości ===")

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
    
    # Analiza wrażliwości na korelację
    correlations = [-1, -0.5, 0.0, 0.5, 0.75, 1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    plt.suptitle(r'Wpływ parametru $\rho$ na cenę opcji dla modelu 1 i 2', fontsize=25)

    
    for i, rho in enumerate(correlations):
        if i >= len(axes):
            break

        params = params.copy()
        params['rho'] = rho
        params['rho_y'] = rho
        
        S_values = np.linspace(0.8 * params['gold']['s0'], 1.2 * params['gold']['s0'], 100)
        prices_model1, prices_model2 = calculate_option_prices(S_values, 1.0, 1.0, params, 'call')
        
        axes[i].plot(S_values, prices_model1, 'b-', linewidth=2, label='Model 1', alpha=0.8)
        axes[i].plot(S_values, prices_model2, 'r-', linewidth=2, label='Model 2', alpha=0.8)
        axes[i].axvline(params['gold']['s0'], color='gray', linestyle='--', alpha=0.7)
        
        axes[i].set_xlabel('Cena aktywa bazowego (USD)', fontsize=14)
        axes[i].set_ylabel('Cena opcji', fontsize=14)
        axes[i].set_title(f'ρ = {rho:.1f}', fontsize=14)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Usuń ostatni pusty subplot
    if len(correlations) < len(axes):
        fig.delaxes(axes[-1])


    plt.savefig('option_price_rho_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Analiza wrażliwości na korelację zapisana jako example6_sensitivity.png")


def run_all_examples():
    """
    Uruchamia wszystkie przykłady.
    """
    print("Uruchamianie wszystkich przykładów...")
    
    #example_basic_usage()
    example_different_strike()
    example_time_decay()
    example_sensitivity_analysis()


if __name__ == "__main__":
    run_all_examples()

