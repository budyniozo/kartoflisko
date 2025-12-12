import warnings
warnings.filterwarnings("ignore")
import platform
import os

# --- 1. AUTOMATYCZNA KONFIGURACJA SYSTEMU ---
SYSTEM_OS = platform.system()
HEADLESS = (SYSTEM_OS == 'Linux')

if HEADLESS:
    import matplotlib
    matplotlib.use('Agg') # Tryb bezokienkowy dla VPS
    print(f"🖥️ Wykryto system: {SYSTEM_OS}. Tryb: HEADLESS (zapis do plików).")
else:
    print(f"🖥️ Wykryto system: {SYSTEM_OS}. Tryb: GUI (wyświetlanie okien).")
# ------------------------------------------

from backtesting import Backtest
from strategies import Strategy2xRSI_Dorsey
from data_loader import prepare_data_with_indicators
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import config

# --- 2. FUNKCJA OCENY (SCORE) ---
def optim_score(stats):
    """
    Ocenia jakość strategii.
    Cel: Wysoki Win Rate (>50%) poparty dużą liczbą transakcji.
    """
    win_rate = stats['Win Rate [%]']
    trades = stats['# Trades']
    
    # FILTR: Odrzucamy strategie z małą liczbą transakcji (szum statystyczny)
    if trades < 30:
        return -1.0

    # Wzór: Nadwyżka WinRate nad 50% * Pierwiastek z liczby transakcji
    # (Używamy pierwiastka, aby 1000 transakcji nie dominowało wyniku nad jakością sygnału)
    return (win_rate - 50) * np.sqrt(trades)

# -----------------------------------------------------------------------

def run_strategy_backtest():
    print(f"--- START BACKTESTU (LTF={config.LTF}, HTF={config.HTF}) ---")
    
    # ==========================================
    # 3. ZAKRESY OPTYMALIZACJI
    # ==========================================
    
    # A. PĘTLA ZEWNĘTRZNA (RSI Length - wymaga przeładowania danych)
    RSI_LENGTHS_TO_TEST = [5, 7, 8, 9, 11, 14] 
    
    # B. OPTYMALIZATOR WEWNĘTRZNY
    r_delta_ltf = range(6, 14, 1) 
    r_delta_htf = range(26, 34,1)
    r_atr = [2.0, 3.0]
    r_rr = [1.0, 1.5] # Sztywne RR=1 dla testu "Edge"

    # Zmienne do śledzenia rekordu
    global_best_score = -9999.0
    global_best_params = {}
    global_best_heatmap = None # Przechowamy heatmapę zwycięzcy

    # Informacyjnie
    combos_per_step = len(r_delta_ltf) * len(r_delta_htf) * len(r_atr) * len(r_rr)
    total_tests = len(RSI_LENGTHS_TO_TEST) * combos_per_step
    print(f"Liczba kombinacji: {total_tests} (w {len(RSI_LENGTHS_TO_TEST)} głównych krokach)\n")

    # ==========================================
    # 4. PROCES OPTYMALIZACJI
    # ==========================================
    
    # Pętla po długościach RSI z paskiem postępu
    for current_rsi_len in tqdm(RSI_LENGTHS_TO_TEST, desc="Postęp Główny"):
        
        # a) Wczytanie danych
        data = prepare_data_with_indicators(config.CSV_PATH, rsi_len=current_rsi_len)
        if data is None: continue

        # b) Init Backtestu
        bt = Backtest(
            data,
            Strategy2xRSI_Dorsey,
            cash=config.CASH,
            commission=config.PROWIZJA,
            margin=0.01 
        )
        
        try:
            # c) Optymalizacja wielowątkowa
            stats, heatmap = bt.optimize(
                rsi_delta_ltf=r_delta_ltf,
                rsi_delta_htf=r_delta_htf,
                atr_multiplier=r_atr,
                risk_reward=r_rr,
                maximize=optim_score,   # <--- Używamy własnej funkcji oceny
                return_heatmap=True     # Pobieramy heatmapę, żeby zapisać ją jeśli wygramy
            )
            
            # d) Ocena wyniku
            current_score = optim_score(stats)
            
            if current_score > global_best_score:
                global_best_score = current_score
                global_best_heatmap = heatmap # Zapisujemy mapę ciepła obecnego lidera
                
                # Zapisujemy parametry mistrza
                global_best_params = {
                    'score': current_score,
                    'wr': stats['Win Rate [%]'],
                    'trades': stats['# Trades'],
                    'rsi_len': current_rsi_len,
                    'delta_htf': stats._strategy.rsi_delta_htf,
                    'delta_ltf': stats._strategy.rsi_delta_ltf,
                    'atr': stats._strategy.atr_multiplier,
                    'rr': stats._strategy.risk_reward
                }
                
                tqdm.write(f"--> NOWY LIDER! RSI({current_rsi_len}) | Score: {current_score:.2f} | WR: {stats['Win Rate [%]']:.2f}% | Trades: {stats['# Trades']}")

        except Exception as e:
            # Ignorujemy błędy braku transakcji w optimize
            pass

    # ==========================================
    # 5. PODSUMOWANIE I RAPORT
    # ==========================================
    print("\n" + "="*50)
    print("       MISTRZ ŚWIATA (GLOBAL BEST)       ")
    print("="*50)
    
    if not global_best_params:
        print("Nie znaleziono strategii spełniającej kryteria (min. 30 transakcji).")
        return

    print(f"💎 Wynik Score:      {global_best_params['score']:.4f}")
    print(f"💰 Win Rate:         {global_best_params['wr']:.2f}%")
    print(f"📊 Liczba transakcji:{global_best_params['trades']}")
    print("-" * 30)
    print(f"🏆 RSI Len:      {global_best_params['rsi_len']}")
    print(f"   Delta HTF:    {global_best_params['delta_htf']}")
    print(f"   Delta LTF:    {global_best_params['delta_ltf']}")
    print(f"   ATR Mult:     {global_best_params['atr']}")
    print(f"   Risk/Reward:  {global_best_params['rr']}")
    print("="*50)

    # --- A. GENROWANIE MAPY CIEPŁA DLA ZWYCIĘZCY ---
    if global_best_heatmap is not None:
        try:
            print("\nGeneruję mapę ciepła dla zwycięskiej konfiguracji...")
            # Grupowanie wg delty HTF i LTF, biorąc MAX Score (lub max WR, zależnie co heatmapa trzyma)
            # Domyślnie heatmapa trzyma wartość z maximize (czyli Score)
            heatmap_grouped = global_best_heatmap.groupby(['rsi_delta_htf', 'rsi_delta_ltf']).max()
            hm_matrix = heatmap_grouped.unstack()
            
            # Zapis do CSV
            hm_matrix.to_csv("best_heatmap_score.csv")
            
            # Wykres
            plt.figure(figsize=(10, 8))
            sns.heatmap(hm_matrix, annot=True, fmt='.1f', cmap='viridis', cbar_kws={'label': 'Optimization Score'})
            plt.title(f'Score Heatmap (RSI Len={global_best_params["rsi_len"]})')
            plt.xlabel('RSI Delta LTF')
            plt.ylabel('RSI Delta HTF')
            plt.gca().invert_yaxis()
            
            # Zapis pliku
            plt.savefig("best_heatmap.png")
            print("Zapisano: best_heatmap.png oraz best_heatmap_score.csv")
            
            # Wyświetlenie (tylko Windows)
            if not HEADLESS:
                plt.show()
                
            plt.close()
        except Exception as e:
            print(f"Błąd rysowania mapy: {e}")

    # --- B. SZCZEGÓŁOWY RAPORT I WYKRES EQUITY ---
    print("\nUruchamiam szczegółowy test dla zwycięzcy...")
    
    # 1. Ponowne wczytanie danych
    final_data = prepare_data_with_indicators(config.CSV_PATH, rsi_len=global_best_params['rsi_len'])
    
    # 2. Uruchomienie testu
    bt_final = Backtest(
        final_data, 
        Strategy2xRSI_Dorsey, 
        cash=config.CASH, 
        commission=config.PROWIZJA, 
        margin=0.01
    )
    
    final_stats = bt_final.run(
        rsi_delta_ltf=global_best_params['delta_ltf'],
        rsi_delta_htf=global_best_params['delta_htf'],
        atr_multiplier=global_best_params['atr'],
        risk_reward=global_best_params['rr'] # float
    )
    
    print(final_stats)

    # 3. Zapis HTML
    try:
        filename = "Best_Strategy_Results.html"
        # Otwórz przeglądarkę tylko jeśli NIE jesteśmy na Linuxie
        bt_final.plot(filename=filename, open_browser=(not HEADLESS))
        print(f"\nZapisano raport HTML do: {filename}")
    except Exception as e:
        print(f"\nBłąd generowania HTML: {e}")

if __name__ == '__main__':
    run_strategy_backtest()
