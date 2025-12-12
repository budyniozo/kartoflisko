import warnings
warnings.filterwarnings("ignore")
import platform

# --- 1. KONFIGURACJA SYSTEMU (Linux/Windows) ---
SYSTEM_OS = platform.system()
HEADLESS = (SYSTEM_OS == 'Linux')

if HEADLESS:
    import matplotlib
    matplotlib.use('Agg')
    print("Tryb: HEADLESS (zapis do plików)")
else:
    print("Tryb: GUI (okienka)")

from backtesting import Backtest
from strategies import Strategy2xRSI_Dorsey
from data_loader import prepare_data_with_indicators
import config
import matplotlib.pyplot as plt

# ==========================================
# 2. TUTAJ WPISZ PARAMETRY "MISTRZA"
# ==========================================
# Te wartości wziąłeś z wyniku optymalizacji (np. z pliku 209.txt)
BEST_RSI_LEN = 7
BEST_DELTA_HTF = 32
BEST_DELTA_LTF = 11
BEST_ATR_MULT = 4.0
BEST_RR = 1.0

# ==========================================
# 3. TUTAJ WPISZ ŚCIEŻKĘ DO DANYCH Z 2024
# ==========================================
# Jeśli masz osobny plik dla 2024:
PATH_2024 = r"xauusd2024_dukas_ohlcv.csv"

# Jeśli masz ten sam duży plik, użyjemy go, a daty przefiltrujemy niżej
# PATH_2024 = config.CSV_PATH 

def run_single_test():
    print(f"--- START POJEDYNCZEGO TESTU ---")
    print(f"Dane: {PATH_2024}")
    print(f"Parametry: RSI={BEST_RSI_LEN}, HTF={BEST_DELTA_HTF}, LTF={BEST_DELTA_LTF}, ATR={BEST_ATR_MULT}")

    # 1. Przygotowanie danych (Obliczenie wskaźników)
    # Ważne: Musimy podać RSI_LEN tutaj, bo to wpływa na budowę kolumn
    data = prepare_data_with_indicators(PATH_2024, rsi_len=BEST_RSI_LEN)
    
    if data is None: return

    # --- FILTROWANIE DATY (OPCJONALNE) ---
    # Jeśli wczytałeś duży plik (2010-2025), a chcesz testować tylko 2024:
    # Odkomentuj poniższą linię:
    # data = data.loc['2024-01-01':'2024-12-31']
    # -------------------------------------

    print(f"Zakres dat: {data.index[0]} do {data.index[-1]}")
    print(f"Liczba świec: {len(data)}")

    # 2. Konfiguracja Backtestu
    bt = Backtest(
        data,
        Strategy2xRSI_Dorsey,
        cash=config.CASH,
        commission=config.PROWIZJA,
        margin=0.01 
    )

    # 3. Uruchomienie (bt.run zamiast bt.optimize)
    stats = bt.run(
        rsi_delta_ltf=BEST_DELTA_LTF,
        rsi_delta_htf=BEST_DELTA_HTF,
        atr_multiplier=BEST_ATR_MULT,
        risk_reward=BEST_RR
    )

    # 4. Wyniki
    print("\n" + "="*40)
    print("       WYNIK WERYFIKACJI (2024)       ")
    print("="*40)
    print(stats)
    
    # 5. Wykresy
    try:
        filename = "Verification_Result_2024.html"
        # Otwieramy przeglądarkę tylko na Windowsie
        bt.plot(filename=filename, open_browser=(not HEADLESS))
        print(f"\nZapisano raport HTML: {filename}")
        
        # Opcjonalnie zapisz sam wykres kapitału jako obrazek (dla Linuxa)
        if HEADLESS:
            # Backtesting.py nie ma prostej metody exportu do PNG samego equity,
            # ale plik HTML wystarczy (można go ściągnąć i otworzyć lokalnie).
            pass
            
    except Exception as e:
        print(f"Błąd generowania wykresu: {e}")

if __name__ == '__main__':
    run_single_test()