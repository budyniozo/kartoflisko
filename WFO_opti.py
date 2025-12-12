import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import platform
import itertools
from backtesting import Backtest

# --- IMPORTY PROJEKTU ---
try:
    from strategies import Strategy2xRSI_Dorsey
    from data_loader import prepare_data_with_indicators
except ImportError as e:
    print(f"❌ BŁĄD IMPORTU: {e}")
    print("Upewnij się, że pliki strategies.py i data_loader.py są w tym samym folderze.")
    exit()

# --- KONFIGURACJA ---
NAZWA_PLIKU = "xauusd_FULL_2024_2025.csv"
LTF = '2min'       
HTF_RES = '1h'     
PROWIZJA = 0.000008
KAPITAL_POCZATKOWY = 10000

# Wykrywanie Systemu
SYSTEM_OPERACYJNY = platform.system() # 'Windows', 'Linux', 'Darwin' (Mac)
IS_WINDOWS = True

print(f"🖥️  Wykryto system: {SYSTEM_OPERACYJNY}")
if IS_WINDOWS:
    print("   👉 Tryb: WINDOWS SAFE (Ręczna pętla, jeden rdzeń, pełna stabilność)")
else:
    print("   👉 Tryb: LINUX PERFORMANCE (Multiprocessing, pełna moc CPU)")

def find_data_file(filename):
    possible_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join(os.getcwd(), filename),
        # Dodaj swoje specyficzne ścieżki jeśli chcesz:
        r"D:/____aaa botyyy/_r312/" + filename, 
    ]
    for path in possible_paths:
        if os.path.exists(path): return path
    return None

def manual_optimization_windows(bt_instance, param_grid):
    """
    WINDOWS ONLY: Ręczna, bezpieczna pętla bez multiprocessingu.
    """
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_result = -np.inf
    best_params = None
    
    # Prosta pętla for (jeden wątek)
    for params in combinations:
        try:
            stats = bt_instance.run(**params)
            result = stats['Equity Final [$]']
            if result > best_result:
                best_result = result
                best_params = params
        except Exception:
            continue
            
    if best_params is None:
        best_params = combinations[0]
        
    # Tworzymy atrapę obiektu wyniku, żeby reszta kodu działała tak samo
    class MockResult:
        def __init__(self, p):
            for k, v in p.items(): setattr(self, k, v)
    return MockResult(best_params)

def walk_forward_optimization(data, strategy_class, window_days=90, step_days=30):
    start_date = data.index[0]
    end_date = data.index[-1]
    current_date = start_date
    results_log = []
    
    print(f"\n--- ROZPOCZYNAM WALK-FORWARD ANALYSIS ---")
    print(f"Zakres: {start_date} -> {end_date}")
    
    iteration = 0
    while current_date + pd.Timedelta(days=window_days + step_days) < end_date:
        iteration += 1
        
        # 1. Definicja Okien
        train_start = current_date
        train_end = current_date + pd.Timedelta(days=window_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=step_days)
        
        train_data = data.loc[train_start:train_end]
        test_data = data.loc[test_start:test_end]
        
        if len(train_data) < 500 or len(test_data) < 50:
            current_date += pd.Timedelta(days=step_days)
            continue

        print(f"🚀 [Iteracja {iteration}] {train_start.date()}->{train_end.date()} | ", end="")

        # 2. OPTYMALIZACJA (In-Sample) - Zależna od systemu
        bt_train = Backtest(train_data, strategy_class, cash=KAPITAL_POCZATKOWY, commission=PROWIZJA, margin=0.01)
        
        # DEFINICJA SIATKI PARAMETRÓW
        # Możesz tu dać więcej parametrów dla Linuxa, bo jest szybszy!
        param_grid = {
            'rsi_delta_ltf': range(4, 15, 2),
            'rsi_delta_htf': range(10, 20, 5),
            'risk_reward': [2.0, 2.5, 3.0],
            'atr_multiplier': [1.0, 1.5, 2.5],
            'di_stdev_len': [21],
            'di_level_long': [50],
        }

        try:
            if IS_WINDOWS:
                # Ścieżka dla Windows (Safe Mode)
                best_params_obj = manual_optimization_windows(bt_train, param_grid)
                print("WinOpti OK | ", end="")
            else:
                # Ścieżka dla Linux (Turbo Mode - Multicore)
                stats_train = bt_train.optimize(
                    **param_grid,
                    maximize='Equity Final [$]',
                    verbose=False
                )
                best_params_obj = stats_train._strategy
                print("LinuxOpti OK | ", end="")

            # 3. TEST (Out-of-Sample)
            bt_test = Backtest(test_data, strategy_class, cash=KAPITAL_POCZATKOWY, commission=PROWIZJA, margin=0.01)
            
            # Wyciągamy parametry niezależnie od metody optymalizacji
            run_params = {
                'rsi_delta_ltf': best_params_obj.rsi_delta_ltf,
                'rsi_delta_htf': best_params_obj.rsi_delta_htf,
                'risk_reward': best_params_obj.risk_reward,
                'atr_multiplier': best_params_obj.atr_multiplier,
                'di_stdev_len': best_params_obj.di_stdev_len,
                'di_level_long': best_params_obj.di_level_long,
                'di_level_short': best_params_obj.di_level_long
            }
            
            stats_test = bt_test.run(**run_params)
            net_profit = stats_test['Equity Final [$]'] - KAPITAL_POCZATKOWY
            
            results_log.append({
                'Period Start': test_start.date(),
                'Period End': test_end.date(),
                'Net Profit': net_profit,
                'Trades': stats_test['# Trades'],
                'Params': f"RSI:{run_params['rsi_delta_ltf']} RR:{run_params['risk_reward']}"
            })
            
            print(f"✅ ZYSK: {net_profit:8.2f}$")
            
        except Exception as e:
            print(f"\n❌ BŁĄD: {e}")
            # Na Linuxie czasem warto wypisać traceback
            if not IS_WINDOWS:
                import traceback
                traceback.print_exc()

        current_date += pd.Timedelta(days=step_days)

    # 4. Podsumowanie
    print("\n" + "="*50)
    if not results_log:
        print("⚠️ Brak wyników.")
        return

    df_res = pd.DataFrame(results_log)
    total = df_res['Net Profit'].sum()
    print(f"SUMA ZYSKÓW WFA: {total:.2f} $")
    print(f"Średnia na miesiąc: {df_res['Net Profit'].mean():.2f} $")
    print("-" * 50)
    print(df_res)

if __name__ == '__main__':
    # Fix dla multiprocessing na Linux (czasem wymagany)
    if not IS_WINDOWS:
        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)

    found_path = find_data_file(NAZWA_PLIKU)
    if not found_path:
        print(f"❌ Nie znaleziono pliku: {NAZWA_PLIKU}")
        exit()
        
    print(f"📂 Wczytywanie: {found_path}")
    data = prepare_data_with_indicators(found_path, ltf_res=LTF, htf_res=HTF_RES)
    
    # Fix Timezone
    if data is not None and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    if data is not None and not data.empty:
        walk_forward_optimization(data, Strategy2xRSI_Dorsey, window_days=90, step_days=30)
