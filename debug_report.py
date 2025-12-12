from backtesting import Backtest
import pandas as pd
import os
import sys

# --- IMPORT TWOICH MODUŁÓW ---
try:
    from strategies import Strategy2xRSI_Dorsey
    from data_loader import prepare_data_with_indicators
    print("✅ Moduły strategies i data_loader załadowane poprawnie.")
except ImportError as e:
    print(f"❌ BŁĄD IMPORTU: {e}")
    sys.exit(1)

# --- KONFIGURACJA ---
FILE_PATH = "xauusd_FULL_2024_2025.csv"
LTF = '2min'
HTF_RES = '1h'

# Parametry (Twoje z logów)
PARAMS = {
    'rsi_delta_ltf': 8,
    'rsi_delta_htf': 15,
    'risk_reward': 2.5,
    'atr_multiplier': 1.5,
    'di_stdev_len': 21,
    'di_level_long': 50
}

def run_debug():
    print("\n--- DIAGNOSTYKA ROZPOCZĘTA ---")
    
    # 1. Sprawdzenie pliku
    if not os.path.exists(FILE_PATH):
        print(f"❌ BŁĄD: Nie widzę pliku '{FILE_PATH}' w katalogu {os.getcwd()}")
        return
    print(f"✅ Plik danych istnieje: {FILE_PATH}")

    # 2. Ładowanie danych
    print("⏳ Wczytuję i przeliczam dane (może chwilę potrwać)...")
    try:
        data = prepare_data_with_indicators(FILE_PATH, ltf_res=LTF, htf_res=HTF_RES)
    except Exception as e:
        print(f"❌ WYJĄTEK w prepare_data_with_indicators: {e}")
        return

    if data is None or data.empty:
        print("❌ BŁĄD: Loader zwrócił puste dane (None lub empty DataFrame).")
        return
    
    # Naprawa stref czasowych (na wszelki wypadek)
    if data.index.tz is not None:
        print("⚠️ Wykryto strefę czasową w indeksie. Usuwam (tz_localize(None))...")
        data.index = data.index.tz_localize(None)

    print(f"✅ Dane wczytane. Rekordów: {len(data)}")
    print(f"   Zakres dostępny: {data.index.min()} -> {data.index.max()}")

    # 3. Wycinanie okresu
    start_date = "2024-01-01"
    end_date = "2025-11-20"
    print(f"\n✂️ Próba wycięcia okresu: {start_date} do {end_date}")
    
    subset = data.loc[start_date:end_date]
    
    if subset.empty:
        print("❌ BŁĄD: Podzbiór danych (subset) jest PUSTY!")
        print("   Sprawdź czy podany zakres dat mieści się w zakresie dostępnym powyżej.")
        return
    
    print(f"✅ Wycięto podzbiór. Liczba świec do testu: {len(subset)}")

    # 4. Backtest
    print("\n🚀 Uruchamiam Backtest...")
    try:
        # Zwiększamy cash do 100k, żeby uniknąć Margin Call i zobaczyć cały wykres
        bt = Backtest(subset, Strategy2xRSI_Dorsey, cash=100000, commission=0.000008, margin=1/100)
        stats = bt.run(**PARAMS)
        print("✅ Backtest zakończony sukcesem.")
        print(f"   Wynik (Equity Final): {stats['Equity Final [$]']:.2f}")
        print(f"   Liczba transakcji: {stats['# Trades']}")
    except Exception as e:
        print(f"❌ BŁĄD w bt.run(): {e}")
        # Częsty błąd: brak parametru w klasie strategii
        return

    # 5. Generowanie HTML
    output_file = "Debug_Luty2025.html"
    print(f"\n💾 Generuję plik HTML: {output_file}")
    try:
        bt.plot(filename=output_file, open_browser=False)
        print(f"✅ SUKCES! Plik {output_file} został utworzony.")
        print("   Otwórz go ręcznie w przeglądarce.")
    except Exception as e:
        print(f"❌ BŁĄD generowania wykresu: {e}")

if __name__ == "__main__":
    run_debug()
