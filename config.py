# config.py

# --- ŚCIEŻKI ---
CSV_PATH = r"xauusd11M_dukas_ohlcv.csv"

# --- BACKTEST ---
LTF = '2min'
HTF = '30min'
PROWIZJA = 0.000008
CASH = 100000

# --- DORSEY INERTIA (Konstrukcyjne) ---
DI_STDEV_LEN = 21
DI_SMOOTH_RV = 14
DI_SMOOTH_DI = 14

# --- POZIOMY SYGNAŁÓW ---
DI_LEVEL_LONG = 50
DI_LEVEL_SHORT = 50

# --- RSI ---
RSI_LEN_DEFAULT = 14    # <--- NOWOŚĆ: Domyślna długość
RSI_DELTA_LTF = 10
RSI_DELTA_HTF = 10

# --- RYZYKO ---
ATR_MULTIPLIER = 3.0
RISK_REWARD = 1.0