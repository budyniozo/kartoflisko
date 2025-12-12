import pandas as pd
import numpy as np
import pandas_ta as ta

# ==========================================
# 1. INTELIGENTNA SEKCJA ŁADOWANIA DANYCH
# ==========================================

def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Uniwersalny loader. Obsługuje:
    1. Pliki przetworzone/scalone (z nagłówkiem 'datetime', 'open'...)
    2. Surowe pliki Dukascopy (bez nagłówka, format GMT)
    """
    print(f"Wczytuję dane z {filepath}...")
    
    try:
        # KROK 1: Szybki podgląd pliku, aby wykryć format
        preview = pd.read_csv(filepath, nrows=1)
        
        # Sprawdzamy, czy plik ma nagłówek (czy kolumny nazywają się sensownie)
        # Nasz scalacz tworzy kolumnę 'datetime'
        is_processed = 'datetime' in preview.columns or 'date' in preview.columns or 'time' in preview.columns
        
        df = None
        
        if is_processed:
            print("   -> Wykryto format: PRZETWORZONY (Standard CSV)")
            df = pd.read_csv(filepath)
            
            # Znajdź kolumnę z datą
            date_col = None
            for col in ['datetime', 'date', 'Date_Time', 'time']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                # Parsujemy datę (Pandas sam zgadnie format ISO/Standard)
                df['Date_Time'] = pd.to_datetime(df[date_col])
                if date_col != 'Date_Time':
                    df.drop(columns=[date_col], inplace=True)
            else:
                raise ValueError("Nie znaleziono kolumny z datą w pliku z nagłówkiem.")

            # Standaryzacja nazw kolumn na Wielkie Litery (Open, High...)
            # Ponieważ reszta kodu oczekuje Open/High/Low/Close/Volume
            rename_map = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
            }
            df.rename(columns=rename_map, inplace=True)

        else:
            print("   -> Wykryto format: SUROWY (Dukascopy/MT5 bez nagłówka)")
            # Stara logika dla surowych plików
            column_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = pd.read_csv(filepath, header=None, names=column_names)
            
            # Specyficzne czyszczenie daty Dukascopy
            # Format: 13.01.2025 00:00:00.000 GMT+0100
            print("   -> Konwersja daty Dukascopy...")
            df['Date_Time'] = df['Date_Time'].astype(str).str.replace(r' GMT[+-]\d{4}', '', regex=True)
            df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d.%m.%Y %H:%M:%S.%f')

        # --- WSPÓLNA OBRÓBKA DANYCH ---
        df.set_index('Date_Time', inplace=True)
        
        # Konwersja na liczby (dla pewności)
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        
        # Filtry jakościowe
        initial_len = len(df)
        df = df[df['Volume'] > 0]
        df = df[df['High'] != df['Low']] # Usunięcie płaskich świec
        
        print(f"   -> Gotowe. Załadowano {len(df)} świec (odrzucono {initial_len - len(df)} pustych).")
        df.sort_index(inplace=True)
        return df

    except Exception as e:
        print(f"❌ KRYTYCZNY BŁĄD w data_loader: {e}")
        import traceback
        traceback.print_exc()
        return None

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    print(f"Resampling do: {timeframe}")
    # Mapowanie kolumn musi pasować do tego co wyszło z loadera (Open, High...)
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    try:
        df_res = df.resample(timeframe).agg(agg)
        df_res.dropna(inplace=True)
        df_res = df_res[df_res['Volume'] > 0]
        return df_res
    except Exception as e:
        print(f"BŁĄD resamplingu: {e}")
        return pd.DataFrame()

# ==========================================
# 2. LOGIKA WSKAŹNIKÓW (BEZ ZMIAN)
# ==========================================

def calculate_dorsey_inertia(df, stdev_len=21, smooth_rv=14, smooth_di=14):
    """
    Oblicza wskaźnik Dorsey Inertia na podstawie DataFrame.
    """
    # Funkcja pomocnicza RVI
    def rv_idi_original(src_series):
        stdev = src_series.rolling(window=stdev_len).std()
        change = src_series.diff()
        up_mask = change >= 0
        
        up_source = pd.Series(0.0, index=src_series.index)
        down_source = pd.Series(0.0, index=src_series.index)
        
        up_source[up_mask] = stdev[up_mask]
        down_source[~up_mask] = stdev[~up_mask]
        
        up_sum = up_source.ewm(span=smooth_rv, adjust=False).mean()
        down_sum = down_source.ewm(span=smooth_rv, adjust=False).mean()
        
        denom = up_sum + down_sum
        rvi = 100 * up_sum / denom.replace(0, np.nan)
        return rvi.fillna(50)

    # Obliczamy dla High i Low
    rvi_high = rv_idi_original(df['High'])
    rvi_low = rv_idi_original(df['Low'])
    rv_idi = (rvi_high + rvi_low) / 2
    
    # Regresja liniowa (Inertia)
    inertia = ta.linreg(rv_idi, length=smooth_di)
    return inertia

def prepare_data_with_indicators(filepath, ltf_res='15min', htf_res='4h'):
    """
    Główna funkcja wywoływana przez backtester.
    """
    df_raw = load_data_from_csv(filepath)
    if df_raw is None or df_raw.empty:
        return None
    
    # Resampling LTF
    df_ltf = resample_data(df_raw, ltf_res)
    if df_ltf.empty:
        return None
    
    print("Obliczam wskaźniki (RSI, Inertia, HTF)...")

    try:
        # RSI LTF i ATR
        df_ltf['RSI_LTF'] = ta.rsi(df_ltf['Close'], length=7)
        df_ltf['ATR'] = ta.atr(df_ltf['High'], df_ltf['Low'], df_ltf['Close'], length=5)

        # Dorsey Inertia
        df_ltf['Inertia'] = calculate_dorsey_inertia(df_ltf)

        # RSI HTF (z zabezpieczeniem shift)
        df_htf = df_raw.resample(htf_res).agg({'Close': 'last'}).dropna()
        df_htf['RSI_HTF_Calc'] = ta.rsi(df_htf['Close'], length=7)
        df_htf['RSI_HTF_Calc'] = df_htf['RSI_HTF_Calc'].shift(1) # Unikamy look-ahead bias
        
        # Merge
        merged_rsi = df_htf['RSI_HTF_Calc'].reindex(df_ltf.index, method='ffill')
        df_ltf['RSI_HTF'] = merged_rsi

        df_final = df_ltf.dropna()
        print(f"Gotowe. Świece po dodaniu wskaźników: {len(df_final)}")
        return df_final

    except Exception as e:
        print(f"BŁĄD podczas obliczania wskaźników: {e}")
        import traceback
        traceback.print_exc()
        return None
