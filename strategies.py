from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- FUNKCJA POMOCNICZA (BEZ ZMIAN) ---
def get_dorsey_inertia(high, low, stdev_len, smooth_rv, smooth_di):
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    def rv_idi(src):
        stdev = src.rolling(window=int(stdev_len)).std()
        change = src.diff()
        up_mask = change >= 0
        up_source = pd.Series(0.0, index=src.index)
        down_source = pd.Series(0.0, index=src.index)
        up_source[up_mask] = stdev[up_mask]
        down_source[~up_mask] = stdev[~up_mask]
        up_sum = up_source.ewm(span=int(smooth_rv), adjust=False).mean()
        down_sum = down_source.ewm(span=int(smooth_rv), adjust=False).mean()
        denom = up_sum + down_sum
        rvi = 100 * up_sum / denom.replace(0, np.nan)
        return rvi.fillna(50)

    rvi_h = rv_idi(high_s)
    rvi_l = rv_idi(low_s)
    rv_avg = (rvi_h + rvi_l) / 2
    
    try:
        inertia = ta.linreg(rv_avg, length=int(smooth_di))
    except:
        inertia = rv_avg
        
    return inertia.fillna(50).to_numpy()


class Strategy2xRSI_Dorsey(Strategy):
    
    # --- PARAMETRY OPTYMALIZOWANE ---
    rsi_delta_ltf = 10
    rsi_delta_htf = 10
    
    # Parametry Dorsey
    di_stdev_len = 21
    di_smooth_rv = 14
    di_smooth_di = 14
    di_level_long = 50
    di_level_short = 50

    # Ryzyko
    atr_multiplier = 1.0
    risk_reward = 1.5
    
    # --- NOWY PARAMETR: FILTR ATR ---
    # 0.0005 oznacza 0.05% ceny. 
    # Dla złota (2600$) to 1.3$. Jeśli świeca (zmienność) jest mniejsza niż 1.3$, nie gramy.
    atr_min_percent = 0.0005 

    # Godziny
    session_start_hour = 8
    session_end_hour = 22
    close_all_hour = 22
    close_all_minute = 30

    def init(self):
        self.inertia = self.I(
            get_dorsey_inertia, 
            self.data.High, 
            self.data.Low, 
            self.di_stdev_len, 
            self.di_smooth_rv, 
            self.di_smooth_di
        )
        
    def next(self):
        # 0. ZAMYKANIE DNIA
        current_time = self.data.index[-1]
        if current_time.hour == self.close_all_hour and current_time.minute >= self.close_all_minute:
            if self.position: self.position.close()
            return

        # 1. SESJA
        if not (self.session_start_hour <= current_time.hour < self.session_end_hour):
            return
            
        # --- NOWOŚĆ: FILTR ATR (ANTI-CHOP) ---
        atr_val = self.data.ATR[-1]
        price = self.data.Close[-1]
        
        # Jeśli zmienność jest zbyt mała (rynek śpi), przerywamy funkcję (nie sprawdzamy dalej sygnałów)
        if atr_val < (price * self.atr_min_percent):
            return
        # -------------------------------------

        # 2. POBRANIE WARTOŚCI
        rsi_ltf = self.data.RSI_LTF[-1]
        rsi_htf = self.data.RSI_HTF[-1]
        inertia_val = self.inertia[-1]
        prev_rsi_ltf = self.data.RSI_LTF[-2]

        # 3. POZIOMY
        hr_up = 50 + self.rsi_delta_htf
        hr_dn = 50 - self.rsi_delta_htf
        lr_up = 50 + self.rsi_delta_ltf
        lr_dn = 50 - self.rsi_delta_ltf

        # 4. LOGIKA
        # Long
        cond_htf_long = rsi_htf > hr_up
        cond_ltf_long_cross = (prev_rsi_ltf < lr_dn) and (rsi_ltf >= lr_dn)
        cond_di_long = inertia_val > self.di_level_long 

        if cond_htf_long and cond_ltf_long_cross and cond_di_long:
            if not self.position:
                sl_dist = atr_val * self.atr_multiplier
                tp_dist = sl_dist * self.risk_reward
                self.buy(sl=price - sl_dist, tp=price + tp_dist, size=0.1)

        # Short
        cond_htf_short = rsi_htf < hr_dn
        cond_ltf_short_cross = (prev_rsi_ltf > lr_up) and (rsi_ltf <= lr_up)
        cond_di_short = inertia_val < self.di_level_short

        if cond_htf_short and cond_ltf_short_cross and cond_di_short:
            if not self.position:
                sl_dist = atr_val * self.atr_multiplier
                tp_dist = sl_dist * self.risk_reward
                self.sell(sl=price + sl_dist, tp=price - tp_dist, size=0.1)
