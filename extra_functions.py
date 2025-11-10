import requests
import pandas as pd
from binance_functions_future import cancel_orders, post_stop_loss_order
import numpy as np
from datetime import datetime
import torch
device = 'cpu' #"cuda" if torch.cuda.is_available() else "cpu"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
from numba import njit
import os
import time

send_tel_messages = True
trade_real = False  # Set to False for testing without real trades

if send_tel_messages:
    TOKEN = os.environ.get(f'bot_AI_model')
    chat_id = os.environ.get(f'chat_id_AI_model')
    def send_telegram_message_HTML(text, chat_id=chat_id, token=TOKEN):
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        response = requests.post(url, data=payload, timeout=1)
        return

def check_candle_integrity(self):
    """Check that candles in self.df have uniform timestamp spacing (no gaps).

    Returns:
        bool: True if spacing is consistent (within a small tolerance), False if gaps/missing candles detected.
    """
    # Basic validations
    if not hasattr(self, 'df') or self.df is None:
        return False
    if 'timestamp' not in self.df.columns:
        return False
    if len(self.df) < 2:
        # Nothing to check, consider it not valid
        return False

    # Ensure timestamps are integers (ms) and sorted
    try:
        timestamps = self.df['timestamp'].astype('int64').values
    except Exception:
        return False

    if not (timestamps[1:] >= timestamps[:-1]).all():
        # Not monotonically non-decreasing -> something off
        return False

    # Compute deltas in milliseconds
    deltas = np.diff(timestamps)

    if len(deltas) == 0:
        return False

    # Most common delta (mode) is probably the expected timeframe delta
    # Use median to be robust to outliers
    expected_delta = int(np.median(deltas))

    # Allow small tolerance (1 second = 1000 ms) for minor timing noise
    tol = 10

    # If expected_delta is zero or negative, fail
    if expected_delta <= 0:
        return False

    # Check how many deltas deviate from expected beyond tolerance
    bad = np.sum(np.abs(deltas - expected_delta) > tol)

    # If any bad gaps present, return False
    return bad == 0

# NY trading hours: 9:30 to 16:00, Monday to Friday
def is_ny_trading_hour(dt):
    # dt is timezone-aware in NY
    if dt.weekday() >= 5:
        return 0
    hour = dt.hour
    minute = dt.minute
    # Trading starts at 9:30, ends at 16:00 (not inclusive)
    if (hour > 9 and hour < 16):
        return 1
    if hour == 9 and minute >= 30:
        return 1
    if hour == 16 and minute == 0:
        return 0
    return 0

def fetch_initial_candles(self):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": self.coin.capitalize()+self.pair.capitalize(),
        "interval": self.timeframe,
        "limit": 1000
    }
    response = requests.get(url, params=params)
    ohlcv = response.json()
    df = pd.DataFrame(ohlcv, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    self.df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].iloc[:-1]  # Exclude the last incomplete candle
    self.df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    self.df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return

def fetch_candles(coin,pair,timeframe,n):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": coin.capitalize()+pair.capitalize(),
        "interval": timeframe,
        "limit": n
    }
    response = requests.get(url, params=params)
    ohlcv = response.json()
    df_h = pd.DataFrame(ohlcv, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    return df_h

def update_candle(self,new_candle):
    # Just prepare the new_row, do NOT add it to df
    new_row = pd.DataFrame([new_candle], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    new_row['date'] = pd.to_datetime(new_row['timestamp'], unit='ms')    
    self.df = pd.concat([self.df, new_row], ignore_index=True)
    return new_row

def open_position_logic(self,normal_check=True):
    self.range_value_target = self.df[f'{self.prices_col_1}_{self.prices_col_2}_{self.method}_range_{self.period_lb}'].iloc[-1]
    if self.range_value_target < self.range_low_limit or self.range_value_target > self.range_top_limit:
        if not normal_check: return True
        self.in_position = False
        print(f'Range value {self.range_value_target:.2f} out of bounds ({self.range_low_limit}-{self.range_top_limit}), skipping prediction at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        return

    features = self.df[self.columns].iloc[[-1]].copy()
    features_scaled = self.ct.transform(features)
    features_t = torch.tensor(features_scaled, dtype=torch.float32, device = device)
    self.model_1.eval()
    prediction_1 = (self.model_1(features_t).cpu() > 0.5).float()   
    if not normal_check:
        if prediction_1 == 1:
            return True
        else:
            return False
    
    if prediction_1 == 1:
        self.in_position = True
        return
    self.in_position = False
    return

def check_position(self, candle):
    current_high = candle[2]
    current_low = candle[3]
    if (self.side == 'long' and (current_high >= self.tp or current_low <= self.sl)) or (self.side == 'short' and (current_low <= self.tp or current_high >= self.sl)):    
        if self.side == 'long': return True, current_high >= self.tp
        if self.side == 'short': return True, current_low <= self.tp
    if not self.moved_to_be and ((self.side == 'long' and current_high >= self.be) or (self.side == 'short' and current_low <= self.be)):    
        self.moved_to_be = True
        if trade_real: self.order_cancel = cancel_orders(self, self.stop_loss_order)
        if self.side == 'short': 
            self.sl = round(self.op, self.price_presition)
            if trade_real: self.stop_loss_order = post_stop_loss_order(self, 'BUY')
        if self.side == 'long': 
            self.sl = round(self.op, self.price_presition)
            if trade_real: self.stop_loss_order = post_stop_loss_order(self, 'SELL')
        msg = f'Moved to BE {self.training_type}, SL adjusted to {self.sl}.'
        print(msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if send_tel_messages: send_telegram_message_HTML(msg)
    return False, None

def set_open_position(self): 
    candle = fetch_candles(self.coin,self.pair_trading,'1m',1)
    if self.side == 'short':
        open_price = float(candle['high'].iloc[0])
        self.op = round(open_price + self.shift_open*self.range_value_target, self.price_presition)
        self.tp = round(self.op - self.ratio * self.range_value_target, self.price_presition) 
        self.sl = round(self.op + self.range_value_target, self.price_presition)
        self.be = round(self.op - self.fract_ratio*self.ratio * self.range_value_target, self.price_presition)
    else:  # Assuming 'long' side
        open_price = float(candle['low'].iloc[0])
        self.op = round(open_price - self.shift_open*self.range_value_target, self.price_presition)
        self.tp = round(self.op + self.ratio * self.range_value_target, self.price_presition)
        self.sl = round(self.op - self.range_value_target, self.price_presition)
        self.be = round(self.op + self.fract_ratio*self.ratio * self.range_value_target, self.price_presition)
    self.size = max([round((10 / self.range_value_target), self.pos_presition),round((self.percentage * self.balance / self.range_value_target), self.pos_presition)])  
    print(f'Placing limit order {self.side}: Open : {self.op} TP : {self.tp} SL : {self.sl} BE : {self.be} Ratio : {self.ratio}, Size : {self.size}, Range Value : {self.range_value_target:.2f}. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    msg = f'Placing limit order {self.training_type} {self.side}\nOpen : {self.op}\nTP : {self.tp}\nSL : {self.sl}\nSize : {self.size}'
    if send_tel_messages: send_telegram_message_HTML(msg)

def add_ny_time_features(df, features_columns, use_NY_trading_hour, use_day_month):
    """Add New York time, trading hour, day, and month features."""
    if use_NY_trading_hour or use_day_month:
        df['date_ny'] = df['date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        def is_ny_trading_hour(dt):
            if dt.weekday() >= 5:
                return 0
            hour = dt.hour
            minute = dt.minute
            if (hour > 9 and hour < 16):
                return 1
            if hour == 9 and minute >= 30:
                return 1
            if hour == 16 and minute == 0:
                return 0
            return 0
        if use_NY_trading_hour:
            df['ny_trading_hour'] = df['date_ny'].apply(is_ny_trading_hour)
            features_columns.append('ny_trading_hour')
    if use_day_month == 'day' or use_day_month == 'day_month':
        df['day'] = df['date_ny'].dt.weekday
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 6)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 6)
        features_columns += ['day_sin', 'day_cos']
    if use_day_month == 'month' or use_day_month == 'day_month':
        df['month'] = df['date_ny'].dt.month - 1
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 11)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 11)
        features_columns += ['month_sin', 'month_cos']
    return df, features_columns

def get_columns_ma(ma_periods):
    return [f'above_MA_{period_ma}' for period_ma in ma_periods]

def get_columns_green():
    return ['green']

def get_columns_vwap(ma_periods):
    return [f'above_vwap_{period_ma}' for period_ma in ma_periods]

def get_columns_range(periods_look_back):
    columns_range = []
    for col_name in ['high_low', 'open_close', 'volume']:
        for period_look_back in periods_look_back:
            columns_range.extend([f'{col_name}_{stat}_range_{period_look_back}' for stat in ['avg', 'max', 'min', 'std']])
    return columns_range

def add_moving_averages(df, features_columns, ma_periods):
    """Add moving average and above_MA features to df."""
    for period_ma in ma_periods:
        df[f'MA_{period_ma}'] = df['close'].rolling(window=period_ma, min_periods=period_ma).mean()
        df[f'above_MA_{period_ma}'] = (df['close'] > df[f'MA_{period_ma}']).astype(int)
        #features_columns.append(f'above_MA_{period_ma}')
    return df, features_columns

def add_vwap_features(df, features_columns, ma_periods):
    """Add VWAP and above_vwap features to df."""
    for period_ma in ma_periods:
        df[f'vwap_{period_ma}'] = (df['volume'] * df['close']).rolling(window=period_ma, min_periods=period_ma).sum() / df['volume'].rolling(window=period_ma, min_periods=period_ma).sum()
        df[f'above_vwap_{period_ma}'] = (df['close'] > df[f'vwap_{period_ma}']).astype(int)
        #features_columns.append(f'above_vwap_{period_ma}')
    return df, features_columns

def add_range_features(df, features_columns, periods_look_back):
    """Add range-based features (avg, max, min, std) for high-low, open-close, volume."""
    for col, col_name in zip(['diff_high_low', 'diff_open_close', 'volume'], ['high_low', 'open_close', 'volume']):
        for period_look_back in periods_look_back:
            df[f'{col_name}_avg_range_{period_look_back}'] = df[col].rolling(window=period_look_back).mean()
            df[f'{col_name}_max_range_{period_look_back}'] = df[col].rolling(window=period_look_back).max()
            df[f'{col_name}_min_range_{period_look_back}'] = df[col].rolling(window=period_look_back).min()
            df[f'{col_name}_std_range_{period_look_back}'] = df[col].rolling(window=period_look_back).std()
            features_columns += [f'{col_name}_avg_range_{period_look_back}', f'{col_name}_max_range_{period_look_back}', f'{col_name}_min_range_{period_look_back}', f'{col_name}_std_range_{period_look_back}']
    return df, features_columns

def add_diff_features(df, features_columns, method='abs'):
    """Add diff_high_low and diff_open_close columns."""
    df['diff_high_low'] = (df['high'] - df['low']).abs()
    df['diff_open_close'] = (df['open'] - df['close']).abs()
    if method == 'raw':
        df['green'] = (df['close'] > df['open']).astype(int)
        #features_columns += ['green']
    #features_columns += ['diff_high_low', 'diff_open_close']
    return df, features_columns

def add_advanced_features(df, features_columns, ma_periods):
    """Add distance to MA/VWAP and slope features."""
    for period_ma in ma_periods:
        df[f'dist_to_MA_{period_ma}'] = df['close'] - df[f'MA_{period_ma}']
        df[f'slope_MA_{period_ma}'] = df[f'MA_{period_ma}'].diff()
        df[f'dist_to_vwap_{period_ma}'] = df['close'] - df[f'vwap_{period_ma}']
        df[f'slope_vwap_{period_ma}'] = df[f'vwap_{period_ma}'].diff()
        features_columns += [f'dist_to_MA_{period_ma}', f'slope_MA_{period_ma}', f'dist_to_vwap_{period_ma}', f'slope_vwap_{period_ma}']
    return df, features_columns

def add_atr_rolling_std(df, features_columns, window=14):
    """Add ATR and rolling std features."""
    df['atr_14'] = (df['high'] - df['low']).rolling(window=window).mean()
    df['rolling_std_14'] = df['close'].rolling(window=window).std()
    features_columns += ['atr_14', 'rolling_std_14']
    return df, features_columns

def add_normalized_features(df, features_columns, ma_periods):
    """Add normalized distance and slope features."""
    for period_ma in ma_periods:
        df[f'norm_dist_to_MA_{period_ma}'] = df[f'dist_to_MA_{period_ma}'] / df['atr_14']
        df[f'norm_dist_to_vwap_{period_ma}'] = df[f'dist_to_vwap_{period_ma}'] / df['atr_14']
        df[f'norm_slope_MA_{period_ma}'] = df[f'slope_MA_{period_ma}'] / df['rolling_std_14']
        df[f'norm_slope_vwap_{period_ma}'] = df[f'slope_vwap_{period_ma}'] / df['rolling_std_14']
        features_columns += [f'norm_dist_to_MA_{period_ma}', f'norm_dist_to_vwap_{period_ma}', f'norm_slope_MA_{period_ma}', f'norm_slope_vwap_{period_ma}']
    return df, features_columns

def calc_adx(df, n=14):
    """Calculate ADX indicator."""
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame({'h-l': high - low, 'h-c': (high - close.shift()), 'l-c': (low - close.shift())})
    tr = tr1.abs().max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.abs().rolling(n).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(n).mean()
    return adx

def add_trend_volatility_features(df, features_columns):
    """Add ADX, realized volatility, and volatility percentile features."""
    df['adx_14'] = calc_adx(df)
    df['realized_vol_14'] = np.log(df['close']).diff().rolling(window=14).std() * np.sqrt(252)
    df['vol_percentile_14'] = df['realized_vol_14'].rank(pct=True)
    features_columns += ['adx_14', 'realized_vol_14', 'vol_percentile_14']
    return df, features_columns

def add_volume_features(df, features_columns):
    """Add volume ratio and rvol features."""
    df['volume_ratio_14'] = df['volume'] / df['volume'].rolling(window=14).mean()
    df['rvol_14'] = (df['volume'] - df['volume'].rolling(window=14).mean()) / df['volume'].rolling(window=14).std()
    features_columns += ['volume_ratio_14', 'rvol_14']
    return df, features_columns

def add_candle_shape_features(df, features_columns):
    """Add body size, wick, and related ratio features."""
    df['body_size'] = (df['close'] - df['open']).abs()
    df['body_size_ratio'] = df['body_size'] / (df['high'] - df['low']).replace(0, np.nan)
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['upper_wick_ratio'] = df['upper_wick'] / (df['high'] - df['low']).replace(0, np.nan)
    df['lower_wick_ratio'] = df['lower_wick'] / (df['high'] - df['low']).replace(0, np.nan)
    features_columns += ['body_size', 'body_size_ratio', 'upper_wick', 'lower_wick', 'upper_wick_ratio', 'lower_wick_ratio']
    return df, features_columns

def add_event_features(df, features_columns, ma_period):
    """Add new high/low, time since, bullish candle, and run length features."""
    df['new_high'] = (df['high'] == df['high'].cummax()).astype(int)
    df['new_low'] = (df['low'] == df['low'].cummin()).astype(int)
    df['time_since_new_high'] = (~df['new_high'].astype(bool)).groupby(df['new_high'].cumsum()).cumcount()
    df['time_since_new_low'] = (~df['new_low'].astype(bool)).groupby(df['new_low'].cumsum()).cumcount()
    df['bullish_candle'] = (df['close'] > df['open']).astype(int)
    df['bullish_count_14'] = df['bullish_candle'].rolling(window=14).sum()
    df['run_length_above_MA'] = df[f'above_MA_{ma_period}'].groupby((df[f'above_MA_{ma_period}'] != df[f'above_MA_{ma_period}'].shift()).cumsum()).cumcount() + 1
    df['run_length_below_MA'] = df['run_length_above_MA'].where(df[f'above_MA_{ma_period}'] == 0, 0)
    features_columns += ['new_high', 'new_low', 'time_since_new_high', 'time_since_new_low', 'bullish_candle', 'bullish_count_14', 'run_length_above_MA', 'run_length_below_MA']
    return df, features_columns

def do_ma_feature_engineering(df, ma_periods, periods_look_back, use_NY_trading_hour=False, use_day_month=None):
    """Main function to add all features and return columns lists."""
    features_columns = []
    df, features_columns = add_ny_time_features(df, features_columns, use_NY_trading_hour, use_day_month)
    df, features_columns = add_moving_averages(df, features_columns, ma_periods)
    df, features_columns = add_vwap_features(df, features_columns, ma_periods)
    df, features_columns = add_diff_features(df, features_columns, method='raw') 
    df, features_columns = add_range_features(df, features_columns, periods_look_back)

    df, features_columns = add_advanced_features(df, features_columns, ma_periods)
    df, features_columns = add_atr_rolling_std(df, features_columns)
    df, features_columns = add_normalized_features(df, features_columns, ma_periods)
    df, features_columns = add_trend_volatility_features(df, features_columns)
    df, features_columns = add_volume_features(df, features_columns)
    df, features_columns = add_candle_shape_features(df, features_columns)
    df, features_columns = add_event_features(df, features_columns, ma_periods[0])
    return df, features_columns

@njit
def build_features(data, window_size, i_start):
    n, f = data.shape
    features = np.empty((n, window_size * f), dtype=np.float64)
    for i in range(i_start, n):
        # Take a window of shape (window_size, f), flatten to (window_size*f,)
        features[i] = data[i - window_size+1:i+1, :].flatten()
    return features

def do_reshape_window(df, window_size, ma_periods, columns_features, columns_for_windows):
    for feature in columns_for_windows:
        if feature == 'ma': cols = get_columns_ma(ma_periods)
        if feature == 'vwap': cols = get_columns_vwap(ma_periods)
        if feature == 'green': cols = get_columns_green()
        data = df[cols].values.astype(int)
        features = build_features(data, window_size, ma_periods[-1] + window_size - 1)
        feature_names = []
        for t in reversed(range(window_size)):
            for col in cols:
                feature_names.append(f"{feature}_{col}_t{t}")
        df[feature_names] = features
        columns_features.extend(feature_names)
    
    return df

# Function to calculate features
def calculate_features(self):
    self.df, self.features_columns = do_ma_feature_engineering(self.df, self.ma_periods, self.periods_look_back, self.use_NY_trading_hour, self.use_day_month)
    self.df = do_reshape_window(self.df, self.window, self.ma_periods, self.features_columns, ['ma', 'vwap', 'green'])
    return
