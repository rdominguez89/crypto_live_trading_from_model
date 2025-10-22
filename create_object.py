import re
import torch
device = 'cpu' #"cuda" if torch.cuda.is_available() else "cpu"
import joblib
import numpy as np

class def_coin:   
    def __init__(self):
        self.trade_real = False 
        self.training_type = input("Enter side s:short, l:long\n")
        self.path = './'
        if self.training_type == 'l8':
            self.model_path = 'best_model_15m_long_ratio_5_window_6_method_avg_high_low_mx_1_period_180_std_0_shift_0.7_balj_1211_pfj_1.33_bala_1196_pfa_1.56'
        if self.training_type == 'l9':
            self.model_path = 'best_model_15m_long_ratio_4_window_6_method_avg_high_low_mx_1_period_200_std_0_shift_0.6_balj_1201_pfj_1.40_bala_1125_pfa_1.47'
        if self.training_type == 'l10':
            self.model_path = 'best_model_15m_long_ratio_5_window_8_method_max_open_close_mx_1_period_50_std_0_shift_0.6_balj_1071_pfj_1.50_bala_1127_pfa_2.47'
        if self.training_type == 'l11':
            self.model_path = 'best_model_15m_long_ratio_5_window_4_method_max_open_close_mx_1_period_180_std_0_shift_0.7_balj_1077_pfj_1.77_bala_1115_pfa_2.89'
        if self.training_type == 'l12':
            self.model_path = 'best_model_30m_long_ratio_5_window_4_method_max_open_close_mx_1_period_30_std_0_shift_0.7_balj_1223_pfj_3.05_bala_1137_pfa_2.90'
           
        if self.training_type == 'l4':
            self.model_path = 'best_model_30m_long_ratio_2_window_4_method_avg_high_low_mx_2_period_70_std_0_shift_0.6_balj_1094_pfj_2.77_bala_1054_pfa_1.76'
        if self.training_type == 'l6':
            self.model_path = 'best_model_15m_long_ratio_2_window_10_method_avg_high_low_mx_1_period_90_std_0_shift_0.7_balj_1072_pfj_1.24_bala_1071_pfa_1.47'
        if self.training_type == 'l7':
            self.model_path = 'best_model_30m_long_ratio_4_window_4_method_avg_high_low_mx_1_period_120_std_0_shift_0.7_balj_1054_pfj_1.28_bala_1068_pfa_1.32'
        if self.training_type == 's':
            self.model_path = 'best_model_30m_short_ratio_2_window_8_method_avg_open_close_mx_1_period_30_std_0_shift_0.7_balj_1119_pfj_1.59_bala_1124_pfa_1.69'

        # Updated regex pattern to match the actual model_path format
        match = re.search(r'best_model_(\d+m)_(\w+)_ratio_(\d+)_window_(\d+)_method_(\w+)_(\w+)_(\w+)_mx_(\d+)_period_(\d+)_std_(\d+)_shift_([\d.]+)', self.model_path)
        if match:
            self.timeframe = match.group(1)  # 15m, 30m
            self.side = match.group(2).lower()  # long, short
            self.ratio = int(match.group(3))
            self.window = int(match.group(4))
            self.method = match.group(5)  # avg
            self.prices_col_1 = match.group(6)  # open
            self.prices_col_2 = match.group(7)  # close
            self.multiplier = int(match.group(8))
            self.period_lb = int(match.group(9))
            self.std_dev = int(match.group(10))
            self.shift_open = float(match.group(11))
            
            # These fields aren't in your current model_path, so set defaults
            self.coin = 'BTC'  # or whatever default you prefer
            self.pair = 'USDT'  # or whatever default you prefer
            self.use_day_month = 'false'  # default
            self.use_NY_trading_hour = 'false'  # default
            self.fract_ratio = 0.6
            self.range_low_limit = 130
            self.range_top_limit = 2000
        else:
            print("Model path does not match expected format.")
            exit(1)
        
        if self.coin == 'BTC' and self.pair == 'USDT':
            self.price_presition = 1
            self.pos_presition = 3
        else:
            print("Coin and pair not supported.")
        
        # self.n_candles_move_be = 150 #not used
        # self.ratio_move_be = 1 #not used
        self.pair_trading  = 'USDC'
        self.in_position = False
        self.waiting_for_fill = False
        self.model_1 = torch.load(f'{self.path}models/{self.model_path}.pt', weights_only=False, map_location=device ) 
        self.ct =  joblib.load(f'{self.path}models/ct_{self.model_path}.pkl')
        #extract features names from ct scaler
        self.columns = self.ct.feature_names_in_.tolist()
        self.client = ""
        self.ma_periods =  np.arange(5,201,5)
        self.periods_look_back = [10, 30, 50, 70, 90, 120, 140, 160, 180, 200]
        self.percentage = 0.01
        self.min_usdt_size = 10

        # Feature names
        self.df, self.tp, self.sl, self.op = None, None, None, None
        self.balance, self.n_win, self.n_loss = 1000.0, 0, 0
        self.size = 0
        self.filled = False
        self.first_candle = False
        print(f"Initialized coin: {self.coin}{self.pair}, trading pair: {self.pair_trading}, timeframe: {self.timeframe}, side: {self.side}, window: {self.window}, period_lb: {self.period_lb}, ratio: {self.ratio}, method: {self.method}, prices_col_1: {self.prices_col_1}, prices_col_2: {self.prices_col_2}, std_dev: {self.std_dev}, multiplier: {self.multiplier}, shift: {self.shift_open}")
