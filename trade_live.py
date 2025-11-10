import asyncio
from calendar import c
import websockets
import json
from datetime import datetime
from binance_functions_future import *
import os
from create_object import def_coin
from binance.client import Client
from extra_functions import *
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*DataFrame is highly fragmented.*")

# Live trading loop
import asyncio
import json
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed

if trade_real:
    user = 'astro'
    api_key = os.environ.get(f'binance_api_futures')
    api_secret = os.environ.get(f'binance_secret_futures')
    if api_key==None:
        print("Binance keys not found")
        exit()

class WebSocketManager:
    def __init__(self, coin_info):
        self.coin_info = coin_info
        self.main_ws_task = None
        self.one_min_ws_task = None
        self.running = False
        
    async def handle_main_ws(self):
        while self.running:
            try:
                ws_url = f"wss://fstream.binance.com/ws/{self.coin_info.coin.lower()}{self.coin_info.pair.lower()}@kline_{self.coin_info.timeframe}"
                async with websockets.connect(ws_url) as ws:
                    print("Main WS connected")
                    async for message in ws:
                        if not self.running:
                            break
                        await self.process_main_message(message)
            except ConnectionClosed:
                print("Main WS connection closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Main WS error: {e}, reconnecting...")
                await asyncio.sleep(5)
    
    async def handle_one_min_ws(self):
        while self.running:
            try:
                ws_url = f"wss://fstream.binance.com/ws/{self.coin_info.coin.lower()}{self.coin_info.pair_trading.lower()}@kline_1m"
                async with websockets.connect(ws_url) as ws:
                    print("1m WS connected")
                    async for message in ws:
                        if not self.running:
                            break
                        if self.coin_info.in_position:
                            await self.process_one_min_message(message)
                        else:
                            # If no position, sleep briefly to reduce CPU usage
                            await asyncio.sleep(0.1)
            except ConnectionClosed:
                print("1m WS connection closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"1m WS error: {e}, reconnecting...")
                await asyncio.sleep(5)
    
    async def process_main_message(self, message):
        data = json.loads(message)
        kline = data['k']
        if kline['x']:
            new_candle = [
                kline['t'],
                float(kline['o']),
                float(kline['h']),
                float(kline['l']),
                float(kline['c']),
                float(kline['v'])
            ]
            update_candle(self.coin_info,new_candle)
            is_correct = check_candle_integrity(self.coin_info)
            if not is_correct: 
                fetch_initial_candles(self.coin_info)
                print(f'Gap candle detected, refetching candles at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            else:
                print(f'New candle received: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            calculate_features(self.coin_info)
            if self.coin_info.in_position:
                current_prediction_favour = open_position_logic(self.coin_info, False)
                if not current_prediction_favour:
                    print(f'Current prediction no longer favours position {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    candle = fetch_candles(self.coin_info.coin,self.coin_info.pair_trading,'1m',2)
                    if self.coin_info.waiting_for_fill and ((self.coin_info.side == 'long' and float(candle['low'].min())>self.coin_info.op) or (self.coin_info.side == 'short' and float(candle['high'].max())<self.coin_info.op)):
                        #cancel order
                        if trade_real:
                            self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.open_limit_order)
                            self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.stop_loss_order)
                        self.coin_info.in_position, self.coin_info.waiting_for_fill, self.coin_info.filled = False, False, False
                        msg = f"Position order cancelled due to change in prediction."
                        print(msg+' Returning to main WS for new signals.', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        if send_tel_messages: send_telegram_message_HTML(msg)
                elif current_prediction_favour and self.coin_info.waiting_for_fill:
                    if self.coin_info.side == 'short': new_open = round(new_candle[4] + self.coin_info.shift_open*self.coin_info.range_value_target, self.coin_info.price_presition)
                    if self.coin_info.side == 'long': new_open = round(new_candle[4] - self.coin_info.shift_open*self.coin_info.range_value_target, self.coin_info.price_presition)
                    if (self.coin_info.side == 'short' and new_open > self.coin_info.op) or (self.coin_info.side == 'long' and new_open < self.coin_info.op):
                        if trade_real:
                            self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.open_limit_order)
                            self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.stop_loss_order)
                        msg = f"Position order modified as it was not filled and prediction still hold."
                        print(msg+' Updationg entry.', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        if send_tel_messages: send_telegram_message_HTML(msg)
                        set_open_position(self.coin_info)
                        if trade_real:
                            if self.coin_info.side == 'long':
                                self.coin_info.stop_loss_order = post_stop_loss_order(self.coin_info, 'SELL')
                                self.coin_info.open_limit_order = post_limit_order(self.coin_info, self.coin_info.op, 'BUY')
                            else:
                                self.coin_info.stop_loss_order = post_stop_loss_order(self.coin_info, 'BUY')
                                self.coin_info.open_limit_order = post_limit_order(self.coin_info, self.coin_info.op, 'SELL')
                        self.coin_info.filled, self.coin_info.moved_to_be, self.coin_info.waiting_for_fill = False, False, True
                        print("Position order placed, 1m loop will handle tracking")
            elif not self.coin_info.in_position:
                open_position_logic(self.coin_info)
                if self.coin_info.in_position and not self.coin_info.waiting_for_fill:
                    set_open_position(self.coin_info)
                    if trade_real:
                        if self.coin_info.side == 'long':
                            self.coin_info.stop_loss_order = post_stop_loss_order(self.coin_info, 'SELL')
                            self.coin_info.open_limit_order = post_limit_order(self.coin_info, self.coin_info.op, 'BUY')
                        else:
                            self.coin_info.stop_loss_order = post_stop_loss_order(self.coin_info, 'BUY')
                            self.coin_info.open_limit_order = post_limit_order(self.coin_info, self.coin_info.op, 'SELL')
                    self.coin_info.filled, self.coin_info.moved_to_be, self.coin_info.waiting_for_fill = False, False, True
                    print("Position order placed, 1m loop will handle tracking")
                else:
                    print(f'No position opened at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    async def process_one_min_message(self, message):
        data = json.loads(message)
        kline = data['k']
        if kline['x']:
            new_candle = [
                kline['t'],
                float(kline['o']),
                float(kline['h']),
                float(kline['l']),
                float(kline['c']),
                float(kline['v'])
            ]
            if not self.coin_info.filled and kline['t'] > self.coin_info.df['timestamp'].iloc[-1]:
                if (self.coin_info.side == 'long' and new_candle[3] < self.coin_info.op) or (self.coin_info.side == 'short' and new_candle[2] > self.coin_info.op):
                    if trade_real:
                        status = check_fill_position(self.coin_info, self.coin_info.open_limit_order)
                    else:
                        status = 'FILLED'
                    if status == 'FILLED':
                        if (self.coin_info.side == 'long' and trade_real): self.coin_info.close_limit_order = post_limit_order(self.coin_info, self.coin_info.tp, 'SELL')
                        if (self.coin_info.side == 'short' and trade_real): self.coin_info.close_limit_order = post_limit_order(self.coin_info, self.coin_info.tp, 'BUY')
                        self.coin_info.filled = True
                        self.coin_info.waiting_for_fill = False
                        msg = f"Position filled {self.coin_info.training_type}"
                        print(msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        if send_tel_messages: send_telegram_message_HTML(msg)
                    elif status == 'UNKNOWN' or status == 'CANCELLED'or status == 'CANCELED':
                        if trade_real:
                            self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.open_limit_order) # add cancel sl
                            self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.stop_loss_order)
                        self.coin_info.in_position, self.coin_info.waiting_for_fill, self.coin_info.filled = False, False, False
                        msg = f"Position order not found (user modification?), returning to wait for new entry."
                        print(status,msg+' Returning to main WS for new signals.', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        if send_tel_messages: send_telegram_message_HTML(msg) # check
                if not self.coin_info.filled and ((self.coin_info.side == 'long' and new_candle[2] >= self.coin_info.tp) or (self.coin_info.side == 'short' and new_candle[3] <= self.coin_info.tp)):
                    #cancel order
                    if trade_real:
                        self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.open_limit_order) # add cancel sl
                        self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.stop_loss_order)
                    self.coin_info.in_position, self.coin_info.waiting_for_fill, self.coin_info.filled = False, False, False
                    msg = f"Position order cancelled due to reached tp before fill."
                    print(msg+' Returning to main WS for new signals.', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    if send_tel_messages: send_telegram_message_HTML(msg) # check
            if self.coin_info.filled:
                closed, win = check_position(self.coin_info, new_candle)
                if closed:
                    self.coin_info.in_position, self.coin_info.waiting_for_fill, self.coin_info.filled = False, False, False
                    if win:
                        if trade_real: 
                            self.coin_info.order_cancel = cancel_orders(self.coin_info, self.coin_info.stop_loss_order)
                            self.coin_info.balance = get_asset_balance(self.coin_info, self.coin_info.pair_trading)
                        else:
                            self.coin_info.balance += self.coin_info.size * (self.coin_info.range_value_target * self.coin_info.ratio)
                        self.coin_info.n_win += 1
                        msg = f'Position closed with profit. {self.coin_info.training_type}'
                        print(msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    else:
                        if trade_real: 
                            self.coin_info.balance = get_asset_balance(self.coin_info, self.coin_info.pair_trading)
                        else:
                            commision_sl = self.coin_info.size * self.coin_info.range_value_target * 0.0005
                            self.coin_info.balance -= (self.coin_info.size * self.coin_info.range_value_target + commision_sl)
                        if not self.coin_info.moved_to_be: 
                            self.coin_info.n_loss += 1
                            msg = f'Position closed with loss. {self.coin_info.training_type}'
                            print(msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        else: 
                            msg = f'Position closed at BE. {self.coin_info.training_type}'
                            self.coin_info.n_be += 1
                            print(msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    if send_tel_messages: send_telegram_message_HTML(msg)
                    print(f"Balance: {self.coin_info.balance:.1f}, Wins: {self.coin_info.n_win}, Losses: {self.coin_info.n_loss}, Break-even: {self.coin_info.n_be}")
                    print("Returning to main WS for new signals.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    if datetime.now().minute % 5 == 0:
                        print(f'Position still open at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    async def start(self):
        self.running = True
        self.main_ws_task = asyncio.create_task(self.handle_main_ws())
        self.one_min_ws_task = asyncio.create_task(self.handle_one_min_ws())
        
        try:
            # Keep the manager running until cancelled
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.stop()
    
    def stop(self):
        self.running = False
        if self.main_ws_task:
            self.main_ws_task.cancel()
        if self.one_min_ws_task:
            self.one_min_ws_task.cancel()

async def main():
    coin_info = def_coin()
    if trade_real:
        coin_info.client = Client(api_key, api_secret) # type: ignore
        coin_info.balance = get_asset_balance(coin_info, coin_info.pair_trading) # type: ignore
    else:
        coin_info.client = None
        coin_info.balance = 1000.0 + (coin_info.n_win*coin_info.ratio - coin_info.n_loss*1.1)*10
    print(f'Initial Balance: {coin_info.balance}')
    fetch_initial_candles(coin_info)
    #calculate_features(coin_info)
    ws_manager = WebSocketManager(coin_info)
    
    try:
        await ws_manager.start()
    except KeyboardInterrupt:
        print("Shutting down...")
        ws_manager.stop()

if __name__ == "__main__":
    asyncio.run(main())