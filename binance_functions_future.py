import os
from binance.client import Client
import numpy as np
import time
from datetime import datetime

def get_asset_balance(self, asset, max_retries=5):
    for attempt in range(max_retries):
        try:
            balances = self.client.futures_account_balance()
            for info in balances:
                if info['asset'] == asset:
                    return float(info['availableBalance'])
            return None
        except Exception as e:
            print(f"Error getting asset balance (attempt {attempt+1}): {e}")
            time.sleep(1)
    raise Exception("Max retries exceeded for get_asset_balance")

def post_trailing_order(coin_info, side_h, size, act_price, trailing, side, max_retries=5):
    for attempt in range(max_retries):
        try:
            limit_order_info = coin_info.client.futures_create_order(
                symbol=coin_info.name + coin_info.pair,
                type="TRAILING_STOP_MARKET",
                callbackRate=trailing,
                side=side_h,
                positionSide=side,
                quantity=size,
                activationPrice=act_price
            )
            return limit_order_info
        except Exception as e:
            print(f"Error placing trailing order (attempt {attempt+1}): {e}")
            if "Order would immediately trigger" in str(e):
                if side == "SHORT":
                    act_price = float(coin_info.presition_price.format(act_price * 1.005))
                if side == "LONG":
                    act_price = float(coin_info.presition_price.format(act_price * 0.995))
            time.sleep(1)
    raise Exception("Max retries exceeded for post_trailing_order")

def post_stop_loss_order(self, side_h, max_retries=10):
    for attempt in range(max_retries):
        try:
            stop_market_info = self.client.futures_create_order(
                symbol=self.coin + self.pair_trading,
                side=side_h,
                positionSide=self.side,
                type='STOP_MARKET',
                stopPrice=self.sl,
                closePosition=True,
                workingType='MARK_PRICE',
                priceProtect=True
            )
            return stop_market_info
        except Exception as e:
            print(f"Error placing stop loss order (attempt {attempt+1}): {e}")
            time.sleep(5 if attempt >= 5 else 1)
    raise Exception("Max retries exceeded for post_stop_loss_order")

def post_limit_order(self, price, side_h, max_retries=5):
    for attempt in range(max_retries):
        try:
            limit_order_info = self.client.futures_create_order(
                symbol=self.coin + self.pair_trading,
                type='LIMIT',
                timeInForce='GTC',
                positionSide=self.side,
                price=price,
                side=side_h,
                quantity=self.size
            )
            return limit_order_info
        except Exception as e:
            print(f"Error placing limit order (attempt {attempt+1}): {e}")
            time.sleep(1)
    raise Exception("Max retries exceeded for post_limit_order")

def check_fill_position(self, order, max_retries=5):
    for attempt in range(max_retries):
        try:
            info = self.client.futures_get_order(
                symbol=self.coin + self.pair_trading,
                orderId=order['orderId'],
                requests_params={'timeout': 15}
            )
            return info['status']
        except Exception as e:
            print(f"Error checking fill position (attempt {attempt+1}): {e}")
            time.sleep(5)
    return "UNKNOWN"

def cancel_orders(self, order, max_retries=5):
    for attempt in range(max_retries):
        try:
            cancel_order_info = self.client.futures_cancel_order(
                symbol=self.coin + self.pair_trading,
                orderId=order['orderId']
            )
            return cancel_order_info
        except Exception as e:
            print(f"Error cancelling order (attempt {attempt+1}): {e}")
            if hasattr(e, 'code') and e.code == -2011: # type: ignore
                print("Order not found", e)
                return []
            time.sleep(1)
    return []

def get_current_size(coin_info,n_try):
    futures = get_info_position_2(coin_info,n_try)
    open_pos = []
    for info in futures:
        if info['symbol'] == coin_info.name+coin_info.pair and info['positionSide'] == coin_info.side:
            open_pos = info
            break
    size = 0
    if len(open_pos)!=0 and abs(float(open_pos['positionAmt']))!=0:size = round(abs(float(open_pos['positionAmt'])),coin_info.pos_presition) # type: ignore
    return size

def get_open_orders_info(coin_info,n_try):
    coin_info.waiting_trade = True
    price_be = 0
    futures = get_info_position_2(coin_info,n_try)
    open_pos = []
    for info in futures:
        if info['symbol'] == coin_info.name+coin_info.pair and info['positionSide'] == coin_info.side:
            open_pos = info
            break
    if len(open_pos)!=0 and abs(float(open_pos['positionAmt']))!=0: # type: ignore
        open_price = float(open_pos['entryPrice']) # type: ignore
        size = abs(float(open_pos['positionAmt'])) # type: ignore
        time_post_order = open_pos['updateTime']-float(datetime.fromtimestamp(open_pos['updateTime']/1000).strftime("%M"))*60000-float(datetime.fromtimestamp(open_pos['updateTime']/1000).strftime("%S"))*1000 # type: ignore
        orders = get_open_orders(coin_info,n_try)
        if len(orders)!=0:
            print(f'{coin_info.side+" "+coin_info.name+coin_info.pair}: Recovering orders')
            if coin_info.side == 'SHORT':side_h = 'BUY'
            if coin_info.side == 'LONG':side_h = 'SELL'
            n_orders_found = 0
            for order in orders:
                if order['side'] == side_h and order['positionSide'] == coin_info.side and order['type'] == "STOP_MARKET":
                    stop_loss_order = order
                    price_stop = round(float(order['stopPrice']),coin_info.price_presition)
                    n_orders_found += 1
                if order['side'] == side_h and order['positionSide'] == coin_info.side and order['type'] == "LIMIT":
                    limit_close_order = order
                    price_tp = round(float(order['price']),coin_info.price_presition)
                    n_orders_found += 1
        else:
            return 0,0,0,0,0,0,False,False,0,0,0,coin_info
        if n_orders_found!=2:
            return 0,0,0,0,0,0,False,False,0,0,0,coin_info
    else:
        return 0,0,0,0,0,0,False,False,0,0,0,coin_info
    coin_info.waiting_trade = False
    atr = abs(open_price-price_stop)
    if coin_info.move_be:
        if coin_info.side == 'LONG':    
            price_be = round(open_price+coin_info.ratio_move_be*coin_info.ratio*atr+0.5,coin_info.price_presition)
        else:
            price_be = round(open_price-coin_info.ratio_move_be*coin_info.ratio*atr-0.5,coin_info.price_presition)
    return open_price,stop_loss_order,price_stop,limit_close_order,price_tp,size,False,False,time_post_order,price_be,atr,coin_info

def check_open_orders(coin_info,side):
    [waiting_pivot,waiting_ema,waiting_fill,waiting_close] = [True,False,False,False]
    orders = get_open_orders(coin_info,0)
    [buy_price,take_profit,stop_loss,position] = [0,0,0,False]
    for order in orders:
        if order['positionSide'] == side:
            position = True
            if(order['type']=='LIMIT' and order['side']=='SELL'):[waiting_pivot,waiting_fill,buy_price] = [False,True,float(order['price'])]
            if(order['type']=='LIMIT' and order['side']=='BUY'):[waiting_close,take_profit] = [True,float(order['price'])]
            if(order['type']=='STOP_LOSS' and order['side']=='BUY'):[waiting_close,stop_loss] = [True,float(order['price'])]
    if position:
        if(stop_loss!=0 and buy_price==0 and take_profit==0):
            [waiting_pivot,waiting_ema,waiting_fill,waiting_close] = [True,False,False,False]
        elif (stop_loss==0 or buy_price==0 or take_profit==0):
            if stop_loss == 0:
                stop_loss = float(coin_info.presition_price.format(float(input("Stop Loss price:"))))
            if buy_price == 0:
                buy_price = float(coin_info.presition_price.format(float(input("Buy price:"))))
            if take_profit == 0:
                take_profit = float(coin_info.presition_price.format(float(input("Take Profit price:"))))
            size = float(coin_info.presition_pos.format(float(input("Position size:"))))
        if(take_profit!=0):waiting_pivot=False
    return waiting_pivot,waiting_ema,waiting_fill,waiting_close,buy_price,take_profit,stop_loss,size

def check_open_orders_2(coin_info,side):
    orders = get_open_orders(coin_info,0)
    for order in orders:
        if order['positionSide'] == side:
            if(order['type']=='LIMIT' and order['positionSide']==side):
                if (side == 'LONG' and order['side']=='SELL') or (side == 'SHORT' and order['side']=='BUY'): 
                    take_profit = float(order['price'])
                    size_tp = float(order['origQty'])                    
            if(order['type']=='STOP_MARKET' and order['positionSide']==side):
                stop_loss = float(order['stopPrice'])
                coin_info.pos_stop_loss = order
    return coin_info,take_profit,size_tp,stop_loss

def get_info_position(coin_info,side):
    try:
        futures = coin_info.client.futures_position_information()
    except Exception as e:
        if(str(e)!="HTTPSConnectionPool(host='fapi.binance.com', port=443): Read timed out. (read timeout=10)"):print(f"Getting future info again {e}")
        time.sleep(5)
        coin_info = get_info_position(coin_info,side)
        return coin_info
    coin_info.size=0
    coin_info.entry = 0
    for future in futures:
        if (future["symbol"] == coin_info.name+coin_info.pair and future["positionAmt"] != "0" and float(future["unRealizedProfit"]) != 0.00000000 and future['positionSide']==side):
            coin_info.size = float(future['positionAmt'])
            coin_info.entry = float(future['entryPrice'])
            coin_info.price_now = float(future['markPrice'])
            coin_info.profit = float(future['unRealizedProfit'])
            coin_info.liquidation = float(future['liquidationPrice'])
            if(side=='SHORT'):coin_info.size = -coin_info.size
            break
    return coin_info

def get_info_position_2(coin_info, max_retries=5):
    for attempt in range(max_retries):
        try:
            futures = coin_info.client.futures_position_information()
            return futures
        except Exception as e:
            print(f"Error getting position info (attempt {attempt+1}): {e}")
            time.sleep(5)
    return []

def get_open_orders(coin_info, max_retries=5):
    for attempt in range(max_retries):
        try:
            orders = coin_info.client.futures_get_open_orders(symbol=coin_info.name + coin_info.pair)
            return orders
        except Exception as e:
            print(f"Error getting open orders (attempt {attempt+1}): {e}")
            time.sleep(1)
    return []


