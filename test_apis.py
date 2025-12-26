from binance.client import Client
from extra_functions import *
from datetime import datetime
import os 
import time

def post_limit_order_test(client, coin, pair_trading, side, size, price, side_h, max_retries=5):
    for attempt in range(max_retries):
        try:
            limit_order_info = client.futures_create_order(
                symbol=coin + pair_trading,
                type='LIMIT',
                timeInForce='GTC',
                positionSide=side,
                price=price,
                side=side_h,
                quantity=size
            )
            return limit_order_info
        except Exception as e:
            print(f"Error placing limit order (attempt {attempt+1}): {e}")
            time.sleep(1)
    raise Exception("Max retries exceeded for post_limit_order")

def post_stop_loss_order_test(client, coin, pair_trading, side, size, price, side_h, max_retries=5):
    """
    Place a stop loss order using Binance's new Algo Order API (as of Dec 9, 2025).
    This uses the /fapi/v1/algoOrder endpoint with algoType=CONDITIONAL.
    """
    for attempt in range(max_retries):
        try:
            # Use the new Algo Order API endpoint
            params = {
                'algoType': 'CONDITIONAL',
                'symbol': coin + pair_trading,
                'side': side_h,
                'positionSide': side,
                'type': 'STOP_MARKET',
                'triggerPrice': str(price),
                'quantity': str(size),
                'workingType': 'MARK_PRICE',
                'priceProtect': 'TRUE'
            }
            
            # Use _request_futures_api like other futures methods
            stop_market_info = client._request_futures_api('post', 'algoOrder', True, data=params)
            return stop_market_info
        except Exception as e:
            print(f"Error placing stop loss order (attempt {attempt+1}): {e}")
            time.sleep(5 if attempt >= 5 else 1)
    raise Exception("Max retries exceeded for post_stop_loss_order")

def cancel_orders_test(client, coin, pair_trading, order, max_retries=5):
    for attempt in range(max_retries):
        try:
            cancel_order_info = client.futures_cancel_order(
                symbol=coin + pair_trading,
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

def cancel_algo_order_test(client, coin, pair_trading, order, max_retries=5):
    """
    Cancel an algo order using Binance's Algo Order API (as of Dec 9, 2025).
    This uses the DELETE /fapi/v1/algoOrder endpoint.
    """
    for attempt in range(max_retries):
        try:
            params = {
                'symbol': coin + pair_trading,
                'algoId': order['algoId']  # Algo orders use algoId instead of orderId
            }
            # Use _request_futures_api with delete method
            cancel_order_info = client._request_futures_api('delete', 'algoOrder', True, data=params)
            return cancel_order_info
        except Exception as e:
            print(f"Error cancelling algo order (attempt {attempt+1}): {e}")
            if hasattr(e, 'code') and e.code == -2011: # type: ignore
                print("Algo order not found", e)
                return []
            time.sleep(1)
    return []


def set_open_position_test(coin,pair_trading,side):
    client = Client(api_key, api_secret)
    candle = fetch_candles(coin,pair_trading,'1m',1)
    op = round(float(candle['low'].iloc[0]) - 1000, 1)
    size = 0.005
    print(f'Placing limit order {side}: Open : {op}. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    msg = f'Placing limit order test'
    #send_telegram_message_HTML(msg)
    close_limit_order = post_limit_order_test(client, coin, pair_trading, 'SHORT', size, op, 'BUY') #close short
    open_limit_order = post_limit_order_test(client, coin, pair_trading, side, size, op, 'BUY')
    stop_loss_order = post_stop_loss_order_test(client, coin, pair_trading, side, size, op, 'SELL')
    time.sleep(5)
    cancel_order = cancel_orders_test(client, coin, pair_trading, open_limit_order, max_retries=5)
    cancel_algo_order = cancel_algo_order_test(client, coin, pair_trading, stop_loss_order, max_retries=5)  # Use algo cancel for stop loss


api_key = os.environ.get(f'binance_api_futures')
api_secret = os.environ.get(f'binance_secret_futures')


set_open_position_test('BTC','USDC','LONG')