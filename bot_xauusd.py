import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. ตั้งค่าบัญชีและพารามิเตอร์ของระบบ
# ==========================================
LOGIN =          # ใส่เลขบัญชี Exness
PASSWORD = "" # รหัสผ่าน MT5
SERVER = "Exness-MT5Real"  # เซิร์ฟเวอร์ (ดูได้จากหน้าต่าง Navigator ใน MT5)

SYMBOL = "XAUUSDm"  # ชื่อออเดอร์ของ Exness (บัญชี Standard มักจะมีตัว m ต่อท้าย)
TIMEFRAME = mt5.TIMEFRAME_M5
LOT_SIZE = 0.01

# ค่าจาก Backtest
LOOKBACK = 50 
ATR_PERIOD = 14
ATR_MULT = 2.0  # เท่ากับ 1 ตามโค้ด Backtest ท้ายสุด
RR = 1.5

# ==========================================
# 2. ฟังก์ชันเชื่่อมต่อ MT5
# ==========================================
def init_mt5():
    if not mt5.initialize():
        print("ไม่สามารถรัน MT5 ได้")
        mt5.shutdown()
        return False
    authorized = mt5.login(LOGIN, PASSWORD, server=SERVER)
    if not authorized:
        print(f"เข้าสู่ระบบล้มเหลว รหัสข้อผิดพลาด: {mt5.last_error()}")
        return False
    print("เชื่อมต่อ MT5 และเข้าสู่ระบบเรียบร้อยแล้ว!")
    return True

# ==========================================
# 3. ฟังก์ชันดึงข้อมูลและคำนวณ Indicator
# ==========================================
def get_data_and_indicators():
    # ดึงข้อมูลย้อนหลัง
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK)
    if rates is None:
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # คำนวณ ATR
    df["H-L"] = df["high"] - df["low"]
    df["H-C"] = abs(df["high"] - df["close"].shift())
    df["L-C"] = abs(df["low"] - df["close"].shift())
    df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(ATR_PERIOD).mean()
    
    # คำนวณกรอบ Breakout
    df["HH"] = df["high"].rolling(LOOKBACK).max()
    df["LL"] = df["low"].rolling(LOOKBACK).min()
    
    return df

# ==========================================
# 4. ฟังก์ชันส่งคำสั่งซื้อขาย
# ==========================================
def send_order(order_type, price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": float(LOT_SIZE),
        "type": order_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 20, # ยอมรับ Slippage
        "magic": 999111, # เลข Magic Number ของบอท
        "comment": "Breakout Python",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, # Exness ส่วนใหญ่ใช้ IOC หรือ FOK
    }
    result = mt5.order_send(request)
    return result

# ==========================================
# 5. ฟังก์ชันหลักสำหรับรัน Bot ทำงานตลอดเวลา
# ==========================================
def run_bot():
    print(f"เริ่มการทำงานของ Bot เฝ้าดูสัญลักษณ์ {SYMBOL}...")
    
    last_bar_time = None
    
    while True:
        df = get_data_and_indicators()
        
        if df is None:
            time.sleep(1)
            continue
            
        current_bar_time = df.iloc[-1]['time']
        
        # ถ้าราคาปิดแท่งเทียน 5 นาทีเรียบร้อยแล้ว (รันเช็คสัญญาณแค่ตอนจบแท่ง)
        if last_bar_time is None or current_bar_time > last_bar_time:
            
            # ตัดแท่งปัจจุบันที่ยังไม่จบออก
            df_closed = df.iloc[:-1].copy()
            
            # ข้อมูลแท่งที่ปิดไปแล้วล่าสุด (i) และก่อนหน้า (i-1)
            last = df_closed.iloc[-1]
            prev = df_closed.iloc[-2]
            
            close = last["close"]
            atr = last["ATR"]
            
            # กรอบของแท่งรับพิจารณา
            upper = prev["HH"] + atr * 2   # หรือเปลี่ยนเป็นตัวคูณตามที่คุณต้องการ 
            lower = prev["LL"] - atr * 2
            
            # ตรวจสอบว่าระบบมีออเดอร์ของบอทที่ถือค้างอยู่หรือไม่?
            # ถ้ามีเราจะยังไม่เปิดออเดอร์ซ้อน (ทำทีละ 1 ไม้)
            positions = mt5.positions_get(symbol=SYMBOL)
            magic_positions = [p for p in positions if p.magic == 999111] if positions else []
            
            if len(magic_positions) == 0:  
                # สัญญาณ BUY
                if close > upper:
                    ask_price = mt5.symbol_info_tick(SYMBOL).ask
                    sl = ask_price - (atr * 1)   # Stop Loss
                    tp = ask_price + (atr * 1.5) # Take Profit
                    
                    print(f"🚨 [BUY Signal] ราคา: {ask_price} | SL: {sl:.2f} | TP: {tp:.2f}")
                    result = send_order(mt5.ORDER_TYPE_BUY, ask_price, sl, tp)
                    print(f"ส่งคำสั่ง: {result.comment}")

                # สัญญาณ SELL
                elif close < lower:
                    bid_price = mt5.symbol_info_tick(SYMBOL).bid
                    sl = bid_price + (atr * 1)   # Stop Loss
                    tp = bid_price - (atr * 1.5) # Take Profit
                    
                    print(f"🚨 [SELL Signal] ราคา: {bid_price} | SL: {sl:.2f} | TP: {tp:.2f}")
                    result = send_order(mt5.ORDER_TYPE_SELL, bid_price, sl, tp)
                    print(f"ส่งคำสั่ง: {result.comment}")
            
            last_bar_time = current_bar_time
            
        # รอ 1 วินาทีก่อนวนลูปเช็คใหม่
        time.sleep(1)

# เริ่มการทำงานของสคริปต์
if __name__ == '__main__':
    if init_mt5():
        # ตรวจสอบชื่อคู่เงินให้พร้อมเทรด
        mt5.symbol_select(SYMBOL, True)
        try:
            run_bot()
        except KeyboardInterrupt:
            print("ปิดบอทเรียบร้อยแล้ว")
            mt5.shutdown()
