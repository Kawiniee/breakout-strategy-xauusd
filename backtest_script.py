import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(symbol="GC=F", period="60d", interval="5m"):
    """
    ฟังก์ชันดาวน์โหลดข้อมูลจาก yfinance
    """
    data = yf.download(symbol, period=period, interval=interval)
    data = data.dropna()
    
    # เคลียร์ MultiIndex columns ให้เข้าถึงได้ง่าย
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            data.columns = data.columns.get_level_values(0)
        elif symbol in data.columns.get_level_values(0):
            data.columns = data.columns.get_level_values(1)
            
    return data

def calc_indicators(data, lookback=50, atr_period=14):
    """
    ฟังก์ชันคำนวณ ATR, กรอบ High-Low
    """
    data["H-L"] = data["High"] - data["Low"]
    data["H-C"] = abs(data["High"] - data["Close"].shift())
    data["L-C"] = abs(data["Low"] - data["Close"].shift())
    data["TR"] = data[["H-L", "H-C", "L-C"]].max(axis=1)
    data["ATR"] = data["TR"].rolling(atr_period).mean()
    
    data["HH"] = data["High"].rolling(lookback).max()
    data["LL"] = data["Low"].rolling(lookback).min()
    
    # เพิ่มเวลา ชั่วโมง เพื่อให้ง่ายต่อการใช้วิเคราะห์ ML 
    data['Hour'] = data.index.hour
    
    return data.dropna()

def run_backtest(data, risk_per_trade=0.2, atr_mult=2.0, rr=1.5):
    """
    ฟังก์ชันรัน Backtest 
    (ในส่วนนี้มีที่ว่างให้ใส่เงื่อนไขจาก Decision Tree)
    """
    balance = 100.0
    initial_balance = balance
    
    equity = []
    trades = []
    
    for i in range(len(data)):
        # รับข้อมูลดัชนีปัจจุบัน
        close = data["Close"].iloc[i]
        atr = data["ATR"].iloc[i]
        hour = data["Hour"].iloc[i]
        
        # แท่งที่ i ต้องเทียบกับแท่งก่อนหน้า i-1 
        if i < 1:
            equity.append(balance)
            continue
            
        upper = data["HH"].iloc[i-1] + atr * atr_mult
        lower = data["LL"].iloc[i-1] - atr * atr_mult
        
        # ===============================================
        # ใส่เงื่อนไข (Rules) ที่ได้จาก Decision Tree ตรงนี้
        # ===============================================
        # ตลาดอเมริกา (NY Session) มักจะอยู่ในช่วงประมาณ 08:00 - 16:00 
        # (กราฟของ yfinance อิงตามโซนเวลานิวยอร์ก หรือสามารถปรับชั่วโมงได้ตามสะดวก)
        is_ny_session = (hour >= 8) and (hour <= 16)
        
        # ขา BUY: 1. ไม่อยู่ใน NY Session และ 2. ATR ไม่โหดเกินไป
        ml_condition_buy = (not is_ny_session) and (atr <= 9.907)
        
        # ขา SELL: 1. ไม่อยู่ใน NY Session (เพราะถ้านอกเวลานี้ อัตราชนะสูงมาก)
        ml_condition_sell = (not is_ny_session)
        
        result = None
        
        # สัญญาณ BUY
        if close > upper and ml_condition_buy:
            entry = close
            sl = entry - atr * 1
            tp = entry + atr * rr
            
            # จำลองผลกำไร-ขาดทุนในแท่งถัดๆ ไป (สูงสุด 30 แท่ง)
            for j in range(i+1, min(i+31, len(data))):
                if data["Low"].iloc[j] <= sl:
                    result = -1  # ขาดทุน
                    break
                if data["High"].iloc[j] >= tp:
                    result = 1   # กำไร
                    break
                    
        # สัญญาณ SELL
        elif close < lower and ml_condition_sell:
            entry = close
            sl = entry + atr * 1
            tp = entry - atr * rr
            
            # จำลองผลกำไร-ขาดทุนในแท่งถัดๆ ไป (สูงสุด 30 แท่ง)
            for j in range(i+1, min(i+31, len(data))):
                if data["High"].iloc[j] >= sl:
                    result = -1  # ขาดทุน
                    break
                if data["Low"].iloc[j] <= tp:
                    result = 1   # กำไร
                    break
                    
        # ตัดยอด Balance ของไม้นี้
        if result is not None:
            trades.append(result)
            balance *= (1 + result * risk_per_trade)
            
        equity.append(balance)
        
    # คืนค่าผลลัพธ์เพื่อนำไปพล็อตกราฟ
    data_res = data.copy()
    data_res['Equity'] = equity
    return data_res, trades, initial_balance, balance

def main():
    print("เริ่มการรัน Backtest...")
    
    print("1. ดาวน์โหลดข้อมูล (yfinance 'GC=F')...")
    df = get_data("GC=F", "60d", "5m")
    
    print("2. คำนวณอินดิเคเตอร์ ATR และ Breakout Levels...")
    df = calc_indicators(df, lookback=50, atr_period=14)
    
    print("3. ทำการเทรดจำลอง...")
    # คุณสามารถปรับตั้งค่า RR, ATR Multiplier หรือ % Risk ต่อเทรดได้ที่นี่
    df_res, trades, init_bal, final_bal = run_backtest(df, risk_per_trade=0.2, atr_mult=2.0, rr=1.5)
    
    # 4. สรุปผลการเทรด (Summary)
    trades = np.array(trades)
    total = len(trades)
    wins = np.sum(trades == 1)
    losses = np.sum(trades == -1)
    winrate = wins / total if total > 0 else 0
    
    print("="*30)
    print("📊 BACKTEST RESULTS 📊")
    print("="*30)
    print(f"จำนวนการเทรดทั้งหมด (Trades): {total}")
    print(f"ฝั่งชนะ (Wins)             : {wins}")
    print(f"ฝั่งแพ้ (Losses)           : {losses}")
    print(f"อัตราการชนะ (Winrate)      : {winrate * 100:.2f}%")
    print(f"เงินทุนเริ่มต้น (Initial)    : {init_bal:.2f}")
    print(f"เงินทุนคงเหลือ (Final)       : {final_bal:.2f}")
    print("="*30)
    
    # 5. สร้างกราฟการเติบโตของพอร์ต
    plt.figure(figsize=(12, 6))
    plt.plot(df_res.index, df_res['Equity'], label='Equity Curve', color='green')
    plt.title("Backtest Equity Curve (Breakout XAUUSD)")
    plt.xlabel("Datetime")
    plt.ylabel("Account Balance")
    plt.legend()
    plt.grid(True)
    
    # ถ้าไม่อยากให้ป๊อปอัพเด้งปิดให้ลบบรรทัดล่างสุด
    plt.show()

if __name__ == "__main__":
    main()
