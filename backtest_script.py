import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(file_path="2026.3.29XAUUSD_M1_UTCPlus07-M1-No Session.csv"):
    """
    ฟังก์ชันโหลดข้อมูลจาก CSV และ Resample เป็น 5 นาที (M5)
    """
    data = pd.read_csv(file_path)
    
    # แปลง Date และ Time ให้เป็น Datetime Index
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.set_index('Datetime', inplace=True)
    data.drop(columns=['Date', 'Time'], inplace=True)
    
    # Resample ข้อมูล 1 นาที (M1) ให้เป็น 5 นาที (M5) เหมือนของเดิม
    data_m5 = data.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
            
    return data_m5

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
    
    # เพิ่มเวลา ชั่วโมงและวันในสัปดาห์ เพื่อให้ง่ายต่อการใช้วิเคราะห์ ML 
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    
    return data.dropna()

def run_backtest(data, risk_per_trade=0.2, atr_mult=2.0, rr=1.5):
    """
    ฟังก์ชันรัน Backtest 
    (ในส่วนนี้มีที่ว่างให้ใส่เงื่อนไขจาก Decision Tree)
    """
    balance = 50.0
    initial_balance = balance
    
    equity = []
    trades = []
    
    # แปลงข้อมูล Column เป็น Numpy Array เพื่อให้ความเร็วในการรันเร็วขึ้น 100 เท่า (ไวกว่า .iloc)
    closes = data["Close"].values
    atrs = data["ATR"].values
    hours = data["Hour"].values
    day_of_weeks = data["DayOfWeek"].values
    hhs = data["HH"].values
    lls = data["LL"].values
    highs = data["High"].values
    lows = data["Low"].values
    
    length = len(data)
    
    for i in range(length):
        # รับข้อมูลดัชนีปัจจุบัน
        close = closes[i]
        atr = atrs[i]
        hour = hours[i]
        
        # แท่งที่ i ต้องเทียบกับแท่งก่อนหน้า i-1 
        if i < 1:
            equity.append(balance)
            continue
            
        upper = hhs[i-1] + atr * atr_mult
        lower = lls[i-1] - atr * atr_mult
        
        # ===============================================
        # ใส่เงื่อนไข (Rules) ที่ได้จาก Decision Tree (อัปเดตใหม่)
        # ===============================================
        # เตรียมตัวแปรสำหรับใช้วิเคราะห์ตาม Tree
        day_of_week = day_of_weeks[i] # 0=จันทร์, 1=อังคาร, ..., 4=ศุกร์
        
        # กำหนดช่วงเวลา NY Session (อิงเวลาไทย UTC+7) ตลาดเปิดช่วง 19:00
        is_ny_session = 1 if (hour >= 19 or hour < 4) else 0
        
        # แปลงกฎจากภาพ Decision Tree (ซ้ายมือคือ True, ขวามือคือ False)
        # Root: Session_NY <= 0.5 (คือถ้า ไม่ใช่ NY ไปฝั่งซ้าย ถ้า ใช่ NY ไปฝั่งขวา)
        is_good_ml = False
        
        if is_ny_session == 0: 
            # ---> ฝั่งซ้ายของ Tree (Not NY)
            if day_of_week == 0: # โหนด Day_of_Week <= 0.5 (วันจันทร์)
                # ไม่ว่า Hour <= 5.5 หรือ > 5.5 ก็ให้ค่า Class = Hit TP ทั้งคู่
                is_good_ml = True
            else:
                # วันอังคาร-ศุกร์ ตกโหนด Hit SL ทั้งหมด (ไม่เกินและเกินบ่ายสองครึ่ง)
                is_good_ml = False
        else: 
            # ---> ฝั่งขวาของ Tree (Yes NY)
            if day_of_week == 0: # วันจันทร์ในเวลา NY 
                is_good_ml = False # Class = Hit SL
            else:
                # วันอังคาร-ศุกร์ ตกโหนด Hour_of_Day <= 22.5
                if hour <= 22: # 19:00 - 22:59
                    is_good_ml = True # Class = Hit TP
                else: 
                    is_good_ml = False # Class = Hit SL
                    
        # ขา BUY และ SELL ใช้เงื่อนไขเดียวกันที่กรองมาแล้ว
        ml_condition_buy = is_good_ml
        ml_condition_sell = is_good_ml
        
        result = None
        
        # สัญญาณ BUY
        if close > upper and ml_condition_buy:
            entry = close
            sl = entry - atr * 1
            tp = entry + atr * rr
            
            # จำลองผลกำไร-ขาดทุนในแท่งถัดๆ ไป (สูงสุด 30 แท่ง)
            max_j = min(i+31, length)
            for j in range(i+1, max_j):
                if lows[j] <= sl:
                    result = -1  # ขาดทุน
                    break
                if highs[j] >= tp:
                    result = 1   # กำไร
                    break
                    
        # สัญญาณ SELL
        elif close < lower and ml_condition_sell:
            entry = close
            sl = entry + atr * 1
            tp = entry - atr * rr
            
            # จำลองผลกำไร-ขาดทุนในแท่งถัดๆ ไป (สูงสุด 30 แท่ง)
            max_j = min(i+31, length)
            for j in range(i+1, max_j):
                if highs[j] >= sl:
                    result = -1  # ขาดทุน
                    break
                if lows[j] <= tp:
                    result = 1   # กำไร
                    break
                    
        # ตัดยอด Balance ของไม้นี้
        if result is not None:
            trades.append(result)
            if result == 1:
                # ชนะ: ได้กำไร = Risk * RR
                balance *= (1 + (risk_per_trade * rr))
            else:
                # แพ้: เสีย = Risk
                balance *= (1 - risk_per_trade)
            
        equity.append(balance)
        
    # คืนค่าผลลัพธ์เพื่อนำไปพล็อตกราฟ
    data_res = data.copy()
    data_res['Equity'] = equity
    return data_res, trades, initial_balance, balance

def main():
    print("เริ่มการรัน Backtest...")
    
    print("1. โหลดข้อมูลจากไฟล์ CSV (2026.3.29XAUUSD_M1_UTCPlus07-M1-No Session.csv)...")
    df = get_data("2026.3.29XAUUSD_M1_UTCPlus07-M1-No Session.csv")
    
    print("2. คำนวณอินดิเคเตอร์ ATR และ Breakout Levels...")
    df = calc_indicators(df, lookback=50, atr_period=14)
    
    print("3. ทำการเทรดจำลอง...")
    # คุณสามารถปรับตั้งค่า RR, ATR Multiplier หรือ % Risk ต่อเทรดได้ที่นี่
    # แนะนำให้ใช้ความเสี่ยงแค่ 2% ต่อไม้ (0.02)
    df_res, trades, init_bal, final_bal = run_backtest(df, risk_per_trade=0.1, atr_mult=3.0, rr=1.5)
    
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
