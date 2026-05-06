import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")

# 币种配置（OKX 交易对）
CRYPTO = {
    "BTC-USDT": "比特币",
    "ETH-USDT": "以太坊"
}

# ========== 技术指标函数 ==========

def calculate_adx(high, low, close, period=14):
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))[1:]
    
    def ema(arr, period):
        return pd.Series(arr).ewm(span=period, adjust=False).mean().values
    
    plus_dm_smooth = ema(plus_dm, period)
    minus_dm_smooth = ema(minus_dm, period)
    tr_smooth = ema(tr, period)
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = ema(dx, period)
    
    return plus_di, minus_di, adx

def calculate_rsi(close, period=14):
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_bollinger_width(close, period=20, std_dev=2):
    close_series = pd.Series(close)
    ma = close_series.rolling(window=period).mean()
    std = close_series.rolling(window=period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    width = (upper - lower) / ma * 100
    return width.iloc[-1]

def calculate_macd(close, fast=12, slow=26, signal=9):
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram.iloc[-1]

# ========== 斐波那契函数 ==========

def find_swing_points(high, low, lookback=30):
    n = len(high)
    if n < lookback:
        lookback = max(2, n // 2)
    recent_high = high[-lookback:]
    recent_low = low[-lookback:]
    swing_high = np.max(recent_high)
    swing_low = np.min(recent_low)
    high_indices = np.where(high == swing_high)[0]
    low_indices = np.where(low == swing_low)[0]
    high_idx_in_range = [idx for idx in high_indices if idx >= n - lookback]
    low_idx_in_range = [idx for idx in low_indices if idx >= n - lookback]
    high_idx = high_idx_in_range[0] if high_idx_in_range else n - 1
    low_idx = low_idx_in_range[0] if low_idx_in_range else n - 1
    is_uptrend = high_idx > low_idx
    return swing_high, swing_low, is_uptrend

def calculate_fibonacci_levels(high, low, lookback=30):
    swing_high, swing_low, is_uptrend = find_swing_points(high, low, lookback)
    fib_levels = {}
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    names = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]
    if is_uptrend:
        diff = swing_high - swing_low
        for ratio, name in zip(ratios, names):
            fib_levels[name] = round(swing_high - diff * ratio, 2)
        fib_type = "📈 斐波那契回撤位 (上升趋势)"
    else:
        diff = swing_high - swing_low
        for ratio, name in zip(ratios, names):
            fib_levels[name] = round(swing_low + diff * ratio, 2)
        fib_type = "📉 斐波那契反弹位 (下降趋势)"
    return fib_levels, fib_type, swing_high, swing_low

def get_price_position_relative_to_fib(current_price, fib_levels):
    items = [(float(level.split('%')[0]), price) for level, price in fib_levels.items()]
    items.sort(key=lambda x: x[1])
    for i, (ratio, price) in enumerate(items):
        if current_price < price:
            if i == 0:
                return f"🔻 低于 {items[0][0]}% 支撑位", "极端低位，强支撑区域"
            prev_ratio, prev_price = items[i-1]
            if abs(current_price - prev_price) < abs(current_price - price):
                return f"📍 位于 {prev_ratio}% - {ratio}% 之间，靠近 {prev_ratio}%", f"支撑/阻力参考: {prev_price:.0f} - {price:.0f}"
            else:
                return f"📍 位于 {prev_ratio}% - {ratio}% 之间，靠近 {ratio}%", f"支撑/阻力参考: {prev_price:.0f} - {price:.0f}"
    return f"🔺 高于 {items[-1][0]}% 阻力位", "极端高位，强阻力区域"

# ========== OKX 数据获取（使用 requests 直接调用 API） ==========

def get_crypto_data(symbol, days=60):
    """
    使用 requests 直接调用 OKX 公开 API
    symbol 格式: 'BTC-USDT' 或 'ETH-USDT'
    """
    try:
        # 1. 获取日线 K 线数据
        params = {
            'instId': symbol,
            'bar': '1D',
            'limit': '300'
        }
        response = requests.get('https://www.okx.com/api/v5/market/history-candles', params=params)
        result = response.json()
        
        if result['code'] != '0':
            print(f"OKX API 错误: {result['msg']}")
            return None
        
        data = result['data']
        if not data:
            return None
        
        # 数据格式: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
        # 注意：返回的是倒序（最新在前），需要反转
        data.reverse()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol',
                                         'volCcy', 'volCcyQuote', 'confirm'])
        for col in ['open', 'high', 'low', 'close', 'vol']:
            df[col] = df[col].astype(float)
        
        # 2. 获取实时 Ticker（最新价和 24h 涨跌幅）
        ticker_response = requests.get('https://www.okx.com/api/v5/market/ticker', params={'instId': symbol})
        ticker_result = ticker_response.json()
        
        if ticker_result['code'] != '0':
            print(f"获取实时价格失败: {ticker_result['msg']}")
            current = df['close'].iloc[-1]
            change_pct = 0.0
        else:
            ticker = ticker_result['data'][0]
            current = float(ticker['last'])
            # 注意：字段名是 'changePct'（大写 P）
            change_pct = float(ticker.get('changePct', 0))
        
        # 计算技术指标（基于日线）
        _, _, adx = calculate_adx(df['high'].values, df['low'].values, df['close'].values, period=14)
        current_adx = adx[-1] if len(adx) > 0 else 0
        
        current_rsi = calculate_rsi(df['close'].values, period=14)
        bb_width = calculate_bollinger_width(df['close'].values, period=20)
        macd_hist = calculate_macd(df['close'].values)
        
        ma20 = df['close'].tail(20).mean()
        ma60 = df['close'].tail(60).mean()
        
        vol_avg = df['vol'].tail(20).mean()
        vol_current = df['vol'].iloc[-1]
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1
        
        # 斐波那契
        fib_levels, fib_type, swing_high, swing_low = calculate_fibonacci_levels(
            df['high'].values, df['low'].values, lookback=30
        )
        fib_position, fib_advice = get_price_position_relative_to_fib(current, fib_levels)
        
        return {
            "symbol": symbol,
            "current": round(current, 2),
            "change_pct": round(change_pct, 2),
            "adx": round(current_adx, 2),
            "rsi": round(current_rsi, 1),
            "bb_width": round(bb_width, 2),
            "macd_hist": round(macd_hist, 4),
            "ma20": round(ma20, 2),
            "ma60": round(ma60, 2),
            "price_above_ma20": current > ma20,
            "price_above_ma60": current > ma60,
            "vol_ratio": round(vol_ratio, 2),
            "fib_levels": fib_levels,
            "fib_type": fib_type,
            "fib_position": fib_position,
            "fib_advice": fib_advice,
            "swing_high": swing_high,
            "swing_low": swing_low
        }
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None

# ========== 指标判断与综合建议 ==========

def judge_bb_width_status(width):
    if width < 5:
        return "⏸️ 极度压缩", "波动率极低，即将变盘"
    elif width < 10:
        return "📉 低波动", "震荡持续，注意突破"
    elif width < 20:
        return "📊 正常波动", "趋势可能延续"
    else:
        return "⚠️ 高波动", "风险加大，严格止损"

def judge_macd_status(macd_hist):
    if macd_hist > 0:
        return "🟢 强劲多头动能" if macd_hist > 100 else "📗 多头动能"
    else:
        return "🔴 强劲空头动能" if macd_hist < -100 else "📘 空头动能"

def judge_rsi_status(rsi):
    if rsi >= 70:
        return "🔴 超买区", "注意回调风险，谨慎追高"
    elif rsi <= 30:
        return "🟢 超卖区", "可能反弹，关注买入机会"
    elif rsi >= 50:
        return "📗 强势区", "多头占优"
    else:
        return "📘 弱势区", "空头占优"

def generate_action_advice(data):
    adx = data['adx']
    rsi = data['rsi']
    bb_width = data['bb_width']
    macd_hist = data['macd_hist']
    price = data['current']
    fib_618 = data['fib_levels'].get("61.8%")
    fib_382 = data['fib_levels'].get("38.2%")
    
    bullish = 0
    bearish = 0
    signals = []
    
    if adx > 25 and data['price_above_ma20'] and data['price_above_ma60']:
        bullish += 2
        signals.append("趋势多头")
    elif adx > 25 and not data['price_above_ma20']:
        bearish += 2
        signals.append("趋势空头")
    elif adx < 20:
        signals.append("震荡")
    
    if rsi < 30:
        bullish += 1
        signals.append("超卖")
    elif rsi > 70:
        bearish += 1
        signals.append("超买")
    
    if macd_hist > 0:
        bullish += 1
        signals.append("MACD+")
    else:
        bearish += 1
        signals.append("MACD-")
    
    if fib_618 and price <= fib_618:
        bullish += 1
        signals.append("接近斐波那契支撑")
    if fib_382 and price >= fib_382 and adx > 25:
        bearish += 1
        signals.append("接近斐波那契阻力")
    
    if bb_width < 5:
        signals.append("变盘预警")
    
    signal_summary = " | ".join(signals)
    
    if bullish >= 3:
        return f"✅ 强烈做多 ({signal_summary})", "顺势持仓，回调至斐波那契支撑位加仓"
    if bearish >= 3:
        return f"❌ 强烈做空 ({signal_summary})", "顺势做空，反弹至斐波那契阻力位加仓"
    if bullish >= 2:
        return f"📈 偏多 ({signal_summary})", "轻仓试多，设好止损"
    if bearish >= 2:
        return f"📉 偏空 ({signal_summary})", "轻仓试空，设好止损"
    if bb_width < 5:
        return f"⚡ 变盘预警 ({signal_summary})", "减小仓位，等待方向明确"
    return f"➡️ 观望 ({signal_summary})", "多空信号不明确，等待共振"

# ========== 飞书推送 ==========

def send_to_feishu(message):
    if not FEISHU_WEBHOOK:
        print("未配置飞书 Webhook")
        return
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": f"📊 加密货币技术分析 (OKX数据) | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "content": [[{"tag": "text", "text": message}]]
                }
            }
        }
    }
    try:
        resp = requests.post(FEISHU_WEBHOOK, json=payload, timeout=10)
        if resp.status_code == 200:
            print("飞书推送成功")
        else:
            print(f"推送失败: {resp.text}")
    except Exception as e:
        print(f"推送异常: {e}")

def format_fib_levels(fib_levels):
    return "\n".join([f"  {k}: ${v:,.0f}" for k, v in fib_levels.items()])

# ========== 主函数 ==========

def main():
    print(f"执行时间: {datetime.now()}")
    report_lines = []
    for symbol, name in CRYPTO.items():
        data = get_crypto_data(symbol, days=60)
        if not data:
            report_lines.append(f"\n❌ 【{name}】数据获取失败")
            continue
        rsi_status, rsi_advice = judge_rsi_status(data['rsi'])
        bb_status, bb_advice = judge_bb_width_status(data['bb_width'])
        macd_status = judge_macd_status(data['macd_hist'])
        signal_summary, action_detail = generate_action_advice(data)
        
        line = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 {name} | {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 现价: ${data['current']:,.0f}  |  {data['change_pct']:+.2f}%

📊 技术指标
├─ ADX: {data['adx']} → {'🔥 趋势' if data['adx'] > 25 else '🌀 震荡' if data['adx'] < 20 else '⚡ 过渡'}
├─ RSI: {data['rsi']} → {rsi_status}
├─ MACD柱: {data['macd_hist']:.4f} → {macd_status}
├─ 布林带宽度: {data['bb_width']}% → {bb_status}
├─ 成交量: {data['vol_ratio']}倍均量
└─ 均线: MA20=${data['ma20']:,.0f} | MA60=${data['ma60']:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 斐波那契 (30日高低点)
{data['fib_type']}
{format_fib_levels(data['fib_levels'])}
📌 当前: {data['fib_position']}
💡 {data['fib_advice']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 综合信号: {signal_summary}
💡 操作建议: {action_detail}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        report_lines.append(line)
        print(line)
    full_report = "".join(report_lines)
    send_to_feishu(full_report)

if __name__ == "__main__":
    main()