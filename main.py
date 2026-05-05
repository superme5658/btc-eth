import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")

CRYPTO = {
    "BTC-USD": "比特币",
    "ETH-USD": "以太坊"
}

# ========== 技术指标函数 ==========

def calculate_adx(high, low, close, period=14):
    """计算 ADX 指标"""
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
    """计算 RSI 指标"""
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
    """计算布林带宽度百分比"""
    close_series = pd.Series(close)
    ma = close_series.rolling(window=period).mean()
    std = close_series.rolling(window=period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    width = (upper - lower) / ma * 100
    return width.iloc[-1]

def calculate_macd(close, fast=12, slow=26, signal=9):
    """计算 MACD 柱状图值"""
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram.iloc[-1]

# ========== 斐波那契函数 ==========

def find_swing_points(high, low, lookback=30):
    """找到近期波段高点和低点（简化方法：找窗口内最高/最低）"""
    n = len(high)
    if n < lookback:
        lookback = n // 2
    
    # 找区间内的最高价和最低价
    recent_high = max(high[-lookback:])
    recent_low = min(low[-lookback:])
    
    # 找对应的位置索引
    high_idx = high[-lookback:].index(recent_high) if recent_high in high[-lookback:] else -1
    low_idx = low[-lookback:].index(recent_low) if recent_low in low[-lookback:] else -1
    
    # 判断趋势方向：如果高点出现在低点之后，视为上升趋势
    is_uptrend = high_idx > low_idx if high_idx != -1 and low_idx != -1 else True
    
    return recent_high, recent_low, is_uptrend

def calculate_fibonacci_levels(high, low, lookback=30):
    """
    计算斐波那契回撤/扩展位
    上升趋势：从低点到高点画斐波那契
    下降趋势：从高点到低点画斐波那契
    """
    swing_high, swing_low, is_uptrend = find_swing_points(high, low, lookback)
    
    fib_levels = {}
    
    # 斐波那契比例
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    names = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]
    
    if is_uptrend:
        # 上升趋势：计算回撤位（从高点到低点）
        diff = swing_high - swing_low
        for ratio, name in zip(ratios, names):
            level = swing_high - diff * ratio
            fib_levels[name] = round(level, 2)
        fib_type = "📈 斐波那契回撤位 (上升趋势)"
    else:
        # 下降趋势：计算反弹位（从低点到高点）
        diff = swing_high - swing_low
        for ratio, name in zip(ratios, names):
            level = swing_low + diff * ratio
            fib_levels[name] = round(level, 2)
        fib_type = "📉 斐波那契反弹位 (下降趋势)"
    
    return fib_levels, fib_type, swing_high, swing_low

def get_price_position_relative_to_fib(current_price, fib_levels):
    """判断当前价格位于斐波那契的哪个区间"""
    # 获取关键位列表
    levels = [(float(level.split('%')[0]), price) for level, price in fib_levels.items()]
    levels.sort(key=lambda x: x[1])  # 按价格排序
    
    for i, (ratio, price) in enumerate(levels):
        if current_price < price:
            if i == 0:
                return f"🔻 低于 {levels[0][0]}% 支撑位", "极端低位，强支撑区域"
            elif i == len(levels) - 1:
                return f"🔺 高于 {levels[-1][0]}% 阻力位", "极端高位，强阻力区域"
            else:
                prev_ratio, prev_price = levels[i-1]
                # 判断更靠近哪一个
                dist_to_prev = abs(current_price - prev_price)
                dist_to_curr = abs(current_price - price)
                if dist_to_prev < dist_to_curr:
                    return f"📍 位于 {prev_ratio}% - {ratio}% 之间，靠近 {prev_ratio}%", f"支撑/阻力参考: {prev_price} - {price}"
                else:
                    return f"📍 位于 {prev_ratio}% - {ratio}% 之间，靠近 {ratio}%", f"支撑/阻力参考: {prev_price} - {price}"
    
    # 理论不会走到这里
    return "📍 位于所有斐波那契位之上/下", "注意极端行情"

# ========== 数据获取 ==========

def get_crypto_data(symbol, days=60):
    """获取加密货币数据及所有技术指标"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d")
        
        if hist.empty or len(hist) < 30:
            return None
        
        current = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else current
        change_pct = ((current - prev_close) / prev_close) * 100
        
        # ADX
        _, _, adx = calculate_adx(
            hist['High'].values, 
            hist['Low'].values, 
            hist['Close'].values,
            period=14
        )
        current_adx = adx[-1] if len(adx) > 0 else 0
        
        # RSI
        current_rsi = calculate_rsi(hist['Close'].values, period=14)
        
        # 布林带宽度
        bb_width = calculate_bollinger_width(hist['Close'].values, period=20)
        
        # MACD
        macd_hist = calculate_macd(hist['Close'].values)
        
        # 均线
        ma20 = hist['Close'].tail(20).mean()
        ma60 = hist['Close'].tail(60).mean()
        
        # 成交量
        vol_avg = hist['Volume'].tail(20).mean()
        vol_current = hist['Volume'].iloc[-1]
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1
        
        # 斐波那契
        fib_levels, fib_type, swing_high, swing_low = calculate_fibonacci_levels(
            hist['High'].values, 
            hist['Low'].values, 
            lookback=30
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

# ========== 指标判断函数 ==========

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
        if macd_hist > 100:
            return "🟢 强劲多头动能"
        else:
            return "📗 多头动能"
    else:
        if macd_hist < -100:
            return "🔴 强劲空头动能"
        else:
            return "📘 空头动能"

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
    """基于全部指标生成操作建议（加入斐波那契判断）"""
    adx = data['adx']
    rsi = data['rsi']
    bb_width = data['bb_width']
    macd_hist = data['macd_hist']
    fib_position = data['fib_position']
    price = data['current']
    
    # 获取斐波那契关键位
    fib_levels = data['fib_levels']
    fib_618 = fib_levels.get("61.8%", None)
    fib_382 = fib_levels.get("38.2%", None)
    
    # 收集信号
    bullish_signals = 0
    bearish_signals = 0
    signal_list = []
    
    # ADX + 方向
    if adx > 25 and data['price_above_ma20'] and data['price_above_ma60']:
        bullish_signals += 2
        signal_list.append("趋势多头")
    elif adx > 25 and not data['price_above_ma20']:
        bearish_signals += 2
        signal_list.append("趋势空头")
    elif adx < 20:
        signal_list.append("震荡")
    
    # RSI
    if rsi < 30:
        bullish_signals += 1
        signal_list.append("超卖")
    elif rsi > 70:
        bearish_signals += 1
        signal_list.append("超买")
    
    # MACD
    if macd_hist > 0:
        bullish_signals += 1
        signal_list.append("MACD+")
    else:
        bearish_signals += 1
        signal_list.append("MACD-")
    
    # 斐波那契支撑/阻力
    if fib_618 and price <= fib_618:
        bullish_signals += 1
        signal_list.append("接近斐波那契支撑")
    if fib_382 and price >= fib_382 and adx > 25:
        bearish_signals += 1
        signal_list.append("接近斐波那契阻力")
    
    # 布林带变盘预警
    if bb_width < 5:
        signal_list.append("变盘预警")
    
    # 综合判断
    signal_summary = " | ".join(signal_list)
    
    if bullish_signals >= 3:
        return f"✅ 强烈做多 ({signal_summary})", "顺势持仓，回调至斐波那契支撑位加仓"
    elif bearish_signals >= 3:
        return f"❌ 强烈做空 ({signal_summary})", "顺势做空，反弹至斐波那契阻力位加仓"
    elif bullish_signals >= 2:
        return f"📈 偏多 ({signal_summary})", "轻仓试多，设好止损"
    elif bearish_signals >= 2:
        return f"📉 偏空 ({signal_summary})", "轻仓试空，设好止损"
    elif bb_width < 5:
        return f"⚡ 变盘预警 ({signal_summary})", "减小仓位，等待方向明确"
    else:
        return f"➡️ 观望 ({signal_summary})", "多空信号不明确，等待共振"

# ========== 主函数 ==========

def send_to_feishu(message):
    if not FEISHU_WEBHOOK:
        print("未配置飞书 Webhook")
        return
    
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": f"📊 加密货币技术分析 | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
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
    """格式化斐波那契水平显示"""
    lines = []
    for name, price in fib_levels.items():
        lines.append(f"  {name}: ${price:,.0f}")
    return "\n".join(lines)

def main():
    print(f"执行时间: {datetime.now()}")
    report_lines = []
    
    for symbol, name in CRYPTO.items():
        data = get_crypto_data(symbol)
        if not data:
            report_lines.append(f"\n❌ 【{name}】数据获取失败")
            continue
        
        rsi_status, rsi_advice = judge_rsi_status(data['rsi'])
        bb_status, bb_advice = judge_bb_width_status(data['bb_width'])
        macd_status = judge_macd_status(data['macd_hist'])
        signal_summary, action_detail = generate_action_advice(data)
        
        line = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 {name} | {symbol.replace('-USD', '')}
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