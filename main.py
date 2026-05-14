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

def calculate_rsi_series(close, period=14):
    """返回完整的 RSI 序列"""
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_macd_series(close, fast=12, slow=26, signal=9):
    """返回 MACD 柱状图（histogram）完整序列"""
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram.values

def calculate_bollinger_width(close, period=20, std_dev=2):
    close_series = pd.Series(close)
    ma = close_series.rolling(window=period).mean()
    std = close_series.rolling(window=period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    width = (upper - lower) / ma * 100
    return width.iloc[-1]

# ========== 背离检测函数 ==========

def find_peaks(series, order=2):
    """
    简单寻找局部极值点
    series: 一维数组
    order: 左右各 order 个点进行比较
    返回 (峰值索引列表, 谷值索引列表)
    """
    peaks = []
    troughs = []
    n = len(series)
    for i in range(order, n - order):
        if all(series[i] >= series[i-j] for j in range(1, order+1)) and \
           all(series[i] >= series[i+j] for j in range(1, order+1)):
            peaks.append(i)
        if all(series[i] <= series[i-j] for j in range(1, order+1)) and \
           all(series[i] <= series[i+j] for j in range(1, order+1)):
            troughs.append(i)
    return peaks, troughs

def detect_rsi_divergence(close, rsi, lookback=50):
    """检测 RSI 顶背离和底背离"""
    if len(close) < lookback:
        lookback = len(close)
    close_seg = close[-lookback:]
    rsi_seg = rsi[-lookback:]
    
    price_peaks, price_troughs = find_peaks(close_seg, order=2)
    rsi_peaks, rsi_troughs = find_peaks(rsi_seg, order=2)
    
    top_div = None
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        last_rsi_peak = rsi_peaks[-1]
        prev_rsi_peak = rsi_peaks[-2]
        if (close_seg[last_price_peak] > close_seg[prev_price_peak] and
            rsi_seg[last_rsi_peak] < rsi_seg[prev_rsi_peak]):
            top_div = f"⚠️ RSI顶背离：价格新高 ({close_seg[last_price_peak]:.2f})，RSI降低 ({rsi_seg[last_rsi_peak]:.1f})"
    
    bottom_div = None
    if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
        last_price_trough = price_troughs[-1]
        prev_price_trough = price_troughs[-2]
        last_rsi_trough = rsi_troughs[-1]
        prev_rsi_trough = rsi_troughs[-2]
        if (close_seg[last_price_trough] < close_seg[prev_price_trough] and
            rsi_seg[last_rsi_trough] > rsi_seg[prev_rsi_trough]):
            bottom_div = f"✅ RSI底背离：价格新低 ({close_seg[last_price_trough]:.2f})，RSI抬高 ({rsi_seg[last_rsi_trough]:.1f})"
    
    return top_div, bottom_div

def detect_macd_divergence(close, macd_hist, lookback=50):
    """检测 MACD 柱顶背离和底背离"""
    if len(close) < lookback:
        lookback = len(close)
    close_seg = close[-lookback:]
    macd_seg = macd_hist[-lookback:]
    
    price_peaks, price_troughs = find_peaks(close_seg, order=2)
    macd_peaks, macd_troughs = find_peaks(macd_seg, order=2)
    
    top_div = None
    if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
        last_pp = price_peaks[-1]
        prev_pp = price_peaks[-2]
        last_mp = macd_peaks[-1]
        prev_mp = macd_peaks[-2]
        if (close_seg[last_pp] > close_seg[prev_pp] and
            macd_seg[last_mp] < macd_seg[prev_mp]):
            top_div = f"⚠️ MACD顶背离：价格新高 ({close_seg[last_pp]:.2f})，MACD柱降低 ({macd_seg[last_mp]:.4f})"
    
    bottom_div = None
    if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
        last_pt = price_troughs[-1]
        prev_pt = price_troughs[-2]
        last_mt = macd_troughs[-1]
        prev_mt = macd_troughs[-2]
        if (close_seg[last_pt] < close_seg[prev_pt] and
            macd_seg[last_mt] > macd_seg[prev_mt]):
            bottom_div = f"✅ MACD底背离：价格新低 ({close_seg[last_pt]:.2f})，MACD柱抬高 ({macd_seg[last_mt]:.4f})"
    
    return top_div, bottom_div

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

# ========== OKX 数据获取（支持多周期） ==========

def get_crypto_data(symbol, bar='1D', limit=300):
    """
    通用数据获取函数
    bar: K线周期，支持 '1D', '4H', '1H' 等
    """
    try:
        params = {
            'instId': symbol,
            'bar': bar,
            'limit': str(limit)
        }
        response = requests.get('https://www.okx.com/api/v5/market/history-candles', params=params)
        result = response.json()
        
        if result['code'] != '0':
            print(f"OKX API 错误 ({bar}): {result['msg']}")
            return None
        
        data = result['data']
        if not data:
            return None
        
        data.reverse()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol',
                                         'volCcy', 'volCcyQuote', 'confirm'])
        for col in ['open', 'high', 'low', 'close', 'vol']:
            df[col] = df[col].astype(float)
        
        # 获取实时价格
        ticker_response = requests.get('https://www.okx.com/api/v5/market/ticker', params={'instId': symbol})
        ticker_result = ticker_response.json()
        
        if ticker_result['code'] != '0':
            current = df['close'].iloc[-1]
            change_pct = 0.0
        else:
            ticker = ticker_result['data'][0]
            current = float(ticker['last'])
            change_pct = float(ticker.get('changePct', 0))
        
        # 计算技术指标
        _, _, adx = calculate_adx(df['high'].values, df['low'].values, df['close'].values, period=14)
        current_adx = adx[-1] if len(adx) > 0 else 0
        
        rsi_series = calculate_rsi_series(df['close'].values, period=14)
        current_rsi = rsi_series[-1] if not np.isnan(rsi_series[-1]) else 50.0
        
        macd_series = calculate_macd_series(df['close'].values)
        current_macd_hist = macd_series[-1] if not np.isnan(macd_series[-1]) else 0.0
        
        bb_width = calculate_bollinger_width(df['close'].values, period=20)
        
        ma20 = df['close'].tail(20).mean()
        ma60 = df['close'].tail(60).mean() if len(df) >= 60 else df['close'].mean()
        
        vol_avg = df['vol'].tail(20).mean()
        vol_current = df['vol'].iloc[-1]
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1
        
        # 背离检测
        rsi_top_div, rsi_bottom_div = None, None
        macd_top_div, macd_bottom_div = None, None
        if len(df) >= 30:
            rsi_top_div, rsi_bottom_div = detect_rsi_divergence(
                df['close'].values, rsi_series, lookback=min(50, len(df))
            )
            macd_top_div, macd_bottom_div = detect_macd_divergence(
                df['close'].values, macd_series, lookback=min(50, len(df))
            )
        
        return {
            "symbol": symbol,
            "bar": bar,
            "current": round(current, 2),
            "change_pct": round(change_pct, 2),
            "adx": round(current_adx, 2),
            "rsi": round(current_rsi, 1),
            "bb_width": round(bb_width, 2),
            "macd_hist": round(current_macd_hist, 4),
            "ma20": round(ma20, 2),
            "ma60": round(ma60, 2),
            "price_above_ma20": current > ma20,
            "price_above_ma60": current > ma60,
            "vol_ratio": round(vol_ratio, 2),
            "rsi_top_div": rsi_top_div,
            "rsi_bottom_div": rsi_bottom_div,
            "macd_top_div": macd_top_div,
            "macd_bottom_div": macd_bottom_div,
        }
    except Exception as e:
        print(f"获取 {symbol} ({bar}) 数据失败: {e}")
        return None

def get_crypto_data_with_fib(symbol):
    """获取日线数据（含斐波那契）"""
    daily_data = get_crypto_data(symbol, bar='1D', limit=300)
    if not daily_data:
        return None
    
    # 单独获取斐波那契需要的日线K线
    try:
        params = {'instId': symbol, 'bar': '1D', 'limit': '60'}
        response = requests.get('https://www.okx.com/api/v5/market/history-candles', params=params)
        result = response.json()
        if result['code'] == '0' and result['data']:
            data = result['data']
            data.reverse()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol',
                                             'volCcy', 'volCcyQuote', 'confirm'])
            for col in ['high', 'low']:
                df[col] = df[col].astype(float)
            
            fib_levels, fib_type, swing_high, swing_low = calculate_fibonacci_levels(
                df['high'].values, df['low'].values, lookback=30
            )
            fib_position, fib_advice = get_price_position_relative_to_fib(daily_data['current'], fib_levels)
            
            daily_data['fib_levels'] = fib_levels
            daily_data['fib_type'] = fib_type
            daily_data['fib_position'] = fib_position
            daily_data['fib_advice'] = fib_advice
            daily_data['swing_high'] = swing_high
            daily_data['swing_low'] = swing_low
    except Exception as e:
        print(f"获取斐波那契数据失败: {e}")
        daily_data['fib_levels'] = {}
        daily_data['fib_type'] = "斐波那契计算失败"
        daily_data['fib_position'] = "N/A"
        daily_data['fib_advice'] = "N/A"
    
    return daily_data

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

def generate_action_advice(daily_data, h4_data):
    """综合日线和4小时线信号给出建议"""
    adx = daily_data['adx']
    rsi = daily_data['rsi']
    bb_width = daily_data['bb_width']
    macd_hist = daily_data['macd_hist']
    price = daily_data['current']
    
    bullish = 0
    bearish = 0
    signals = []
    
    # 日线趋势判断
    if adx > 25 and daily_data['price_above_ma20'] and daily_data['price_above_ma60']:
        bullish += 2
        signals.append("日线趋势多头")
    elif adx > 25 and not daily_data['price_above_ma20']:
        bearish += 2
        signals.append("日线趋势空头")
    elif adx < 20:
        signals.append("日线震荡")
    
    # RSI
    if rsi < 30:
        bullish += 1
        signals.append("日线超卖")
    elif rsi > 70:
        bearish += 1
        signals.append("日线超买")
    
    # MACD
    if macd_hist > 0:
        bullish += 1
        signals.append("日线MACD+")
    else:
        bearish += 1
        signals.append("日线MACD-")
    
    # 4小时线信号加分
    if h4_data:
        h4_adx = h4_data['adx']
        h4_macd = h4_data['macd_hist']
        h4_rsi = h4_data['rsi']
        
        if h4_adx > 25:
            if h4_data['price_above_ma20']:
                bullish += 1
                signals.append("4H趋势多头")
            else:
                bearish += 1
                signals.append("4H趋势空头")
        
        if h4_macd > 0:
            bullish += 1
            signals.append("4H MACD+")
        else:
            bearish += 1
            signals.append("4H MACD-")
        
        if h4_rsi < 30:
            bullish += 1
            signals.append("4H超卖")
        elif h4_rsi > 70:
            bearish += 1
            signals.append("4H超买")
    
    # 背离信号
    if daily_data.get('rsi_bottom_div'):
        bullish += 1
        signals.append("日线RSI底背离")
    if daily_data.get('rsi_top_div'):
        bearish += 1
        signals.append("日线RSI顶背离")
    if daily_data.get('macd_bottom_div'):
        bullish += 1
        signals.append("日线MACD底背离")
    if daily_data.get('macd_top_div'):
        bearish += 1
        signals.append("日线MACD顶背离")
    
    if h4_data:
        if h4_data.get('rsi_bottom_div'):
            bullish += 1
            signals.append("4H RSI底背离")
        if h4_data.get('rsi_top_div'):
            bearish += 1
            signals.append("4H RSI顶背离")
        if h4_data.get('macd_bottom_div'):
            bullish += 1
            signals.append("4H MACD底背离")
        if h4_data.get('macd_top_div'):
            bearish += 1
            signals.append("4H MACD顶背离")
    
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

def format_divergence_section(data, title):
    """格式化背离信息"""
    if not data:
        return f"  {title}: 数据获取失败\n"
    
    lines = [f"  {title}:"]
    div_count = 0
    if data.get('rsi_top_div'):
        lines.append(f"    {data['rsi_top_div']}")
        div_count += 1
    if data.get('rsi_bottom_div'):
        lines.append(f"    {data['rsi_bottom_div']}")
        div_count += 1
    if data.get('macd_top_div'):
        lines.append(f"    {data['macd_top_div']}")
        div_count += 1
    if data.get('macd_bottom_div'):
        lines.append(f"    {data['macd_bottom_div']}")
        div_count += 1
    if div_count == 0:
        lines.append("    未检测到明显背离")
    return "\n".join(lines)

def format_tech_section(data, title):
    """格式化技术指标"""
    if not data:
        return f"  {title}: 数据获取失败\n"
    
    rsi_status, _ = judge_rsi_status(data['rsi'])
    macd_status = judge_macd_status(data['macd_hist'])
    bb_status, _ = judge_bb_width_status(data['bb_width'])
    
    adx_status = '🔥 趋势' if data['adx'] > 25 else '🌀 震荡' if data['adx'] < 20 else '⚡ 过渡'
    
    return f"""  {title}:
    ADX: {data['adx']} → {adx_status}
    RSI: {data['rsi']} → {rsi_status}
    MACD柱: {data['macd_hist']:.4f} → {macd_status}
    布林带宽度: {data['bb_width']}% → {bb_status}
    成交量: {data['vol_ratio']}倍均量
    均线: MA20=${data['ma20']:,.0f} | MA60=${data['ma60']:,.0f}"""

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
    if not fib_levels:
        return "  斐波那契数据获取失败"
    return "\n".join([f"  {k}: ${v:,.0f}" for k, v in fib_levels.items()])

# ========== 主函数 ==========

def main():
    print(f"执行时间: {datetime.now()}")
    report_lines = []
    
    for symbol, name in CRYPTO.items():
        print(f"\n正在分析 {name}...")
        
        # 获取日线数据（含斐波那契）
        daily_data = get_crypto_data_with_fib(symbol)
        if not daily_data:
            report_lines.append(f"\n❌ 【{name}】日线数据获取失败")
            continue
        
        # 获取4小时线数据
        h4_data = get_crypto_data(symbol, bar='4H', limit=200)
        
        # 综合建议
        signal_summary, action_detail = generate_action_advice(daily_data, h4_data)
        
        # 构建报告
        line = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 {name} | {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 现价: ${daily_data['current']:,.0f}  |  {daily_data['change_pct']:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 技术指标
{format_tech_section(daily_data, '日线')}
{format_tech_section(h4_data, '4小时') if h4_data else '  4小时: 数据获取失败'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 背离检测
{format_divergence_section(daily_data, '日线')}
{format_divergence_section(h4_data, '4小时') if h4_data else '  4小时: 数据获取失败'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 斐波那契 (日线30日高低点)
{daily_data['fib_type']}
{format_fib_levels(daily_data.get('fib_levels', {}))}
📌 当前: {daily_data['fib_position']}
💡 {daily_data['fib_advice']}

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