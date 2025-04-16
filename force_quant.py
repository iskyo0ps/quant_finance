import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

# ===================
# 数据获取模块 (网页6/7/8/10)
# ===================
class DataFetcher:
    def __init__(self):
        self.binance = ccxt.binance()
        
    def get_fed_data(self):
        """获取美联储利率决策数据 (网页1/2/3/4)"""
        df = pd.read_csv('https://api.fiscaldata.gov/fiscaldata/v1/accounting/od/rates_of_exchange')
        return df[df['country_currency_desc'] == 'U.S. DOLLAR'].set_index('record_date')

    def get_stock_data(self, ticker):
        """获取美股数据 (网页6/7)"""
        data = yf.download(ticker, start='2020-01-01', end='2025-04-16')
        return data[['Close', 'Volume']]

    def get_crypto_data(self, symbol='BTC/USDT'):
        """获取加密货币数据 (网页9/10/11)"""
        ohlcv = self.binance.fetch_ohlcv(symbol, '4h')
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')

# ===================
# 三因子分析模块 (网页1/2/3/4)
# ===================
class FedImpactAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def rate_sensitivity_beta(self, stock_data, bond_yield):
        """利率敏感度计算 (网页1因子1)"""
        merged = pd.merge(stock_data, bond_yield, left_index=True, right_index=True)
        X = merged['bond_yield'].pct_change().dropna().values.reshape(-1,1)
        y = merged['Close'].pct_change().dropna().values
        return linregress(X.flatten(), y).slope * 100  # 转换为Beta值

    def liquidity_pulse(self, m2_data):
        """流动性脉冲指标 (网页1因子2)"""
        return self.scaler.fit_transform(m2_data.values.reshape(-1,1)).flatten()
    
    def sentiment_divergence(self, put_call_ratio, institutional_position):
        """情绪错杀指标 (网页1因子3)"""
        return (put_call_ratio - institutional_position).rolling(14).mean()

# ===================
# BTC量化策略模块 (网页9/10/11)
# ===================
class BitcoinStrategy:
    def __init__(self, atr_period=8, vol_weight=21):
        self.atr_period = atr_period
        self.vol_weight = vol_weight
        
    def atr_channel(self, df):
        """ATR波动通道 (网页11策略)"""
        high = df['high'].rolling(self.atr_period).max()
        low = df['low'].rolling(self.atr_period).min()
        return (high + low) / 2
    
    def volume_weighted_breakout(self, df):
        """成交量加权突破信号 (网页9策略)"""
        df['vol_ma'] = df['volume'].rolling(self.vol_weight).mean()
        return df['volume'] > df['vol_ma'] * 1.5
    
    def overnight_effect_signal(self, df):
        """隔夜效应捕捉 (网页9现象)"""
        df['hour'] = df.index.hour
        return df['hour'].between(16, 22)  # 美东时间下午4点至晚10点

# ===================
# 多资产联动分析 (网页1/9/10)
# ===================
class CrossAssetAnalyzer:
    def correlation_matrix(self, assets):
        """30日滚动相关系数矩阵 (网页10分析)"""
        return assets.pct_change().rolling(30).corr()
    
    def volatility_ratio_arbitrage(self, btc_vol, spx_vol):
        """波动率比值套利信号 (网页10策略)"""
        ratio = btc_vol / spx_vol
        return ratio > 3.5
    
    def blackswan_insurance(self, df, strike_multiplier=0.8):
        """黑天鹅保险策略 (网页10建议)"""
        return df['close'] * strike_multiplier

# ===================
# 执行示例
# ===================
if __name__ == "__main__":
    # 初始化各模块
    fetcher = DataFetcher()
    fed_analyzer = FedImpactAnalyzer()
    btc_strategy = BitcoinStrategy()
    
    # 获取数据
    sp500 = fetcher.get_stock_data('^GSPC')
    btc_data = fetcher.get_crypto_data()
    fed_rates = fetcher.get_fed_data()
    
    # 利率敏感度分析
    beta = fed_analyzer.rate_sensitivity_beta(sp500, fed_rates['rate'])
    print(f"标普500利率敏感Beta值: {beta:.2f}")
    
    # BTC策略信号生成
    btc_data['atr_channel'] = btc_strategy.atr_channel(btc_data)
    btc_data['breakout_signal'] = btc_strategy.volume_weighted_breakout(btc_data)
    btc_data['overnight_signal'] = btc_strategy.overnight_effect_signal(btc_data)
    
    # 多资产分析
    assets = pd.DataFrame({
        'SP500': sp500['Close'],
        'BTC': btc_data['close'],
        'Gold': fetcher.get_stock_data('GC=F')['Close']
    }).dropna()
    corr_matrix = CrossAssetAnalyzer().correlation_matrix(assets)