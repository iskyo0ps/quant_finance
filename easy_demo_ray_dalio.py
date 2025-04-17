economic_states = {
    '通胀上升+增长上升': ['商品', '周期股'],
    '通胀下降+增长上升': ['股票', '信用债'],
    '通胀上升+增长下降': ['黄金', '国债'],
    '通胀下降+增长下降': ['国债', '防御股']
}


import os
import pandas as pd
import yfinance as yf
import riskfolio as rp

# Function to load data from local or download if not available
def load_or_download_data(filename, tickers, start=None, end=None, period=None):
    if os.path.exists(filename):
        print(f"Loading data from local file: {filename}")
        return pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        print(f"Downloading data for: {tickers}")
        if period:
            data = yf.download(tickers, period=period)['Adj Close']
        else:
            data = yf.download(tickers, start=start, end=end)['Adj Close']
        data.to_csv(filename)
        return data

# 获取宏观经济数据（示例：中美30年期国债利差）
macro_data_file = 'macro_data.csv'
macro_data = load_or_download_data(macro_data_file, ['^FVX', '^TNX'], start='1995-01-01')

# 计算美债-中债利差
debt_spread_file = 'debt_spread.csv'
if os.path.exists(debt_spread_file):
    print(f"Loading debt spread from local file: {debt_spread_file}")
    debt_spread = pd.read_csv(debt_spread_file, index_col=0, parse_dates=True)
else:
    debt_spread = macro_data['^TNX'] - macro_data['^FVX']  # 美债-中债利差
    debt_spread.to_csv(debt_spread_file)

# 配置四大类资产
assets = ['SPY', 'TLT', 'GLD', 'GSG']
returns_file = 'asset_returns.csv'
returns = load_or_download_data(returns_file, assets, period='30y').pct_change().dropna()

# 风险平价优化
port = rp.Portfolio(returns=returns)
port.assets_stats(method_cov='hist', d=0.94)  # 使用EWMA协方差矩阵
port.rp_optimization(obj='RiskParity', rf=0, hist=True)
print(port.optimized_weights)

# 获取多资产数据（含股债商品）
prices_file = 'prices.csv'
prices = load_or_download_data(prices_file, assets, start="1995-01-01", end="2025-04-18")

# 创建HRP优化器
hrp = HRPOpt(returns=prices.pct_change().dropna())

# 改进点：使用Ward方差最小化聚类法
hrp.optimize(linkage_method='ward')
print("分层风险平价权重:", hrp.clean_weights())

# 可视化树状图
hrp.plot_clusters()


from pypfopt import risk_models

# Ledoit-Wolf 收缩估计（改进协方差矩阵稳定性）
cov_matrix = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

# 结合半定规划优化
hrp = HRPOpt(covariance=cov_matrix, returns=prices.pct_change())
hrp.optimize()


from pypfopt import EfficientFrontier
from pypfopt import objective_functions

# 创建风险平价与均值方差混合模型
ef = EfficientFrontier(None, cov_matrix)  # 不依赖收益率预测
ef.add_objective(objective_functions.risk_parity)

# 添加流动性约束（假设流动性数据）
liquidity = {"SPY":0.9, "TLT":0.7, "GLD":0.6, "GSG":0.5} 
ef.add_constraint(lambda w: sum([w[i]*liquidity[asset] for i,asset in enumerate(assets)]) >= 0.8)

# 求解优化问题
weights = ef.nonconvex_objective(objective_functions.sharpe_ratio, weights_sum_to_one=True)


# 定义随时间变化的风险预算
def dynamic_risk_budget(returns):
    # 基于波动率regime调整预算
    recent_vol = returns.iloc[-60:].std()
    return recent_vol / recent_vol.sum()

from pypfopt import RiskParityPortfolio
from pypfopt.hierarchical_portfolio import HRPOpt


# 应用时变风险预算
hrp = HRPOpt(returns)
hrp.optimize(risk_budget=dynamic_risk_budget(hrp.returns))


# 分层风险平价回测
perf = hrp.portfolio_performance(verbose=True)

# 与传统方法对比

rp = RiskParityPortfolio(cov_matrix=cov_matrix)
rp.optimize()
print("传统风险平价:", rp.weights)

# 绘制有效前沿对比
ef = EfficientFrontier(None, cov_matrix)
ef.efficient_risk(target_risk=0.15)
ef.portfolio_performance(verbose=True)