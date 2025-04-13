economic_states = {
    '通胀上升+增长上升': ['商品', '周期股'],
    '通胀下降+增长上升': ['股票', '信用债'],
    '通胀上升+增长下降': ['黄金', '国债'],
    '通胀下降+增长下降': ['国债', '防御股']
}


import yfinance as yf
import pandas as pd
import riskfolio as rp

# 获取宏观经济数据（示例：中美10年期国债利差）
macro_data = yf.download(['^FVX', '^TNX'], start='2010-01-01')['Adj Close']
debt_spread = macro_data['^TNX'] - macro_data['^FVX']  # 美债-中债利差

# 债务周期判断函数
def debt_cycle_phase(gdp_growth, credit_growth):
    if credit_growth > gdp_growth * 1.5:
        return "债务扩张期"
    elif credit_growth < gdp_growth * 0.8:
        return "债务收缩期"
    else:
        return "平衡期"
    
# 配置四大类资产
assets = ['SPY', 'TLT', 'GLD', 'GSG']  # 美股/美债/黄金/大宗商品
returns = yf.download(assets, period='5y')['Adj Close'].pct_change().dropna()

# 风险平价优化
port = rp.Portfolio(returns=returns)
port.assets_stats(method_cov='hist', d=0.94)  # 使用EWMA协方差
port.rp_optimization(obj='RiskParity', rf=0, hist=True)
print(port.optimized_weights)


def bubble_detector(stock):
    data = yf.download(stock, period='3y')
    # 计算六大指标
    metrics = {
        '估值分位': data['Close'].rolling(250).quantile(0.8),
        '散户杠杆': (data['Volume'] * data['Close']).pct_change(21),
        '期权看涨比': None,  # 需接入期权数据API
        'IPO热度': len(yf.Ticker(stock).calendar),  # 示例简化
        '资本支出偏离': data['Close'].diff(60) / data['Close'].shift(60) - data['Volume'].pct_change(60)
    }
    return pd.DataFrame(metrics)

from pypfopt import HRPOpt
import yfinance as yf

# 获取多资产数据（含股债商品）
assets = ["SPY", "TLT", "GLD", "GSG"]
prices = yf.download(assets, start="2020-01-01", end="2025-04-01")["Adj Close"]

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

# 应用时变风险预算
hrp = HRPOpt(returns)
hrp.optimize(risk_budget=dynamic_risk_budget(hrp.returns))


# 分层风险平价回测
perf = hrp.portfolio_performance(verbose=True)

# 与传统方法对比
from pypfopt import RiskParityPortfolio
rp = RiskParityPortfolio(cov_matrix=cov_matrix)
rp.optimize()
print("传统风险平价:", rp.weights)

# 绘制有效前沿对比
ef = EfficientFrontier(None, cov_matrix)
ef.efficient_risk(target_risk=0.15)
ef.portfolio_performance(verbose=True)