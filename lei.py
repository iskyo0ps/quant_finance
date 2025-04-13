import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
  
# 模拟数据（在实际应用中应该从可靠的经济数据库获取）  
np.random.seed(42)  # 为了获得可重复结果  
years = np.arange(2000, 2023)  
gdp_growth = np.random.uniform(2, 7, len(years))  # 模拟GDP增长率  
inflation_rate = np.random.uniform(1, 3, len(years))  # 模拟通货膨胀率  
interest_rate = np.random.uniform(0.5, 5, len(years))  # 模拟利率  
  
# 创建数据框  
df = pd.DataFrame({  
    'Year': years,  
    'GDP Growth': gdp_growth,  
    'Inflation Rate': inflation_rate,  
    'Interest Rate': interest_rate  
})  
  
# 市场分析函数  
def analyze_market(data):  
    # 短期债务周期（例如5年移动平均）  
    data['Short-term Debt Cycle'] = data['GDP Growth'].rolling(window=5).mean()  
    # 长期债务周期（例如10年移动平均）  
    data['Long-term Debt Cycle'] = data['GDP Growth'].rolling(window=10).mean()  
  
    # 简单的市场状态逻辑示例  
    conditions = [  
        (data['GDP Growth'] > data['GDP Growth'].mean()) & (data['Inflation Rate'] < data['Inflation Rate'].mean()),  
        (data['GDP Growth'] < data['GDP Growth'].mean()) & (data['Inflation Rate'] > data['Inflation Rate'].mean())  
    ]  
    choices = ['Expansion', 'Contraction']  
    data['Market State'] = np.select(conditions, choices, default='Stable')  
      
    return data  
  
# 运行市场分析  
df = analyze_market(df)  
  
# 可视化  
plt.figure(figsize=(14, 7))  
plt.plot(df['Year'], df['GDP Growth'], label='GDP Growth', marker='o')  
plt.plot(df['Year'], df['Short-term Debt Cycle'], label='Short-term Debt Cycle', linestyle='--')  
plt.plot(df['Year'], df['Long-term Debt Cycle'], label='Long-term Debt Cycle', linestyle='--')  
plt.xlabel('Year')  
plt.ylabel('Rate (%)')  
plt.title('Market Analysis Based on Ray Dalio\'s Principles')  
plt.legend()  
plt.grid(True)  
plt.show()  
  
# 打印分析结果  
print(df[['Year', 'GDP Growth', 'Inflation Rate', 'Interest Rate', 'Market State']])  
