import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# 加载GDP时间序列数据
data = pd.read_csv('gdp.csv', parse_dates=['Date'], index_col=['Date'])

# 转换为时间序列数据
ts = data['GDP']

# 绘制时间序列图
plt.plot(ts)
plt.title('GDP Time Series')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()

# 绘制ACF和PACF图
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
sm.graphics.tsa.plot_acf(ts, lags=40, ax=ax[0])
sm.graphics.tsa.plot_pacf(ts, lags=40, ax=ax[1])
plt.show()

# 对时间序列数据进行差分
diff = ts.diff(periods=1).dropna()

# 绘制差分后的时间序列图
plt.plot(diff)
plt.title('Differenced GDP Time Series')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()

# 绘制差分后时间序列的ACF和PACF图
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
sm.graphics.tsa.plot_acf(diff, lags=40, ax=ax[0])
sm.graphics.tsa.plot_pacf(diff, lags=40, ax=ax[1])
plt.show()

# 使用自动ARIMA模型确定最佳参数
arima_model = ARIMA(ts, order=(1,1,1)).fit()

# 打印模型参数
print(arima_model.summary())

# 预测未来一段时间的GDP值
forecast = arima_model.forecast(steps=12)

# 绘制预测结果
plt.plot(ts)
plt.plot(forecast[0])
plt.title('GDP Forecast')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()
