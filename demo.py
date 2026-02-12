
import akshare as ak
import pandas as pd
import talib
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 2. 获取并准备数据
logging.info("1. 正在获取数据...")
symbol = "600036"
df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")

# 3. 特征工程
logging.info("2. 构建特征...")
df['RSI'] = talib.RSI(df['收盘'], timeperiod=14)
df['MA5'] = df['收盘'].rolling(5).mean()
df['MA10'] = df['收盘'].rolling(10).mean()
df['Return'] = df['收盘'].pct_change()

# 准备标签 (Y)：预测下一天的涨跌
df['Next_Return'] = df['收盘'].shift(-1).pct_change() # 仅用于计算收益
df['Target'] = (df['收盘'].shift(-1) > df['收盘']).astype(int)

# 4. 清洗数据
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

features = ['RSI', 'MA5', 'MA10', 'Return']
X = df[features]
y = df['Target']

# 5. 拆分训练集与测试集 (注意：时间序列不能随机打乱 shuffle=False)
# 我们用前 80% 的数据训练，后 20% 的数据测试
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
test_dates = df['日期'].iloc[split:] # 保存测试集日期用于画图
actual_returns = df['收盘'].pct_change().iloc[split:] # 测试集对应期间的股票真实涨跌幅

# 6. 训练模型
logging.info(f"3. 训练模型中 (训练集样本: {len(X_train)}, 测试集样本: {len(X_test)})...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 7. 测试模型结果
y_pred = model.predict(X_test)

logging.info("\n=== 模型性能报告 ===")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print("\n分类详细报告:")
print(classification_report(y_test, y_pred))

# 8. 简单策略回测：如果预测为 1，则持有股票；如果预测为 0，则空仓
# 策略每日收益 = 预测信号 * 实际次日涨跌幅
# 注意：这里的实际涨跌幅需要对齐
# 我们的 Target 是预测 shift(-1) 的，所以 y_pred 对应的是测试集当天的决策，作用于次日的收益
strategy_returns = y_pred * actual_returns

# 计算累计收益
cum_stock_return = (1 + actual_returns).cumprod()
cum_strategy_return = (1 + strategy_returns).cumprod()

logging.info("\n=== 策略回测结果 ===")
print(f"股票累计收益: {cum_stock_return.iloc[-1]:.2f}")
print(f"模型策略累计收益: {cum_strategy_return.iloc[-1]:.2f}")

# 9. 可视化
plt.figure(figsize=(12, 6))
plt.plot(test_dates, cum_stock_return, label="Stock Buy & Hold (招商银行)")
plt.plot(test_dates, cum_strategy_return, label="ML Strategy (Random Forest)")
plt.title(f"ML Strategy Backtest: {symbol}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()