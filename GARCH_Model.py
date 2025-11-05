import yfinance as yf
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# Download historical price data for Reliance
ticker = 'RELIANCE.NS'
data = yf.download(ticker, start='2008-01-01')

# Compute daily log returns
data['returns'] = 100 * data['Close'].pct_change().dropna()  # Percent returns
returns = data['returns'].dropna()

# Fit GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
garch_fit = model.fit(update_freq=5)
print(garch_fit.summary())

# Forecast volatility for the next 10 days
forecast = garch_fit.forecast(horizon=10)
print(forecast.variance[-1:])  # Forecasted variance

# Plot the volatility (conditional variance)
plt.figure(figsize=(10,6))
plt.plot(garch_fit.conditional_volatility, label='GARCH Volatility')
plt.title('GARCH(1,1) Conditional Volatility')
plt.legend()
plt.show()
