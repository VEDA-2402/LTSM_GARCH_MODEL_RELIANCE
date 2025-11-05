import yfinance as yf
import pandas as pd

# Define the ticker symbol for Reliance on NSE
ticker = 'RELIANCE.NS'

# Download historical data from 2008-01-01 to today
data = yf.download(ticker, start='2008-01-01', end=None)

# Extract the 'Close' price
reliance_close = data['Close']

# Save to CSV for reference
reliance_close.to_csv('reliance_closing_prices_2008_to_now.csv')

print("Data downloaded and saved to reliance_closing_prices_2008_to_now.csv")
print(reliance_close.head())

