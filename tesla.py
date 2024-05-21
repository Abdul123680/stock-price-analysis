import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Download Tesla stock data for the last 5 years
ticker = 'TSLA'
data = yf.download(ticker, period='5y')

# Calculate the 30-day moving average of the Close price
data['MA_30'] = data['Close'].rolling(window=30).mean()

# Calculate the difference between the Close price and the moving average
data['Diff_MA'] = data['Close'] - data['MA_30']

# Create a column that shows the Close price 2 weeks (10 trading days) later
data['Close_2w_later'] = data['Close'].shift(-10)

# Create a column to calculate the percentage difference between the Close price and the price 2 weeks later
data['Pct_Diff'] = ((data['Close_2w_later'] - data['Close']) / data['Close']) * 100

# Round all numerical columns to 2 decimal places
data = data.round(2)

# Drop rows with NaN values (due to rolling mean and shift operations)
data.dropna(inplace=True)

# Export the data to an Excel file
data.to_excel('tesla_stock_data.xlsx', index=True)

# Features and target variable
features = ['Close', 'MA_30', 'Diff_MA', 'Volume']
target = 'Pct_Diff'

X = data[features]
y = data[target]