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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Add the predictions to the test set for comparison
X_test['Actual_Pct_Diff'] = y_test
X_test['Predicted_Pct_Diff'] = y_pred

# Calculate the actual and predicted prices 2 weeks later based on percentage differences
X_test['Actual_Close_2w_later'] = X_test['Close'] * (1 + X_test['Actual_Pct_Diff'] / 100)
X_test['Predicted_Close_2w_later'] = X_test['Close'] * (1 + X_test['Predicted_Pct_Diff'] / 100)

# Display comparison
print(X_test[['Close', 'Actual_Close_2w_later', 'Predicted_Close_2w_later']].head(10))