import pandas as pd
import torch


# Add averages to datasets (only run once for train + test csv):
# Load the dataset
# data = pd.read_csv('Google stock prices/Google_Stock_Price_Train.csv')
# # Remove commas and convert to float for the necessary columns
# data[['Volume', 'Open', 'High', 'Low', 'Close']] = (
#     data[['Volume', 'Open', 'High', 'Low', 'Close']].replace(',', '', regex=True).astype(float)
# )
#
# # Calculate daily averages
# data['Average'] = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)
# data['Average'] = data['Average'].round(2)
# # Save the updated DataFrame back to a new CSV file
# data.to_csv('Google stock prices/Google_Stock_Price_Train.csv', index=False)

def prepare_data():
    train_data = pd.read_csv('Google stock prices/Google_Stock_Price_Train.csv')
    test_data = pd.read_csv('Google stock prices/Google_Stock_Price_Test.csv')

    # Extract date and average columns
    train_data = train_data[['Date', 'Average']]
    test_data = test_data[['Date', 'Average']]

    # Convert date to datetime
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    test_data['Date'] = pd.to_datetime(test_data['Date'])

    # Set date as index
    train_data.set_index('Date', inplace=True)
    test_data.set_index('Date', inplace=True)

    # Normalize the data
    train_data['Average'] = (train_data['Average'] - train_data['Average'].mean()) / train_data['Average'].std()
    test_data['Average'] = (test_data['Average'] - test_data['Average'].mean()) / test_data['Average'].std()

    return train_data, test_data


def create_dataset(data):
    # Convert the index (dates) to numeric format (Unix timestamp)
    X = (data.index.astype('int64') // 10**9).values.reshape(-1, 1)  # Convert to seconds
    y = data['Average'].values.reshape(-1, 1)  # Averages as output
    return torch.FloatTensor(X), torch.FloatTensor(y)

