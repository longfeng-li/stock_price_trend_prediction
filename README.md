# stock_price_trend_prediction
We use RNN structure to predict the stock price trend for the next 21 days. Our input data is 10-year S&P index. This is not exactly our input data. We did some preprocessing. We use this raw data to calculate MACD, R parameter, Bias and EMA. And then we use these economic indicators as our input.
For these whole project, we use TensorFlow to implement RNN structure. The reason we use RNN, is because current stock price is related to previous stock price.
