from model import predict_stock

df, summary = predict_stock("AAPL", "2023-01-01", "2024-01-01")

print(summary)
print(df)
