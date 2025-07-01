import pandas as pd

# Load your dataset (adjust path as needed)
df = pd.read_excel('../data/processed.xlsx')

# Group by CustomerId and create aggregate features
agg_features = df.groupby("CustomerId").agg(
    TotalTransactionAmount=('Amount', 'sum'),
    AverageTransactionAmount=('Amount', 'mean'),
    TransactionCount=('Amount', 'count'),
    StdDevTransactionAmount=('Amount', 'std'),
    MaxTransactionAmount=('Amount', 'max')
).reset_index()

# Optional: fill NaN std values (if only one transaction)
agg_features['StdDevTransactionAmount'] = agg_features['StdDevTransactionAmount'].fillna(0)

# Merge the aggregate features back to the original dataframe
df = df.merge(agg_features, on='CustomerId', how='left')

# Save the updated DataFrame to the same Excel file (overwrite)
df.to_excel('../data/processed.xlsx', index=False)