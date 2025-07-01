import pandas as pd

df = pd.read_excel('../data/processed.xlsx')

# Convert to datetime, force errors to NaT
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

# Remove timezone info if present
if pd.api.types.is_datetime64tz_dtype(df['TransactionStartTime']):
    df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)

# Extract features only for valid dates
df['TransactionHour'] = df['TransactionStartTime'].dt.hour
df['TransactionDay'] = df['TransactionStartTime'].dt.day
df['TransactionMonth'] = df['TransactionStartTime'].dt.month
df['TransactionYear'] = df['TransactionStartTime'].dt.year

df.to_excel('../data/processed.xlsx', index=False)