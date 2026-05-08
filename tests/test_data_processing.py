import pandas as pd

from src.data_processing import add_temporal_features, assign_high_risk_label, build_rfm


def test_add_temporal_features():
    data = pd.DataFrame(
        {
            'TransactionId': [1, 2],
            'CustomerId': [101, 102],
            'TransactionStartTime': ['2025-05-01 12:00:00', '2025-05-02 18:30:00'],
            'Amount': [100, 200],
            'Value': [100, 200],
        }
    )
    result = add_temporal_features(data)
    assert 'TransactionHour' in result.columns
    assert result.loc[0, 'TransactionHour'] == 12
    assert result.loc[1, 'TransactionDay'] == 2


def test_assign_high_risk_label():
    data = pd.DataFrame(
        {
            'TransactionId': [1, 2, 3, 4, 5, 6],
            'CustomerId': [1, 1, 2, 2, 3, 3],
            'TransactionStartTime': [
                '2025-01-01 10:00:00',
                '2025-01-02 11:00:00',
                '2024-12-01 09:00:00',
                '2024-12-02 08:00:00',
                '2024-01-01 07:00:00',
                '2024-01-02 06:00:00',
            ],
            'Amount': [100, 110, 20, 25, 5, 5],
            'Value': [100, 110, 20, 25, 5, 5],
        }
    )
    rfm = build_rfm(data, snapshot_date=pd.to_datetime('2025-05-01'))
    out = assign_high_risk_label(rfm, random_state=42)
    assert set(out['is_high_risk'].unique()).issubset({0, 1})
    assert len(out) == 3
