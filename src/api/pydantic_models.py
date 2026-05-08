from pydantic import BaseModel


class PredictRequest(BaseModel):
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str
    FraudResult: int
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: int
    StdDevTransactionAmount: float
    MaxTransactionAmount: float
    TotalValue: float
    AverageTransactionHour: float
    AverageTransactionDay: float
    AverageTransactionMonth: float
    AverageTransactionYear: float
    recency_days: float
    frequency: int
    monetary: float


class PredictResponse(BaseModel):
    risk_probability: float
    risk_label: int
