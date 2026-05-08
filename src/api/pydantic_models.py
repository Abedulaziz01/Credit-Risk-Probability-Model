from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int
    FraudResult: int
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: int
    StdDevTransactionAmount: float
    MaxTransactionAmount: float
    TotalValue: float
    AverageTransactionHour: float
    StdTransactionHour: float
    AverageTransactionDay: float
    StdTransactionDay: float
    AverageTransactionMonth: float
    StdTransactionMonth: float
    AverageTransactionYear: float
    recency_days: float
    frequency: int
    monetary: float


class PredictResponse(BaseModel):
    risk_probability: float
    risk_label: int
    credit_score: int
    recommended_loan_amount: float
    recommended_loan_duration_days: int
