"""Request/response schemas for the credit-default-risk inference API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

EmploymentType = Literal["Salaried", "Self-employed", "Unemployed", "Retired"]


class PredictionRequest(BaseModel):
    """Raw applicant features. Optional fields may be omitted or null — the
    underlying pipeline imputes missing numeric values and handles missing/
    unseen categories the same way it was trained to (see model-tuning notebook).
    """

    age: Optional[float] = Field(
        default=None, ge=18, le=100, description="Applicant age in years"
    )
    income: Optional[float] = Field(
        default=None, ge=0, description="Annual income"
    )
    loan_amount: float = Field(
        ..., gt=0, description="Requested loan amount"
    )
    credit_score: Optional[float] = Field(
        default=None, ge=300, le=850, description="Credit score (300-850)"
    )
    debt_to_income: Optional[float] = Field(
        default=None, ge=0, le=1, description="Debt-to-income ratio, 0-1"
    )
    employment_type: Optional[EmploymentType] = Field(
        default=None, description="Employment status"
    )
    state: str = Field(
        ..., min_length=2, max_length=10, description="State code, e.g. 'ST03'"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "income": 55000,
                "loan_amount": 15000,
                "credit_score": 680,
                "debt_to_income": 0.25,
                "employment_type": "Salaried",
                "state": "ST03",
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(
        ..., description="0 = no default predicted, 1 = default predicted"
    )
    probability_default: float = Field(
        ..., description="Model's predicted probability of default, 0-1"
    )
    risk_label: str = Field(
        ..., description="Human-readable risk bucket: Low / Medium / High"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
