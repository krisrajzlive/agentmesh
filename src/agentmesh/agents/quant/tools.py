"""Quantitative tools exposed to the Agent Framework agent."""

from __future__ import annotations

import statistics
from typing import Annotated

from agent_framework import tool


@tool
def descriptive_statistics(
    values: Annotated[list[float], "Numbers to summarise (at least one)."],
) -> dict[str, float]:
    """Mean, median, standard deviation, minimum and maximum of a list of numbers."""
    if not values:
        raise ValueError("values must not be empty")
    return {
        "count": float(len(values)),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


@tool
def loan_payment(
    principal: Annotated[float, "Amount borrowed."],
    annual_rate_percent: Annotated[float, "Nominal annual interest rate in percent."],
    years: Annotated[float, "Loan term in years."],
) -> dict[str, float]:
    """Fixed monthly payment, total paid and total interest for an amortising loan."""
    if principal <= 0 or years <= 0 or annual_rate_percent < 0:
        raise ValueError("principal and years must be positive and the rate non-negative")
    months = round(years * 12)
    monthly_rate = annual_rate_percent / 100 / 12
    if monthly_rate == 0:
        payment = principal / months
    else:
        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)
    total = payment * months
    return {
        "monthly_payment": round(payment, 2),
        "total_paid": round(total, 2),
        "total_interest": round(total - principal, 2),
    }


@tool
def compound_growth(
    principal: Annotated[float, "Starting amount."],
    annual_rate_percent: Annotated[float, "Annual growth rate in percent."],
    years: Annotated[float, "Number of years."],
    compounds_per_year: Annotated[int, "Compounding periods per year (12 = monthly)."] = 12,
) -> dict[str, float]:
    """Future value of an investment with periodic compounding."""
    if compounds_per_year <= 0 or principal < 0 or years < 0:
        raise ValueError("invalid compounding inputs")
    rate = annual_rate_percent / 100 / compounds_per_year
    future = principal * (1 + rate) ** (compounds_per_year * years)
    return {"future_value": round(future, 2), "gain": round(future - principal, 2)}
