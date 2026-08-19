"""Wage normalisation for the DOL LCA data.

Two problems make raw LCA wages incomparable. Pay is quoted on five different
bases (hourly through yearly), so everything must be annualised before offered
and prevailing wages can be divided. And the prevailing-wage level is written
"Level I" in FY2017-FY2019 but "I" from FY2020, so the label needs normalising
before it can be pooled across years.
"""
import numpy as np
import pandas as pd

# Periods per year for each unit-of-pay label DOL uses. 2080 = 40 hours x 52 weeks.
PER_YEAR = {
    "YEAR": 1, "MONTH": 12, "BI-WEEKLY": 26, "BIWEEKLY": 26,
    "WEEK": 52, "HOUR": 2080,
}

# Plausible annual salary band. Anything outside it is a data-entry error --
# typically an annual figure typed into the hourly field -- and would otherwise
# produce wage ratios in the thousands.
WAGE_MIN, WAGE_MAX = 15_000, 2_000_000


def annualise(amount, unit):
    """Convert a pay amount to an annual figure using its unit-of-pay label."""
    periods = unit.astype(str).str.upper().str.strip().map(PER_YEAR)
    return amount * periods


def clean_wage_level(series):
    """Normalise PW_WAGE_LEVEL to the bare numeral (I, II, III, IV, or NaN)."""
    lvl = (series.astype(str).str.upper()
           .str.replace("LEVEL", "", regex=False).str.strip())
    return lvl.where(lvl.isin(["I", "II", "III", "IV"]), np.nan)
