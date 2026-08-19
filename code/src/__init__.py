"""Shared functions for the QSS 45 H-1B pipeline.

Imported at the top of every notebook so that logic is defined once and the
notebooks stay readable as narrative.
"""
from .config import (ROOT, DATA, LCA_RAW, LCA_SLIM, OUTPUT, FIGURES, TABLES,
                     USCIS_YEARS, LCA_YEARS, MODEL_YEARS)
from .names import canon
from .wages import annualise, PER_YEAR, clean_wage_level
from .diagnostics import describe_frame, merge_report
from .plotting import finish, save, C1, C2, C3, SEQ, INK, INK2, MUTED, SURFACE
