"""Diagnostic printing.

The project rubric asks for diagnostic output before and after every data merge.
`describe_frame` prints the state of a table at a checkpoint; `merge_report`
prints what a join actually did, including the petition-weighted match rate,
which is the figure that matters for the models.
"""
import pandas as pd


def describe_frame(df, label, keys=None):
    """Print shape, key cardinality and missingness for a table."""
    print(f"--- {label} ---")
    print(f"    rows: {len(df):,}   columns: {df.shape[1]}")
    if keys:
        for k in keys:
            if k in df.columns:
                print(f"    distinct {k}: {df[k].nunique():,}")
    miss = df.isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        print(f"    columns with missing values: {len(miss)}"
              f" (worst: {miss.index[0]} at {miss.iloc[0]:.1%})")
    else:
        print("    no missing values")


def merge_report(merged, indicator_col, weight_col=None, by=None):
    """Report match rates after a left join.

    `indicator_col` is any column that comes only from the right-hand table, so
    a null means the row did not match. `weight_col` gives the petition-weighted
    rate; `by` breaks the rate down by a grouping column.
    """
    matched = merged[indicator_col].notna()
    print(f"    rows matched: {matched.sum():,} / {len(merged):,}"
          f"  ({matched.mean():.1%})")
    if weight_col:
        w = merged[weight_col]
        print(f"    weighted by {weight_col}: "
              f"{w[matched].sum() / w.sum():.1%}")
    if by:
        out = merged.groupby(by).apply(
            lambda g: pd.Series({
                "rows": len(g),
                "matched_pct": g[indicator_col].notna().mean() * 100,
                "weighted_pct": (g.loc[g[indicator_col].notna(), weight_col].sum()
                                 / g[weight_col].sum() * 100) if weight_col else None,
            }), include_groups=False)
        print(out.round(1).to_string())
    return matched
