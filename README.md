# Predictability, Wages, and Gatekeeping in the H-1B Program

**QSS 45 — Artificial Intelligence & Machine Learning for Social Science**

Dartmouth College, Summer 2026 · Muhammad Moiz

**Project website:** https://qss45-h1b.vercel.app

**Final paper:** [PDF](paper/H1B_Final_Paper.pdf) · [LaTeX source](paper/H1B_Final_Paper.tex)

## Research question and headline result

Among initial H-1B petitions filed from FY2017 to FY2022, how well do sponsor history,
industry, size, geography, wages, occupation, and fiscal year predict a sponsor's initial
denial rate, both for sponsors not seen in training and for a fiscal year not seen in
training?

The answer depends on the validation target. On held-out sets containing entirely unseen
employers but familiar fiscal years, LightGBM reaches petition-weighted $R^2=0.414$ for
the denial rate and AUC $=0.798$ for any denial, averaged over ten sponsor-disjoint splits
(seeds 45 to 54, `output/tables/model_metrics_seeds.csv`). Wage and occupation features
improve those means to $R^2=0.512$ and AUC $=0.855$ on the exact-match DOL subsample.
The single seed 45 split reported in the tables below gives 0.419, 0.797, 0.528, and 0.854. Forward
transfer is poor: models trained through FY2021 and tested on FY2022 have negative
petition-weighted $R^2$ because the aggregate denial rate fell from 15.0% in the training
period to 2.2% in FY2022.

## Repository layout

```text
code/                         sequential, executed notebooks
  src/                        shared functions imported at notebook start
  00_pull.ipynb
  01_extract_lca.ipynb
  02_merge.ipynb
  03_eda.ipynb
  04_models.ipynb
  temporal_rank_check.py      rank metrics for the FY2022 temporal test (Spearman, AUC, deciles)
data/                         USCIS inputs and reproducible derived panels
output/
  figures/                    figures in 300-dpi PNG and vector PDF
  tables/                     diagnostics, estimates, metrics, and importance
paper/                        final paper PDF and LaTeX source
requirements.txt
```

There are no spaces in repository filenames. All paths are derived from the repository
root in `code/src/config.py`; no machine-specific path is hardcoded. Functions are defined
in `code/src/` or in the functions cell at the beginning of the relevant notebook.

## Numbered notebooks

Run the notebooks in numeric order. Each documents its inputs, function, and outputs.

| # | Notebook | Inputs | Function | Outputs |
|---|---|---|---|---|
| 00 | `00_pull.ipynb` | USCIS download URLs; input manifests for both agencies | Download and checksum seven USCIS Employer Data Hub files; verify the manual DOL acquisition | `data/uscis_2017.csv` … `data/uscis_2023.csv` |
| 01 | `01_extract_lca.ipynb` | 15 DOL disclosure workbooks in `data/lca/` | Stream three historical workbook layouts and retain the required columns | `data/lca_slim/lca_*.csv` |
| 02 | `02_merge.ipynb` | USCIS annual CSVs; slim LCA CSVs | Build canonical employer-years, backward-looking history, cleaned wage measures, and the exact employer-year agency join | `data/analysis_panel.csv`, `data/lca_employer_year.csv`, `data/analysis_panel_wages.csv`; merge diagnostic tables |
| 03 | `03_eda.ipynb` | `data/analysis_panel_wages.csv` | Describe outcomes, sponsors, sectors, match selection, and wage patterns | figures 01–05 and 10–12; `output/tables/eda_*.csv` |
| 04 | `04_models.ipynb` | `data/analysis_panel_wages.csv` | Fit WLS, logistic regression, and LightGBM with sponsor-disjoint validation; run temporal validation and SHAP | figures 06–09 and 13–14; model, seed-repetition, robustness, coefficient, split, and importance tables |

### Merge diagnostics

`02_merge.ipynb` performs four joins. Each operation has an immediately preceding
diagnostic for both inputs and an immediately following diagnostic for output size,
key uniqueness, and match status:

1. petition totals + dominant worksite descriptors;
2. employer-year outcomes + exact prior-year sponsor history;
3. DOL employer-year measures + position-weighted modal occupation; and
4. USCIS employer-years + DOL employer-years.

The notebook asserts one row per canonical `(key, fy)` after every relevant join. It also
asserts that aggregation preserves all four published USCIS count totals exactly.

## Data and processing

| Source or derived file | Coverage | Rows | Distributed here? |
|---|---:|---:|---|
| USCIS H-1B Employer Data Hub | FY2017–FY2023 | 374,253 raw worksite rows | yes |
| DOL certified H-1B LCA records | FY2017–FY2022 | 3,750,059 applications | raw workbooks: no |
| `analysis_panel.csv` | FY2017–FY2023 | 176,404 canonical employer-years | yes |
| `lca_employer_year.csv` | FY2017–FY2022 | 329,829 canonical employer-years | yes |
| `analysis_panel_wages.csv` | FY2017–FY2023 | 176,404 × 38 | yes |

The USCIS unit is one canonical employer in one fiscal year. Counts are summed across
worksites; the state and industry attached to the largest worksite cell supply descriptive
labels. Missing employer names receive unique placeholders rather than being dropped.
On the DOL side, 61 certified LCA rows with a blank employer name cannot form a key and
are dropped before the employer-year aggregation. The FY2021 Q2 and Q3 DOL workbooks are
cumulative year-to-date files (each repeats the earlier quarters of FY2021), so the FY2021
stack contains roughly 278,000 repeated certified rows. Repetition leaves employer-year
medians and shares unchanged but inflates FY2021 filing and position counts; the rows are
not deduplicated, and the paper reports this as a limitation. The USCIS aggregation itself preserves
757,806 initial approvals, 107,839 initial denials, 1,884,861 continuing approvals, and
122,194 continuing denials exactly. The committed analysis panel then retains the 176,404
employer-years with at least one initial petition. It therefore preserves both initial
counts but, by design, excludes continuation-only records; its continuing counts are
1,637,883 approvals and 107,629 denials.

The DOL pipeline retains certified H-1B LCAs, annualizes all wage units, rejects implausible
annual wage fields before constructing ratios, and keeps missing yes/no values missing.
The share at prevailing-wage levels I or II and the modal occupation are weighted by the
number of sponsored positions. Application-level wage measures use medians to limit the
influence of extreme salaries.

The agencies publish no shared employer ID. Their records are linked on fiscal year and a
fixed, documented name canonicalization in `code/src/names.py`. In FY2017–FY2022, exact
matching recovers 119,185 of 165,593 USCIS employer-years (72.0%) and covers 87.0% of
initial petitions. Selection is size-related: match rates range from about 63–66% among
one-petition cells to 88.8% in the largest decile. The wage comparison therefore holds
the matched rows and data partitions fixed across specifications.

## Model design

The initial-denial rate is a proportion with unequal denominators. Rate models use initial
petition counts as weights, and held-out $R^2$ uses the same weights. Linear estimates are
WLS with confidence intervals clustered by canonical employer. The parallel any-denial
task uses logistic regression and LightGBM classification with ROC AUC.

For the main validation, employers are assigned to disjoint train (56%), validation
(14%), and test (30%) partitions with seed 45. Employer overlap is asserted to be zero.
LightGBM early stopping uses validation rows only; the selected tree count is then refit
on the development set before the test set is scored once. `prior_active_years` and every
other history feature use only years earlier than the outcome.

The temporal test uses FY2017–FY2020 for LightGBM fitting, FY2021 for selecting tree count,
then refits through FY2021 and scores FY2022. Fiscal year is excluded because a dummy for
an unseen year cannot be extrapolated.

## Executed results

The paper, repository outputs, and project website report the same canonical
employer-year pipeline and validation design.

All model-window employer-years, unseen-employer test set:

| Model | Target | Metric | Held-out score |
|---|---|---|---:|
| WLS | Initial denial rate | petition-weighted $R^2$ | 0.2957 |
| LightGBM | Initial denial rate | petition-weighted $R^2$ | **0.4187** |
| Logistic regression | Any initial denial | ROC AUC | 0.7884 |
| LightGBM | Any initial denial | ROC AUC | **0.7973** |

Exact-match DOL subsample, identical employers and partitions:

| Model family | Features | Petition-weighted $R^2$ | AUC |
|---|---|---:|---:|
| WLS / logistic | baseline | 0.3530 | 0.8260 |
| WLS / logistic | + wage and occupation | 0.3821 | 0.8421 |
| LightGBM | baseline | 0.4771 | 0.8356 |
| LightGBM | + wage and occupation | **0.5283** | **0.8537** |

Forward validation, train through FY2021 and test FY2022:

| Model | Petition-weighted $R^2$ |
|---|---:|
| WLS | −1.7203 |
| LightGBM | −1.2398 |

Repeating the employer split over ten seeds (45 to 54) gives the following means and standard
deviations (`output/tables/model_metrics_seeds.csv`):

| Sample | WLS $R^2$ | LightGBM $R^2$ | Logistic AUC | LightGBM AUC |
|---|---:|---:|---:|---:|
| All employer-years | 0.3031 (0.0231) | 0.4140 (0.0223) | 0.7895 (0.0022) | 0.7984 (0.0019) |
| Matched, baseline | 0.3346 (0.0304) | 0.4583 (0.0280) | 0.8281 (0.0025) | 0.8369 (0.0021) |
| Matched, with wages | 0.3635 (0.0280) | 0.5122 (0.0249) | 0.8443 (0.0027) | 0.8548 (0.0020) |

The levels move by about 0.02 to 0.03 across splits, but the within-seed gaps are much
tighter. LightGBM beats WLS on $R^2$ by 0.1109 on average (SD 0.0072, range 0.0986 to
0.1230) and beats logistic regression on AUC by 0.0088 (SD 0.0008). Adding wage features
raises LightGBM $R^2$ by 0.0539 (SD 0.0091) and AUC by 0.0179 (SD 0.0007). Every paired
gap is positive in all ten seeds.

Robustness checks on the seed 45 split (`output/tables/robustness.csv`):

| Specification | Sample | Metric | Score |
|---|---|---|---:|
| Lagged-only features, WLS | all | petition-weighted $R^2$ | 0.2892 |
| Lagged-only features, LightGBM | all | petition-weighted $R^2$ | 0.3643 |
| Lagged-only features, logistic | all | ROC AUC | 0.7487 |
| Lagged-only features, LightGBM classifier | all | ROC AUC | 0.7600 |
| Fractional logit, main features | all | petition-weighted $R^2$ | 0.3408 |
| Fractional logit, with wage features | matched | petition-weighted $R^2$ | 0.4843 |

The lagged-only set uses only the sponsor's own prior-year record (previous initial count,
previous continuing share, previous denial rate, prior active years, repeat flag). The
fractional logit is a binomial GLM on the rate with petition weights and employer-clustered
standard errors; it fixes the 10.9% of WLS test predictions that fall outside 0 and 1.

These estimands are intentionally separate. The sponsor-disjoint test asks whether the
model ranks new employers within familiar years; the temporal test asks whether pre-2022
patterns recover a new year's outcome level.

## Reproduce

There are two reproducibility paths. A results-only rerun uses the committed derived panel
and needs no manual downloads; after creating the environment, run notebooks 03 and 04.
That regenerates every reported figure, estimate, metric, and robustness table.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd code
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 03_eda.ipynb
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04_models.ipynb
```

An end-to-end rebuild additionally requires the 15 raw DOL workbooks listed below. They
total roughly 1.8 GB and are excluded from Git. After placing them in `data/lca/`, run all
five notebooks in numeric order with the same `nbconvert` command. Notebook 01 takes about
20–40 minutes on a cold run; notebooks 00 and 01 verify their inputs before processing.

The USCIS files and
the DOL workbooks used here were downloaded on 2026-08-28 from the
[DOL OFLC performance data page](https://www.dol.gov/agencies/eta/foreign-labor/performance)
and the [USCIS H-1B Employer Data Hub archive](https://www.uscis.gov/archive/h-1b-employer-data-hub-files).
`data/uscis_manifest.csv` records the source URL, row count, byte size, and SHA-256 checksum
for every distributed USCIS file; notebook 00 verifies each file against it. Place the 15
DOL workbooks in `data/lca/` under these names (the inventory in `00_pull.ipynb` checks
them). `data/lca_manifest.csv` records the exact byte size and SHA-256 checksum of every
workbook used in the submitted analysis:

```text
LCA_FY2017.xlsx  LCA_FY2018.xlsx  LCA_FY2019.xlsx
LCA_FY2020_Q1.xlsx  LCA_FY2020_Q2.xlsx  LCA_FY2020_Q3.xlsx  LCA_FY2020_Q4.xlsx
LCA_FY2021_Q1.xlsx  LCA_FY2021_Q2.xlsx  LCA_FY2021_Q3.xlsx  LCA_FY2021_Q4.xlsx
LCA_FY2022_Q1.xlsx  LCA_FY2022_Q2.xlsx  LCA_FY2022_Q3.xlsx  LCA_FY2022_Q4.xlsx
```

Notebook 01 reconstructs the ignored slim extracts. The derived panels committed here
allow notebooks 03 and 04 to run without the raw workbooks.

## Principal references

Borjas (2026); Bourveau et al. (2025); Costa and Hira (2020); Glennon (2024);
Ke and Qiao (2019); Lipsky (1980); Mayda et al. (2020); Peri, Shih, and Sparber (2015);
Piore (1979); Spence (1973). Full citations appear in the paper.
