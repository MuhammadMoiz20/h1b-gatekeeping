# Predictability, Wages, and Gatekeeping in the H-1B Program

**QSS 45 — Artificial Intelligence & Machine Learning for Social Science**
Dartmouth College, Summer 2026 · Muhammad Moiz

Among initial H-1B petitions filed from FY2017 onward, how well do employer sponsorship
history, industry, firm size, geography, offered wage relative to the prevailing wage, and
policy year predict a USCIS denial?

**Headline finding.** Within a policy regime, denial is modestly predictable (AUC ≈ 0.80).
Across one, it is not: a model trained on FY2017–FY2021 and tested on FY2022 posts a
*negative* R². The binding variable is administrative guidance, not the employer.

---

## Repository layout

```
code/                 numbered sequential notebooks + shared source module
  src/                functions imported by every notebook
  00_pull.ipynb
  01_extract_lca.ipynb
  02_merge.ipynb
  03_eda.ipynb
  04_models.ipynb
data/                 USCIS source files and the derived analysis panels
output/
  figures/            14 figures (PNG)
  tables/             model comparison and feature-importance tables
requirements.txt
```

## Notebooks

Run in order. Each notebook reads what the previous one wrote.

| # | Notebook | Inputs | Function | Outputs |
|---|---|---|---|---|
| 00 | `00_pull.ipynb` | none | Download the seven USCIS Employer Data Hub files; verify row counts and header drift; check for the fifteen manually-acquired DOL LCA workbooks | `data/uscis_2017.csv` … `data/uscis_2023.csv` |
| 01 | `01_extract_lca.ipynb` | `data/lca/LCA_FY*.xlsx` (15 workbooks, ~1.8 GB) | Stream each workbook and keep the ~20 needed columns, reconciling three different column layouts (52 / 260 / 96 columns) | `data/lca_slim/lca_*.csv` (15 files, ~600 MB) |
| 02 | `02_merge.ipynb` | `data/uscis_*.csv`, `data/lca_slim/lca_*.csv` | Collapse worksites to employer-years, build backward-looking history features, clean and annualise LCA wages, aggregate to employer-year, and join the two sources on a canonicalised employer name. Four merges, each with before-and-after diagnostics | `data/analysis_panel.csv`, `data/lca_employer_year.csv`, `data/analysis_panel_wages.csv` |
| 03 | `03_eda.ipynb` | `data/analysis_panel_wages.csv` | Describe the outcome, predictors and wage block | `output/figures/01`–`05`, `10`–`12`; `output/tables/eda_*.csv` |
| 04 | `04_models.ipynb` | `data/analysis_panel_wages.csv` | OLS / logistic / LightGBM on the denial rate and on any-denial; temporal robustness test; SHAP; with-vs-without wage comparison | `output/figures/06`–`09`, `13`–`14`; `output/tables/model_comparison*.csv`, `*_importance*.csv` |

### Shared module — `code/src/`

Functions are defined here and imported at the top of each notebook rather than
redefined per notebook.

| File | Provides |
|---|---|
| `config.py` | Path resolution from the repository root; fiscal-year constants |
| `names.py` | `canon()` — employer-name canonicalisation, the basis of the USCIS↔DOL join |
| `wages.py` | `annualise()`, `clean_wage_level()`, plausible-wage bounds |
| `diagnostics.py` | `describe_frame()`, `merge_report()` — the before/after merge diagnostics |
| `plotting.py` | Shared matplotlib style, `finish()`, `save()` |

## Data

| Source | Coverage | Rows | In this repo? |
|---|---|---|---|
| USCIS H-1B Employer Data Hub | FY2017–FY2023 | 374,253 raw | yes, `data/uscis_*.csv` |
| DOL LCA disclosure workbooks | FY2017–FY2022 | 3,750,059 certified H-1B | **no** — ~1.8 GB, see below |
| Derived: USCIS employer-year panel | FY2017–FY2023 | 184,301 | yes, `data/analysis_panel.csv` |
| Derived: LCA employer-year aggregate | FY2017–FY2022 | 330,042 | yes, `data/lca_employer_year.csv` |
| Derived: joined analysis panel | FY2017–FY2023 | 184,301 × 36 | yes, `data/analysis_panel_wages.csv` |

**The raw DOL workbooks are not in this repository.** They total ~1.8 GB, and
`www.dol.gov` returns HTTP 403 to programmatic requests while the old OFLC Data Center
domain no longer resolves — so they cannot be fetched by script. Download the fifteen
files from the [DOL OFLC performance data page](https://www.dol.gov/agencies/eta/foreign-labor/performance)
and place them in `data/lca/` using the names `00_pull.ipynb` lists. Notebooks 02–04 run
without them, from the derived panels committed here.

## The merge

USCIS and DOL publish **no shared employer identifier**, so the join is on the employer
name as typed — and the two agencies type it differently (`UNIV OF N CAROLINA AT
CHARLOTTE` vs `UNIVERSITY OF NORTH CAROLINA AT CHARLOTTE`). Both sides pass through the
same `canon()` reduction: uppercase, strip punctuation, map long forms onto short forms,
drop corporate suffixes.

**Match rate: 72.6% of employer-years, covering 86.0% of initial petitions** (FY2017–FY2022;
FY2023 matches at 0% by construction since DOL coverage stops at FY2022).

The gap between those two figures is the caveat that matters: match rate is far higher
for large sponsors (86-89% in the top two deciles) than for single-petition ones (59-68%),
though not monotonically so, and the wage subsample therefore tilts toward larger sponsors.
`04_models` therefore refits the no-wage baseline on the matched subsample, so the
with-versus-without comparison isolates the features rather than the sample.

## Results

Full modelling window, no wage features:

| Model | Target | Metric | Held-out |
|---|---|---|---|
| OLS | initial denial rate | R² | 0.105 |
| LightGBM | initial denial rate | R² | 0.147 |
| Logistic | any initial denial | AUC | 0.790 |
| LightGBM | any initial denial | AUC | 0.802 |

Matched subsample, with and without the DOL wage block (identical rows both ways):

| Model | Features | R² | AUC |
|---|---|---|---|
| OLS | baseline | 0.115 | 0.824 |
| OLS | + wages | 0.143 | 0.839 |
| LightGBM | baseline | 0.160 | 0.833 |
| LightGBM | + wages | **0.205** | **0.850** |

Temporal split, train ≤FY2021 → test FY2022, year feature dropped:
**OLS R² = −0.618, LightGBM R² = −0.709.**

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name qss45-h1b

cd code
jupyter nbconvert --to notebook --execute --inplace 00_pull.ipynb
jupyter nbconvert --to notebook --execute --inplace 01_extract_lca.ipynb   # slow; needs data/lca/
jupyter nbconvert --to notebook --execute --inplace 02_merge.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_models.ipynb
```

All paths resolve from the repository root, so the notebooks run from any checkout
location. Random seed is 45 throughout.

## References

Borjas, G. (2026). Wage effects of high-skilled immigration.
Bourveau, T., et al. (2025). H-1B wage discounts within firms.
Costa, D., & Hira, R. (2020). *H-1B visas and prevailing wage levels*. EPI.
Glennon, B. (2024). How do restrictions on high-skilled immigration affect offshoring?
Ke, R., & Qiao, S. (2021). Predicting LCA certification outcomes.
Lipsky, M. (1980). *Street-Level Bureaucracy*.
Mayda, A. M., et al. (2020). The effect of the H-1B quota on employment.
Peri, G., Shih, K., & Sparber, C. (2015). STEM workers, H-1B visas, and productivity.
Piore, M. (1979). *Birds of Passage*.
Spence, M. (1973). Job market signaling.
