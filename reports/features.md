# Features — step 2

Produced by `src/features.py` (`python -m src.features`) on 2026-10-29 from `data/raw/train.csv` (real L&T data). IV and bins are fitted on the Aug–Sep training window only; PSI compares the training window with October (out-of-time test) and with test.csv (November, unlabelled).

## Splits

| split | rows | default rate |
|:--|--:|--:|
| fit | 107,832 | 20.4% |
| valid | 26,958 | 20.2% |
| oot | 98,364 | 23.5% |

## Decisions

- `bureau_score` is missing for score 0 *and* for the 'Not Scored' placeholder codes 11–18; `not_scored` flags the latter. The headline no-score subgroup stays `PERFORM_CNS.SCORE == 0`; `no_usable_score` (both groups) is reported alongside.
- Protected attributes `age`, `age_band`, `State_ID` are in the table for fairness only and are **not** model inputs (CLAUDE.md rule 3). The guide lists `age` and `state_te` as features; CLAUDE.md takes precedence.
- `Employee_code_ID` is not used (CLAUDE.md rule 4). `disb_month` is not built: the training window has only Aug and Sep, so October would be an unseen value.
- `score_desc_grp` keeps each 'Not Scored' reason as its own group (thin history, not enough info, inactive, no updates in 36 months, guarantor only), since the reason carries risk information.
- Winsorising (`src/load.py`) clips only PRI/SEC *amount* columns, at the 1st/99th percentile of their non-zero values. Clipping every PRI/SEC column at plain percentiles, as the guide's snippet does, made `SEC.OVERDUE.ACCTS` constant and capped co-applicant amounts at ₹50,000, because 98–99% of SEC values are 0.
- `new_emi_est` assumes 1.5% a month over 36 months.
- IDs are target-encoded out-of-fold (5 folds, smoothing m = 50) on the training window; October and November use the full training-window mapping. `*_n` columns are loan counts per ID in the training window.

## Dropped from both models (constant, or near-duplicate with |Spearman| ≥ 0.99)

- `first_loan`: |Spearman| 1.00 with `no_score` (kept: preferred for readability)
- `pri_disbursed_amount`: |Spearman| 1.00 with `pri_sanctioned_amount` (kept: higher IV)
- `pri_overdue_accts`: |Spearman| 1.00 with `overdue_ratio` (kept: preferred for readability)
- `sec_sanctioned_amount`: |Spearman| 1.00 with `sec_disbursed_amount` (kept: higher IV)
- `down_payment_share`: |Spearman| 0.99 with `ltv` (kept: preferred for readability)

## Excluded from the scorecard only (LightGBM keeps them)

IV < 0.02: `no_score`, `not_scored`, `overdue_ratio`, `delinq_6m`, `active_ratio`, `leverage`, `inquiries`, `new_accts_6m`, `acct_age_m`, `existing_emi`, `pri_active_accts`, `sec_no_accts`, `sec_active_accts`, `sec_overdue_accts`, `sec_current_balance`, `sec_disbursed_amount`, `asset_cost_log`, `kyc_count`, `has_aadhar`, `has_pan`, `has_voterid`, `has_driving`, `employment_self`, `employment_unknown`, `dob_jan1`, `disb_weekday`, `disb_day`, `manufacturer_te`, `supplier_n`, `pincode_n`.

Correlated above 0.8 with a kept scorecard feature:

- `pri_current_balance`: |Spearman| 0.96 with `pri_sanctioned_amount` (kept: higher IV)
- `pri_no_accts`: |Spearman| 0.90 with `hist_len_m` (kept: preferred for readability)

PSI > 0.25 between Aug–Sep and Oct (investigate): `asset_cost_log`.

## Correlated pairs (|Spearman| > 0.8)

- `no_score` – `first_loan`: 1.00
- `pri_sanctioned_amount` – `pri_disbursed_amount`: 1.00
- `overdue_ratio` – `pri_overdue_accts`: 1.00
- `sec_sanctioned_amount` – `sec_disbursed_amount`: 1.00
- `ltv` – `down_payment_share`: 0.99
- `sec_active_accts` – `sec_sanctioned_amount`: 0.99
- `sec_active_accts` – `sec_disbursed_amount`: 0.99
- `hist_len_m` – `acct_age_m`: 0.98
- `pri_current_balance` – `pri_disbursed_amount`: 0.97
- `pri_current_balance` – `pri_sanctioned_amount`: 0.96
- `leverage` – `pri_current_balance`: 0.96
- `pri_active_accts` – `pri_sanctioned_amount`: 0.96
- `pri_active_accts` – `pri_disbursed_amount`: 0.96
- `active_ratio` – `pri_active_accts`: 0.94
- `pri_active_accts` – `pri_current_balance`: 0.93
- `no_score` – `pri_no_accts`: 0.93
- `first_loan` – `pri_no_accts`: 0.93
- `active_ratio` – `pri_sanctioned_amount`: 0.92
- `active_ratio` – `pri_disbursed_amount`: 0.92
- `leverage` – `pri_disbursed_amount`: 0.91
- `leverage` – `pri_sanctioned_amount`: 0.90
- `hist_len_m` – `pri_no_accts`: 0.90
- `sec_current_balance` – `sec_disbursed_amount`: 0.90
- `no_score` – `hist_len_m`: 0.90
- `hist_len_m` – `first_loan`: 0.90
- `sec_current_balance` – `sec_sanctioned_amount`: 0.89
- `no_score` – `acct_age_m`: 0.89
- `acct_age_m` – `first_loan`: 0.89
- `active_ratio` – `pri_current_balance`: 0.89
- `sec_active_accts` – `sec_current_balance`: 0.89
- `leverage` – `pri_active_accts`: 0.89
- `pri_no_accts` – `pri_active_accts`: 0.89
- `kyc_count` – `has_pan`: 0.88
- `active_ratio` – `leverage`: 0.88
- `has_aadhar` – `has_voterid`: 0.86
- `pri_no_accts` – `pri_sanctioned_amount`: 0.85
- `pri_no_accts` – `pri_disbursed_amount`: 0.85
- `acct_age_m` – `pri_no_accts`: 0.85
- `pri_no_accts` – `pri_current_balance`: 0.83
- `existing_emi` – `total_emi`: 0.82
- `no_score` – `pri_active_accts`: 0.81
- `first_loan` – `pri_active_accts`: 0.81
- `no_score` – `active_ratio`: 0.81
- `active_ratio` – `first_loan`: 0.81
- `sec_no_accts` – `sec_active_accts`: 0.80

## Information value and stability (sorted by IV)

IV guide: < 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, > 0.3 strong. PSI guide: < 0.10 stable, 0.10–0.25 watch, > 0.25 investigate.

|                       | set         |    iv |   psi_oct |   psi_nov | status                 |
|:----------------------|:------------|------:|----------:|----------:|:-----------------------|
| supplier_te           | alternative | 0.105 |     0.011 |     0.026 | scorecard + lgbm       |
| pincode_te            | alternative | 0.078 |     0.008 |     0.016 | scorecard + lgbm       |
| bureau_score          | bureau      | 0.076 |     0.001 |     0.003 | scorecard + lgbm       |
| branch_te             | alternative | 0.072 |     0.025 |     0.079 | scorecard + lgbm       |
| ltv                   | alternative | 0.072 |     0.015 |     0.031 | scorecard + lgbm       |
| down_payment_share    | alternative | 0.068 |     0.005 |     0.016 | dropped                |
| score_desc_grp        | bureau      | 0.062 |     0.001 |     0.004 | scorecard + lgbm       |
| new_emi_est           | alternative | 0.051 |     0.092 |     0.081 | scorecard + lgbm       |
| total_emi             | combined    | 0.042 |     0.045 |     0.043 | scorecard + lgbm       |
| pri_sanctioned_amount | bureau      | 0.03  |     0.001 |     0.004 | scorecard + lgbm       |
| pri_disbursed_amount  | bureau      | 0.029 |     0.001 |     0.004 | dropped                |
| branch_n              | alternative | 0.029 |     0.013 |     0.078 | scorecard + lgbm       |
| pri_current_balance   | bureau      | 0.026 |     0.001 |     0.004 | lgbm only (correlated) |
| hist_len_m            | bureau      | 0.025 |     0.001 |     0.002 | scorecard + lgbm       |
| pri_no_accts          | bureau      | 0.023 |     0.001 |     0.003 | lgbm only (correlated) |
| overdue_ratio         | bureau      | 0.02  |     0     |     0.001 | lgbm only (IV < 0.02)  |
| pri_active_accts      | bureau      | 0.02  |     0.001 |     0.005 | lgbm only (IV < 0.02)  |
| acct_age_m            | bureau      | 0.019 |     0.001 |     0.002 | lgbm only (IV < 0.02)  |
| dob_jan1              | alternative | 0.016 |     0     |     0     | lgbm only (IV < 0.02)  |
| no_score              | bureau      | 0.016 |     0     |     0.001 | lgbm only (IV < 0.02)  |
| first_loan            | bureau      | 0.016 |     0     |     0.001 | dropped                |
| has_voterid           | alternative | 0.015 |     0     |     0.005 | lgbm only (IV < 0.02)  |
| leverage              | bureau      | 0.014 |     0.001 |     0.004 | lgbm only (IV < 0.02)  |
| has_aadhar            | alternative | 0.014 |     0     |     0.007 | lgbm only (IV < 0.02)  |
| active_ratio          | bureau      | 0.014 |     0.001 |     0.004 | lgbm only (IV < 0.02)  |
| disb_day              | alternative | 0.012 |     0.195 |     0.544 | lgbm only (IV < 0.02)  |
| pri_overdue_accts     | bureau      | 0.011 |     0     |     0     | dropped                |
| new_accts_6m          | bureau      | 0.011 |     0     |     0.007 | lgbm only (IV < 0.02)  |
| manufacturer_te       | alternative | 0.008 |     0.003 |     0.01  | lgbm only (IV < 0.02)  |
| inquiries             | bureau      | 0.008 |     0.001 |     0.002 | lgbm only (IV < 0.02)  |
| supplier_n            | alternative | 0.007 |     0.035 |     0.059 | lgbm only (IV < 0.02)  |
| existing_emi          | bureau      | 0.007 |     0.001 |     0.003 | lgbm only (IV < 0.02)  |
| delinq_6m             | bureau      | 0.007 |     0     |     0.001 | lgbm only (IV < 0.02)  |
| employment_self       | alternative | 0.005 |     0     |     0.001 | lgbm only (IV < 0.02)  |
| pincode_n             | alternative | 0.003 |     0.025 |     0.035 | lgbm only (IV < 0.02)  |
| asset_cost_log        | alternative | 0.002 |     0.278 |     0.193 | lgbm only (IV < 0.02)  |
| disb_weekday          | alternative | 0.002 |     0.105 |     0.013 | lgbm only (IV < 0.02)  |
| sec_no_accts          | bureau      | 0.001 |     0.027 |     0.028 | lgbm only (IV < 0.02)  |
| sec_active_accts      | bureau      | 0.001 |     0.017 |     0.017 | lgbm only (IV < 0.02)  |
| sec_disbursed_amount  | bureau      | 0.001 |     0.017 |     0.017 | lgbm only (IV < 0.02)  |
| sec_sanctioned_amount | bureau      | 0.001 |     0.017 |     0.017 | dropped                |
| kyc_count             | alternative | 0.001 |     0.001 |     0.103 | lgbm only (IV < 0.02)  |
| sec_overdue_accts     | bureau      | 0     |     0.006 |     0.006 | lgbm only (IV < 0.02)  |
| has_driving           | alternative | 0     |     0.002 |     0.004 | lgbm only (IV < 0.02)  |
| has_pan               | alternative | 0     |     0.001 |     0.116 | lgbm only (IV < 0.02)  |
| not_scored            | bureau      | 0     |     0     |     0.002 | lgbm only (IV < 0.02)  |
| employment_unknown    | alternative | 0     |     0.001 |     0     | lgbm only (IV < 0.02)  |
| sec_current_balance   | bureau      | 0     |     0     |     0     | lgbm only (IV < 0.02)  |

Final counts: 43 features for LightGBM (bureau 22, alternative 20), 11 for the scorecard.
