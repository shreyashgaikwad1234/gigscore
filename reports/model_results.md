# Model results — step 3

Produced by `src/train.py` (`python -m src.train --trials 40 --boot 500`) on 2026-10-29. Real L&T data. Tuning (Optuna on 5-fold CV), tree counts, scorecard C and isotonic calibration (on out-of-fold predictions) are all decided inside Aug–Sep 2018; final models are fit on all of Aug–Sep and tested once, **out-of-time on October 2018** (98,364 loans). CIs are 95% percentile intervals from 500 paired bootstrap resamples of the October rows.

Leakage check (CLAUDE.md rule 2): no AUC above 0.75.

## all borrowers — 98,364 loans, default rate 23.5%

| model | AUC (95% CI) | Gini (95% CI) | KS (95% CI) |
|:--|--:|--:|--:|
| scorecard_bureau | 0.555 (0.551–0.559) | 0.110 (0.102–0.118) | 0.081 (0.076–0.088) |
| lgbm_bureau | 0.588 (0.584–0.592) | 0.175 (0.168–0.184) | 0.113 (0.108–0.120) |
| scorecard_alternative | 0.617 (0.612–0.621) | 0.233 (0.224–0.242) | 0.170 (0.163–0.178) |
| lgbm_alternative | 0.631 (0.627–0.635) | 0.262 (0.254–0.271) | 0.187 (0.181–0.195) |
| scorecard_combined | 0.629 (0.625–0.633) | 0.258 (0.250–0.267) | 0.189 (0.183–0.197) |
| lgbm_combined | 0.658 (0.654–0.662) | 0.316 (0.308–0.325) | 0.229 (0.223–0.237) |
| scorecard_fair | 0.597 (0.593–0.601) | 0.194 (0.186–0.202) | 0.144 (0.137–0.152) |
| lgbm_fair | 0.637 (0.633–0.641) | 0.273 (0.266–0.282) | 0.197 (0.191–0.205) |

## no bureau score (score == 0) — 49,748 loans, default rate 24.1%

| model | AUC (95% CI) | Gini (95% CI) | KS (95% CI) |
|:--|--:|--:|--:|
| scorecard_bureau | 0.499 (0.498–0.500) | -0.002 (-0.004–-0.000) | 0.000 (0.000–0.001) |
| lgbm_bureau | 0.507 (0.505–0.510) | 0.014 (0.010–0.020) | 0.015 (0.011–0.020) |
| scorecard_alternative | 0.625 (0.620–0.630) | 0.250 (0.240–0.261) | 0.183 (0.175–0.194) |
| lgbm_alternative | 0.640 (0.635–0.645) | 0.280 (0.270–0.291) | 0.199 (0.191–0.210) |
| scorecard_combined | 0.625 (0.620–0.630) | 0.250 (0.239–0.260) | 0.184 (0.175–0.195) |
| lgbm_combined | 0.643 (0.638–0.649) | 0.287 (0.276–0.298) | 0.203 (0.195–0.213) |
| scorecard_fair | 0.580 (0.574–0.585) | 0.160 (0.148–0.170) | 0.129 (0.119–0.138) |
| lgbm_fair | 0.611 (0.606–0.617) | 0.223 (0.213–0.234) | 0.165 (0.156–0.174) |

## no usable score (0 or 'Not Scored') — 55,237 loans, default rate 24.0%

| model | AUC (95% CI) | Gini (95% CI) | KS (95% CI) |
|:--|--:|--:|--:|
| scorecard_bureau | 0.503 (0.500–0.506) | 0.005 (-0.001–0.011) | 0.006 (0.003–0.012) |
| lgbm_bureau | 0.519 (0.515–0.522) | 0.038 (0.031–0.045) | 0.027 (0.021–0.033) |
| scorecard_alternative | 0.624 (0.618–0.629) | 0.247 (0.236–0.258) | 0.180 (0.173–0.191) |
| lgbm_alternative | 0.638 (0.633–0.642) | 0.275 (0.265–0.285) | 0.196 (0.188–0.205) |
| scorecard_combined | 0.624 (0.618–0.629) | 0.247 (0.236–0.258) | 0.182 (0.174–0.192) |
| lgbm_combined | 0.643 (0.638–0.648) | 0.287 (0.277–0.297) | 0.203 (0.196–0.213) |
| scorecard_fair | 0.580 (0.575–0.586) | 0.161 (0.149–0.171) | 0.130 (0.121–0.138) |
| lgbm_fair | 0.612 (0.607–0.617) | 0.224 (0.214–0.235) | 0.166 (0.157–0.175) |

## self-employed — 54,373 loans, default rate 24.6%

| model | AUC (95% CI) | Gini (95% CI) | KS (95% CI) |
|:--|--:|--:|--:|
| scorecard_bureau | 0.550 (0.545–0.555) | 0.100 (0.090–0.110) | 0.073 (0.065–0.081) |
| lgbm_bureau | 0.583 (0.578–0.588) | 0.167 (0.156–0.177) | 0.105 (0.098–0.114) |
| scorecard_alternative | 0.616 (0.611–0.622) | 0.232 (0.221–0.243) | 0.170 (0.162–0.180) |
| lgbm_alternative | 0.629 (0.624–0.635) | 0.258 (0.247–0.269) | 0.185 (0.176–0.195) |
| scorecard_combined | 0.627 (0.621–0.632) | 0.253 (0.243–0.264) | 0.186 (0.177–0.197) |
| lgbm_combined | 0.655 (0.649–0.660) | 0.309 (0.298–0.320) | 0.222 (0.214–0.233) |
| scorecard_fair | 0.590 (0.585–0.596) | 0.181 (0.170–0.191) | 0.134 (0.126–0.144) |
| lgbm_fair | 0.631 (0.626–0.636) | 0.261 (0.252–0.272) | 0.190 (0.182–0.200) |

## Gain from alternative data (AUC difference vs the bureau-only model, paired 95% CI)

| model type | feature set | subgroup | AUC gain (95% CI) |
|:--|:--|:--|--:|
| lgbm | alternative vs bureau | no_score | +0.133 (+0.127 to +0.139) |
| lgbm | combined vs bureau | no_score | +0.136 (+0.131 to +0.142) |
| lgbm | fair vs bureau | no_score | +0.104 (+0.098 to +0.110) |
| scorecard | alternative vs bureau | no_score | +0.126 (+0.121 to +0.132) |
| scorecard | combined vs bureau | no_score | +0.126 (+0.121 to +0.132) |
| scorecard | fair vs bureau | no_score | +0.081 (+0.075 to +0.086) |
| lgbm | alternative vs bureau | no_usable_score | +0.119 (+0.113 to +0.125) |
| lgbm | combined vs bureau | no_usable_score | +0.125 (+0.119 to +0.130) |
| lgbm | fair vs bureau | no_usable_score | +0.093 (+0.088 to +0.099) |
| scorecard | alternative vs bureau | no_usable_score | +0.121 (+0.115 to +0.127) |
| scorecard | combined vs bureau | no_usable_score | +0.121 (+0.116 to +0.127) |
| scorecard | fair vs bureau | no_usable_score | +0.078 (+0.072 to +0.083) |
| lgbm | alternative vs bureau | all | +0.044 (+0.038 to +0.049) |
| lgbm | combined vs bureau | all | +0.070 (+0.066 to +0.074) |
| lgbm | fair vs bureau | all | +0.049 (+0.045 to +0.052) |
| scorecard | alternative vs bureau | all | +0.062 (+0.056 to +0.067) |
| scorecard | combined vs bureau | all | +0.074 (+0.070 to +0.078) |
| scorecard | fair vs bureau | all | +0.042 (+0.039 to +0.045) |
| lgbm | alternative vs bureau | self_employed | +0.046 (+0.039 to +0.053) |
| lgbm | combined vs bureau | self_employed | +0.071 (+0.066 to +0.077) |
| lgbm | fair vs bureau | self_employed | +0.047 (+0.043 to +0.052) |
| scorecard | alternative vs bureau | self_employed | +0.066 (+0.059 to +0.073) |
| scorecard | combined vs bureau | self_employed | +0.077 (+0.071 to +0.082) |
| scorecard | fair vs bureau | self_employed | +0.040 (+0.035 to +0.045) |

## Calibration-in-the-large on October

Mean predicted PD vs observed default rate. LightGBM is isotonic-calibrated on Aug–Sep out-of-fold predictions; October's default rate is higher (23.5% vs 20.4%), so some under-prediction is expected. Full calibration analysis is step 4.

| model | mean predicted PD | observed |
|:--|--:|--:|
| scorecard_bureau | 20.4% | 23.5% |
| lgbm_bureau | 20.5% | 23.5% |
| scorecard_alternative | 20.1% | 23.5% |
| lgbm_alternative | 20.1% | 23.5% |
| scorecard_combined | 20.1% | 23.5% |
| lgbm_combined | 20.2% | 23.5% |
| scorecard_fair | 20.6% | 23.5% |
| lgbm_fair | 20.7% | 23.5% |

## Training details

- **scorecard_bureau**: 4 features; Aug–Sep 5-fold CV AUC 0.5752; C = 1.0; removed for a wrong-sign coefficient: none.
- **lgbm_bureau**: 22 features; Aug–Sep 5-fold CV AUC 0.6032; params {'num_leaves': 39, 'min_child_samples': 220, 'feature_fraction': 0.6854, 'bagging_fraction': 0.9953, 'lambda_l1': 1.8475, 'lambda_l2': 11.6166}; 270 trees (fold early-stops [290, 163, 329, 192, 254]).
- **scorecard_alternative**: 6 features; Aug–Sep 5-fold CV AUC 0.6282; C = 0.01; removed for a wrong-sign coefficient: none.
- **lgbm_alternative**: 20 features; Aug–Sep 5-fold CV AUC 0.6467; params {'num_leaves': 36, 'min_child_samples': 292, 'feature_fraction': 0.6323, 'bagging_fraction': 0.9991, 'lambda_l1': 0.1414, 'lambda_l2': 27.2863}; 506 trees (fold early-stops [380, 411, 299, 747, 461]).
- **scorecard_combined**: 10 features; Aug–Sep 5-fold CV AUC 0.6476; C = 1.0; removed for a wrong-sign coefficient: ['hist_len_m'].
- **lgbm_combined**: 43 features; Aug–Sep 5-fold CV AUC 0.6777; params {'num_leaves': 49, 'min_child_samples': 162, 'feature_fraction': 0.4079, 'bagging_fraction': 0.9753, 'lambda_l1': 4.8625, 'lambda_l2': 22.034}; 559 trees (fold early-stops [410, 545, 547, 693, 348]).
- **scorecard_fair**: 7 features; Aug–Sep 5-fold CV AUC 0.6158; C = 0.1; removed for a wrong-sign coefficient: none.
- **lgbm_fair**: 32 features; Aug–Sep 5-fold CV AUC 0.6537; params {'num_leaves': 42, 'min_child_samples': 93, 'feature_fraction': 0.439, 'bagging_fraction': 0.9744, 'lambda_l1': 7.2867, 'lambda_l2': 26.619}; 428 trees (fold early-stops [463, 436, 338, 424, 284]).

- lgbm_bureau top features by gain: `bureau_score`, `pri_sanctioned_amount`, `hist_len_m`, `existing_emi`, `pri_current_balance`, `score_desc_grp`, `leverage`, `acct_age_m`, `pri_no_accts`, `inquiries`
- lgbm_alternative top features by gain: `supplier_te`, `ltv`, `pincode_te`, `new_emi_est`, `branch_te`, `asset_cost_log`, `supplier_n`, `pincode_n`, `disb_day`, `branch_n`
- lgbm_combined top features by gain: `supplier_te`, `ltv`, `bureau_score`, `pincode_te`, `branch_te`, `total_emi`, `pri_sanctioned_amount`, `new_emi_est`, `score_desc_grp`, `asset_cost_log`
- lgbm_fair top features by gain: `bureau_score`, `ltv`, `new_emi_est`, `total_emi`, `score_desc_grp`, `pri_sanctioned_amount`, `manufacturer_te`, `hist_len_m`, `asset_cost_log`, `pri_current_balance`
