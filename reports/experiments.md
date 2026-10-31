# Improvement experiments (Aug–Sep cross-validation only)

Produced by `src/experiments.py` on 2026-10-29. 5-fold CV inside Aug–Sep 2018 using the same folds as the out-of-fold target encoding; LightGBM with the step 3 combined-model parameters at learning rate 0.05. **October was not used.** A change is adopted only if it gains at least 0.002 CV AUC.

| experiment | features | CV AUC | gain | note |
|:--|--:|--:|--:|:--|
| baseline (current combined set) | 43 | 0.6765 | +0.0000 |  |
| + supplier_mfr_te | 44 | 0.6767 | +0.0001 |  |
| + branch_mfr_te | 44 | 0.6767 | +0.0002 |  |
| + branch_supplier_te | 44 | 0.6769 | +0.0004 |  |
| + ltv_vs_dealer | 44 | 0.6772 | +0.0006 |  |
| + dealer_noscore_share | 44 | 0.6757 | -0.0008 |  |
| + asset_vs_dealer | 44 | 0.6767 | +0.0002 |  |
| all candidates together | 49 | 0.6772 | +0.0007 |  |
| baseline, 3-seed bagging | 43 | 0.6769 | +0.0004 |  |
| blend LightGBM 1.00 + scorecard 0.00 | 43 | 0.6765 | +0.0000 | weight chosen on OOF |
