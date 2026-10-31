# Data profile — L&T train.csv

Produced by `src/profiling.py` (called from `notebooks/01_profile.ipynb`) on 2026-10-29 from `data/raw/train.csv`. Real L&T data; nothing here is simulated.

## Headline

- Rows: **233,154**, columns: **41**
- Default rate (first EMI missed): **21.7%**
- No bureau score (`PERFORM_CNS.SCORE == 0`): **50.2%** of rows, default rate 23.1% vs 20.3% for scored rows
- Self employed: **54.7%** of rows, default rate 22.8%
- Disbursal dates: **01 Aug 2018 – 31 Oct 2018**

## Rows by disbursal month

| month   |   rows | share   | default_rate   |
|:--------|-------:|:--------|:---------------|
| 2018-08 |  68002 | 29.2%   | 21.6%          |
| 2018-09 |  66788 | 28.6%   | 19.2%          |
| 2018-10 |  98364 | 42.2%   | 23.5%          |

## No-score vs scored

| group      |   rows | share   | default_rate   |
|:-----------|-------:|:--------|:---------------|
| score == 0 | 116950 | 50.2%   | 23.1%          |
| score > 0  | 116204 | 49.8%   | 20.3%          |

## Score groups (treating 'Not Scored' placeholder scores as no score)

| group                       |   rows | share   | default_rate   |
|:----------------------------|-------:|:--------|:---------------|
| No Bureau History (score 0) | 116950 | 50.2%   | 23.1%          |
| Real bureau score           | 103369 | 44.3%   | 20.1%          |
| Not Scored (score 11-18)    |  12835 | 5.5%    | 21.5%          |

AUC of the raw bureau score on rows with a real score: 0.590 (in-sample, all months, no model; a ceiling check for the bureau-only feature set).

First-time borrowers (`PRI.NO.OF.ACCTS == 0`): 50.2%.

## Bureau score description

| PERFORM_CNS.SCORE.DESCRIPTION                           |   rows | default_rate   |   score_min |   score_max |
|:--------------------------------------------------------|-------:|:---------------|------------:|------------:|
| No Bureau History Available                             | 116950 | 23.1%          |           0 |           0 |
| C-Very Low Risk                                         |  16045 | 17.3%          |         736 |         760 |
| A-Very Low Risk                                         |  14124 | 16.6%          |         806 |         890 |
| D-Very Low Risk                                         |  11358 | 15.0%          |         706 |         735 |
| B-Very Low Risk                                         |   9201 | 13.1%          |         761 |         805 |
| M-Very High Risk                                        |   8776 | 30.5%          |         300 |         300 |
| F-Low Risk                                              |   8485 | 18.6%          |         651 |         680 |
| K-High Risk                                             |   8277 | 27.8%          |         351 |         520 |
| H-Medium Risk                                           |   6855 | 24.2%          |         601 |         630 |
| E-Low Risk                                              |   5821 | 17.2%          |         681 |         705 |
| I-Medium Risk                                           |   5557 | 27.3%          |         571 |         600 |
| G-Low Risk                                              |   3988 | 19.7%          |         631 |         650 |
| Not Scored: Sufficient History Not Available            |   3765 | 25.6%          |          15 |          15 |
| J-High Risk                                             |   3748 | 25.2%          |         521 |         570 |
| Not Scored: Not Enough Info available on the customer   |   3672 | 21.0%          |          17 |          17 |
| Not Scored: No Activity seen on the customer (Inactive) |   2885 | 18.4%          |          16 |          16 |
| Not Scored: No Updates available in last 36 months      |   1534 | 19.0%          |          18 |          18 |
| L-Very High Risk                                        |   1134 | 28.0%          |         301 |         350 |
| Not Scored: Only a Guarantor                            |    976 | 21.3%          |          14 |          14 |
| Not Scored: More than 50 active Accounts found          |      3 | 0.0%           |          11 |          11 |

## Employment type

7,661 blank values in the raw file, filled as `Unknown`.

| Employment.Type   |   rows | share   | default_rate   |
|:------------------|-------:|:--------|:---------------|
| Self employed     | 127635 | 54.7%   | 22.8%          |
| Salaried          |  97858 | 42.0%   | 20.3%          |
| Unknown           |   7661 | 3.3%    | 21.5%          |

## Missing values

Columns with > 5% missing: **none**.

All columns with any missing: `Employment.Type` (3.3%).

## Data-quality checks

- Birth years moved back a century by the pivot rule (> 2025 → 19xx): 19,866
- Age at disbursal after fixes: 18.0 – 69.1 years (0 under 18, 0 over 75)
- Unparsed dates: DOB 0, disbursal 0
- Unparsed account-age strings: AVERAGE.ACCT.AGE 0, CREDIT.HISTORY.LENGTH 0
- Duplicate UniqueID: 0
- Date of birth recorded as 1 January (likely a placeholder): 23.7%
- test.csv: 112,392 rows, disbursed 03 Nov 2018 – 30 Nov 2018
- PRI/SEC columns with negative values (before winsorising): `PRI.CURRENT.BALANCE` (448), `SEC.CURRENT.BALANCE` (61)
- Document-flag share = 1: `MobileNo_Avl_Flag` 100.0%, `Aadhar_flag` 84.0%, `PAN_flag` 7.6%, `VoterID_flag` 14.5%, `Driving_flag` 2.3%, `Passport_flag` 0.2%
- Distinct levels: `branch_id` 82, `supplier_id` 2,953, `manufacturer_id` 11, `Current_pincode_ID` 6,698, `State_ID` 22, `Employee_code_ID` 3,270

## Anything that contradicts CLAUDE.md or the guide

- 12,835 rows have a 'Not Scored'/'No Bureau' description but a non-zero PERFORM_CNS.SCORE (values 11–18). CLAUDE.md treats only score == 0 as no score; these rows would count as 'scored' under that rule. Decision needed before features.
- October default rate is 23.5% vs 20.4% in Aug–Sep: a base-rate shift in the out-of-time month. Probabilities calibrated on Aug–Sep will under-predict October; check calibration-in-the-large in step 4.
- test.csv is disbursed 03 Nov – 30 Nov 2018, after train.csv, not Aug–Oct as the guide says. It has no label, so this changes nothing for evaluation, but it is a later-period sample usable for score-drift (PSI) checks.
