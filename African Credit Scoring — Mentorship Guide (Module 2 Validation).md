# African Credit Scoring — Mentorship Guide (Module 2: Validation)

Nov 3, 2026 · @Bilikisu Wunmi Olatunji

## Why validation comes first

Your validation score is the only evidence you will have that the model works, so the validation set must look like the data the model will face. Get this wrong and every later choice (features, model, threshold) is tuned to a fiction.

- **In a competition**, a bad scheme makes your local score disagree with the leaderboard; you chase noise.
- **In a bank**, a bad scheme gets a model approved that then loses money on real borrowers. Model risk teams (and regulators) ask first: *how was this validated, and on what population?*
- **The question to ask every time:** "At prediction time, what will the model know, and what will it never have seen?" Your validation split must hide exactly that.

This module answers that question with measurements, not assumptions, then locks in a protocol we keep for the rest of the project.

## What the real test looks like

The test set is two different problems glued together: Kenya test (81% of rows) is a random draw from the same loan book as train, while Ghana test (19%) is a new market the model has never seen.

| Question | Kenya test (15,069 rows) | Ghana test (3,525 rows) |
| --- | --- | --- |
| Customer seen in train? | 98.4% of rows | 0% |
| Same loan already in train (another lender's share)? | 920 rows (6.1%) | 0 |
| Disbursement-quarter mix vs train | Near-identical (e.g., 2022 Q3: 58.7% vs 59.1%) | Different: Q1 2022–Q1 2023 only |
| Test loan falls inside that customer's train date range | 86% (7% after, 7% before) | — |
| Lenders and products | Same as train | Mostly unseen |
| Labels to check against | None, but same population as train | None, and a different population |

What this tells us:

- **Kenya test was split randomly by loan, not by time or customer.** Its customers and dates are interleaved with train. So for Kenya, the honest validation is "hold out loans, keep customers" (scheme B below).
- **Ghana is a true out-of-distribution problem.** No split of Kenya data reproduces it exactly; we can only stress-test with proxies (scheme E).
- **A production lender faces a third problem the competition hides: the future.** Real loans are scored *before* they happen, so we also test forward in time (scheme D), even though it does not match the leaderboard.

## Adversarial validation

Adversarial validation trains a classifier to tell train rows from test rows; if it cannot (AUC near 0.5), the two sets come from the same distribution. Kenya test scored 0.50; Ghana test scored 1.00, a perfect separation.

How it works:

1. Stack train and test features, label train = 0 and test = 1 (ignore `target`).
2. Run 5-fold cross-validation with any model (we used LightGBM on loan amount, markup, duration, new/repeat, lender share).
3. Read the AUC: 0.5 = indistinguishable, 1.0 = completely different. Then check which feature gives the game away.

| Comparison | AUC, all 5 features | Single most revealing feature |
| --- | --- | --- |
| Train vs Kenya test | 0.50 | none |
| Train vs Ghana test | 1.00 | markup alone: 0.995; lender share alone: 0.983 |

The surprise is not currency (raw amount alone gives 0.77) but **pricing and funding structure**. Ghana loans cluster at a 3% markup and lender shares of 13–20%; half of Kenya loans carry under 1% markup, and most have a 30% share. Markup is our strongest Kenya predictor, so its meaning may not transfer.

The consequence, measured: a simple Kenya-trained model flags 1.8% of Kenya test loans as defaulters but **11.3% of Ghana loans**. That may be real (Ghana inflation ran at 31–38% in 2022–2023), or it may be the model misreading Ghana's normal pricing as risk. We cannot tell without labels, and that uncertainty must go in the model card.

## The validation schemes

We run five schemes; each simulates one situation the model will meet, and each answers a different question.

| Scheme | How rows are split | Simulates | Question it answers |
| --- | --- | --- | --- |
| A. Random rows | `StratifiedKFold` on rows | Nothing real | How much leakage flatters you (a warning, not a score) |
| B. Group by loan | `StratifiedGroupKFold`, groups = `tbl_loan_id` | Kenya test: known customers, unseen loans | **Primary score for the leaderboard** |
| C. Group by customer | `StratifiedGroupKFold`, groups = `customer_id` | Brand-new applicants | Can we score people with no history? |
| D. Forward in time | Train before 1 Jan 2023, validate 2023 onward | Production: scoring next month's loans | Does the model survive vintage drift? |
| E. Leave one lender out | Train on 3 lenders, validate on the 4th; no lender or product codes | Ghana: unseen lenders and products | Does ranking and threshold transfer to a new channel? |

Why group by loan for B: 2,134 loans have several lender rows with the same outcome. A random split puts one share in train and its twin in validation, and the model simply recognises the loan.

Why `Stratified`: with 1.8% defaults, each fold must keep roughly the same default rate (about 250 defaulters per fold), or fold scores swing for no reason.

## Baseline results

Our baseline scores F1 0.823 on the primary scheme (B), but falls to 0.565 when tested forward in time, and its best threshold swings from 0.05 to 0.71 across lenders. Ranking travels better than thresholds do.

Baseline model, kept deliberately simple: LightGBM (300 trees, learning rate 0.05) on 5 numeric features (log amount, markup, duration, new/repeat, lender share) plus `lender_id` and `loan_type` as categories. F1 uses a threshold chosen on *other* folds (see next section).

| Scheme | Validation rows | Defaults | ROC AUC | PR AUC | F1 (mean ± sd across folds) |
| --- | --- | --- | --- | --- | --- |
| A. Random rows | 68,654 | 1,258 | 0.994 | 0.924 | 0.847 ± 0.004 |
| B. Group by loan (primary) | 68,654 | 1,258 | 0.993 | 0.903 | 0.823 ± 0.020 |
| C. Group by customer | 68,654 | 1,258 | 0.993 | 0.896 | 0.810 ± 0.017 |
| D. Forward in time (2023+) | 4,210 | 401 | 0.897 | 0.609 | 0.565 (single split) |

Scheme E, leave one lender out (no lender or product codes):

| Held-out lender | Rows | Actual default rate | Mean predicted PD | ROC AUC | F1-optimal threshold |
| --- | --- | --- | --- | --- | --- |
| 267278 | 64,653 | 1.2% | 2.1% | 0.987 | 0.71 |
| 251804 | 3,542 | 8.9% | 8.0% | 0.980 | 0.37 |
| 267277 | 271 | 29.5% | 20.4% | 0.903 | 0.05 |
| 245684 | 188 | 41.5% | 36.1% | 0.793 | 0.06 |

How to read this:

- **A vs B (0.847 vs 0.823):** the 2.4-point gap is pure leakage from split loans. Never report scheme A.
- **B vs C (0.823 vs 0.810):** knowing the customer helps a little even without history features. Module 3's customer-history features should widen this gap; scheme C keeps us honest about new applicants.
- **D (0.565):** 2023+ loans defaulted at 9.5%, but the model trained on the mostly safe pre-2023 book (1.3%) predicted only 5.8% on average. This is the number a credit committee should see before production.
- **E:** the model still *ranks* borrowers well on an unseen lender (AUC 0.79–0.99), but the cut-off that works on one lender is useless on another. For Ghana, threshold choice is the main risk.

Rule of thumb from the fold spread: with about 250 defaulters per fold, F1 differences under about 0.02 are noise. Do not celebrate a 0.005 gain.

## Choosing the F1 threshold honestly

If you pick the threshold that maximises F1 on the same predictions you then score, you are grading your own exam. Pick it on data the score is not computed on.

The honest procedure we use ("nested" threshold):

1. Get out-of-fold (OOF) predictions for every training row under scheme B.
2. For fold *k*, find the F1-best threshold using the OOF predictions of the *other* four folds only.
3. Apply that threshold to fold *k* and record F1.
4. Report the mean and spread of the five fold scores. For the final submission, use the threshold that maximises F1 on all OOF predictions.

What we measured:

| Scheme | F1 at a threshold tuned on the same rows | Honest F1 | Thresholds picked per fold |
| --- | --- | --- | --- |
| B. Group by loan | 0.828 | 0.823 | 0.38–0.51 |
| D. Forward in time | 0.607 (tuned on 2023+, cheating) | 0.565 (tuned on pre-2023) | 0.33 (pre-2023) vs 0.14 (2023+) |

Three lessons:

- **Within Kenya the threshold is stable** (0.38–0.51), so the optimism is small. That will not hold once we add class weights or resampling, which push probabilities up and the best threshold with them.
- **Across time and lenders it is not stable.** The best cut-off moved from 0.33 to 0.14 in scheme D and from 0.05 to 0.71 in scheme E. A threshold is a business setting to monitor and recalibrate, not a fixed model parameter.
- **Threshold on the probability, not on the rank.** In Module 5 we will instead set cut-offs by target default rate per risk band, which is how lenders actually operate.

## The validation protocol (locked for Modules 3–6)

From now on every experiment is scored the same way, so results are comparable and nobody can cherry-pick.

1. **Folds are fixed.** The notebook writes `outputs/folds.csv` (one row per `ID`, columns `fold_loan` and `fold_customer`, seed 42). Every experiment reads this file; nobody re-splits.
2. **Primary metric:** honest F1 on scheme B, reported as mean ± sd. Also report PR AUC, which does not depend on the threshold.
3. **Guardrails:** every experiment also reports scheme C (new applicants), scheme D (forward in time) and the scheme E table. A change that raises B but clearly lowers D or E is rejected.
4. **Significance:** a change counts only if B improves by more than one fold standard deviation (about 0.02 today) *or* improves in at least 4 of 5 folds.
5. **Everything inside the fold.** Any step that learns from data (scaling, encodings, target encoding, imputation, feature selection, threshold) is fitted on the training folds only. Use a scikit-learn `Pipeline` so this is automatic.
6. **Log every run.** Append one row to `outputs/experiments.csv`: date, description, features, scheme B/C/D scores, threshold. A score not in the log does not exist.
7. **The test set is touched once per finished model**, never to choose features or thresholds.

## Your Module 2 tasks

Work through `02_validation.ipynb` in your project folder. It reproduces every number in this doc and creates `outputs/folds.csv` and `outputs/experiments.csv`. Full run takes a few minutes on a laptop.

Hands-on:

- [ ] Run the notebook top to bottom; confirm your scheme B score matches 0.823 ± 0.020
- [ ] TODO 3: rerun the Ghana adversarial check without markup and lender share, then with USD amounts. What still separates Ghana?
- [ ] TODO 8: score each held-out lender with the scheme B threshold. How much F1 is lost?
- [ ] TODO 10a: fix the starter logistic regression, wrap it in a `Pipeline`, score it under scheme B, log it
- [ ] TODO 10b: plot F1-vs-threshold per fold; describe where the curve is flat
- [ ] TODO 10c: make one baseline submission; if you can submit, compare the leaderboard score to scheme B

Questions to answer in writing:

1. Why is group-by-loan the right primary scheme for this competition, but not enough for a real lender?
2. Our model predicted a 5.8% default rate for 2023+ loans that actually defaulted at 9.5%. What would that mean for the asset manager's provisions and pricing?
3. Adversarial validation showed markup alone separates Ghana from Kenya. Should we drop markup, transform it, or keep it? Argue one option.
4. Scheme E shows the best threshold ranging from 0.05 to 0.71 by lender. Propose a way to set Ghana's threshold without Ghana labels.
5. Why is a 0.005 F1 improvement on scheme B not evidence of a better model?

Next: Module 3, feature engineering (currency-neutral amounts, customer history built without time travel, smoothed encodings, macro features), scored with this exact protocol.
