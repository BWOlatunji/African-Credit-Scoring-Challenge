# African Credit Scoring — Mentorship Guide (Module 1)

Oct 29, 2026 · @Bilikisu Wunmi Olatunji

## How we'll work

We build this in six modules; you finish each module's tasks before we move on. This doc is Module 1, and its companion notebook `01_data_understanding.ipynb` is in your project folder.

| Module | Goal | You hand in |
| --- | --- | --- |
| 1. Understand the problem and data | Know every column, the grain, the shifts, the external data | Answers to the Module 1 questions + completed notebook |
| 2. Validation design | A CV scheme that mimics the real test (Kenya repeat customers + unseen Ghana) | `02_validation.ipynb` with a baseline score |
| 3. Feature engineering | Currency-neutral, customer-history and macro features | Feature pipeline in `src/features.py` |
| 4. Modelling and threshold | Gradient boosting, F1-optimal threshold, calibration | Model card + CV report |
| 5. Credit scoring function | Turn probabilities into risk bands and a score scale | Scorecard table + band default rates |
| 6. Production | Packaged pipeline, FastAPI endpoint, tests, monitoring plan | Running API in Docker |

Rule for the whole project: every number you report must come from code you can rerun. No eyeballed figures.

## The business problem

The client is a private asset manager that co-funds loans across several African markets; it needs to know, before money goes out, how likely a loan is to default. Your model is the engine of their future credit score.

- **Who funds what.** A loan is originated by a lender (e.g., a digital lending partner). The asset manager funds a *portion* of it, which is why you see `Amount_Funded_By_Lender` and `Lender_portion_Funded`. Their money is at risk only on that portion.
- **What they want.** A probability of default (PD) for each loan, then a decision (lend / don't lend) and later a risk band. The brief asks for a model that works for existing customers *and* new applicants, and for a new market (Ghana).
- **Why F1.** Only 1.8% of training loans defaulted. Accuracy is useless here: predicting "no one defaults" scores 98.2% accuracy and catches zero defaulters. F1 balances precision (of the loans we flag, how many really default) and recall (of the real defaulters, how many we catch).
- **What F1 forces on you.** The submission is a hard 0/1, so you must choose a probability *threshold*. The default 0.5 is almost never right for rare events. Choosing it well is a Module 4 skill.
- **The second deliverable.** The top teams also had to build a credit scoring function: bin the probabilities into risk categories and propose a scalable score. Design with this in mind from day one; it is what the business actually uses.

```latex
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

## Credit risk 101

Lenders lose money in a predictable way, and the industry describes it with three numbers: PD, LGD and EAD. In this challenge you model only PD, but you should know where it sits.

```latex
\text{Expected Loss} = PD \times LGD \times EAD
```

| Term | Meaning | In this dataset |
| --- | --- | --- |
| Default | Borrower fails to repay as agreed (usually measured some days past due) | `target = 1`; the exact days-past-due rule is not given — note it as an open question |
| PD (probability of default) | Chance a loan defaults | What your model outputs |
| LGD (loss given default) | Share of exposure lost after recoveries | Not available |
| EAD (exposure at default) | Money at risk when default happens | Approximated by `Lender_portion_to_be_repaid` |
| Principal | Amount lent | `Total_Amount` |
| Interest + fees (markup) | Price of the loan | `Total_Amount_to_Repay − Total_Amount` |
| Tenor | Loan length | `duration` (days) |
| New vs repeat borrower | First loan vs returning customer | `New_versus_Repeat` |
| Risk-based pricing | Riskier borrowers pay more | Visible in the markup: high-markup loans default far more |
| Vintage | Loans grouped by disbursement month/quarter | Default rate swings sharply by quarter (see findings) |
| Credit scorecard | Maps PD to a points scale (e.g., 300–850) and risk bands | Module 5 deliverable |

Two mindset shifts for finance work: a false negative (missed defaulter) costs principal, a false positive (rejected good customer) costs only lost interest, so the costs are not symmetric. And regulators and credit committees will ask *why* a customer was declined, so explainability is not optional.

## The files and every variable

You have 68,654 labelled training rows (Kenya only) and 18,594 test rows (15,069 Kenya, 3,525 Ghana); there are no missing values in either.

| File | Rows | What it is |
| --- | --- | --- |
| `Train.csv` | 68,654 | Loans with the `target` label; Kenya only |
| `Test.csv` | 18,594 | Same columns minus `target`; Kenya + Ghana |
| `SampleSubmission.csv` | 18,594 | Format: `ID`, `target` (0/1) |
| `economic_indicators.csv` | 27 | 9 macro indicators × 3 countries (Ghana, Kenya, Côte d'Ivoire), yearly 2001–2023 |
| `VariableDefinitions.txt` | — | Column dictionary |
| `Starter_Notebook_.ipynb` | — | Baseline logistic regression (has bugs; see Traps) |

| Column | Type | Plain-language meaning | How to treat it |
| --- | --- | --- | --- |
| `ID` | key | Row id = customer\_id + loan\_id + lender\_id glued together | Never a feature |
| `customer_id` | key | The borrower | Group key for CV and history features; never a raw numeric feature |
| `country_id` | category | Kenya or Ghana | Drives currency and macro joins |
| `tbl_loan_id` | key | The loan; can repeat across lenders | Group key; never a raw feature |
| `lender_id` | category | Who funded this row's share | Categorical, but 4 Ghana lenders never appear in train |
| `loan_type` | category | Product (Type\_1…Type\_24) | Categorical; Type\_3 and Type\_8 appear only in test |
| `Total_Amount` | money (local currency) | Principal disbursed | Convert to USD or scale-free ratios |
| `Total_Amount_to_Repay` | money | Principal + interest + fees | Derive markup ratio |
| `disbursement_date` | date | When money went out | Vintage, macro year, customer sequence |
| `due_date` | date | Repayment deadline | Always disbursement + duration (checked: 0 mismatches) |
| `duration` | days | Tenor; 95% of train loans are 7-day | Numeric + bucketed |
| `New_versus_Repeat` | category | First loan or not | Strong signal; only 567 new loans in train |
| `Amount_Funded_By_Lender` | money | This lender's share of principal | Exposure for this row |
| `Lender_portion_Funded` | ratio 0–1 | Share of principal funded by this lender (mostly 0.3) | Numeric; 1 row is >1 (data error) |
| `Lender_portion_to_be_repaid` | money | This lender's share of what is owed | Approximates EAD |
| `target` | label | 1 = defaulted, 0 = repaid | 1.8% positive |

## Data structure: the grain

One row is one lender's share of one loan to one customer, not one customer and not one loan. Get this wrong and every downstream step leaks.

```
Customer (6,540 in train)
  └─ Loan  (66,520 distinct tbl_loan_id)
        └─ Lender share (68,654 rows)   ← one row
```

- **Customers repeat.** The median customer has 6 rows; the busiest has 208. Their loans form a time-ordered history you can learn from.
- **Loans can be split across lenders.** 2,134 loans have more than one lender row. The target is the same across a loan's rows (only 2 exceptions), so it is a loan-level outcome copied onto each share.
- **Customers' outcomes vary.** 634 of 6,540 customers have both repaid and defaulted loans, so "bad customer" is not a fixed label.
- **Test overlaps train by customer, not by loan.** 80% of test rows belong to customers seen in train, but only 4.9% of test loan ids appear in train. None of the 3,525 Ghana rows belong to a known customer.

Consequence: a random row split puts the same customer (and sometimes the same loan) in both train and validation, which inflates scores. Module 2 uses grouped and time-aware validation instead.

## What the data already tells us

Risk is concentrated: the 7-day, repeat-customer, Type\_1 loans from lender 267278 dominate volume and rarely default, while everything else defaults at 5–70%. Your notebook reproduces every number below.

| Segment | Train rows | Default rate |
| --- | --- | --- |
| All loans | 68,654 | 1.8% |
| New loan (first-time borrower) | 567 | 21.0% |
| Repeat loan | 68,087 | 1.7% |
| Duration 7 days | 64,973 | 1.2% |
| Duration 14 days | 1,567 | 8.2% |
| Duration 30 days | 958 | 15.7% |
| Duration 90 days | 249 | 23.7% |
| Lender 267278 | 64,653 | 1.2% |
| Lender 251804 | 3,542 | 8.9% |
| Lender 267277 | 271 | 29.5% |
| Lender 245684 | 188 | 41.5% |
| Loan Type\_1 | 61,723 | 0.8% |
| Loan Type\_4 | 1,235 | 12.3% |
| Loan Type\_14 | 99 | 71.7% |
| Loan Type\_2 | 74 | 60.8% |
| Markup (repay ÷ amount − 1), top 20% | 13,731 | 9.0% |
| Markup, bottom 80% | 54,923 | under 0.2% |

What this means for you:

1. **Pricing carries risk information.** Lenders already charge riskier customers more, so the markup ratio is one of your strongest features. Ask yourself whether that is leakage (it is not; the price is known at disbursement).
2. **Small categories, big rates.** Type\_14 has only 99 rows. Rates on tiny groups are noisy; use smoothing when you encode them (Module 3).
3. **Volume hides risk.** 94% of rows are the safe 7-day product, so a model can score well overall while being poor on the products that actually lose money. Always slice your errors by segment.
4. **Default rate depends on when the loan was made** (chart below).

&#91;embedded content: Train.csv · 68,654 rows · Oct 2021–Nov 2024\]

The overall 1.8% rate is an average of two different worlds: a huge, safe 2022 book and small, risky books before and after. A model trained mostly on 2022 Q3–Q4 may underestimate risk on later loans, which is why Module 2 adds a time-based validation fold.

## The Ghana problem

Ghana is 19% of the test set and shares almost nothing with training data: new lenders, mostly new products, longer tenors, another currency and no known customers. This is the real test of generalisation.

| Attribute | Train (Kenya) | Test Kenya | Test Ghana |
| --- | --- | --- | --- |
| Rows | 68,654 | 15,069 | 3,525 |
| Lenders | 267278, 251804, 267277, 245684 | Same four | 296542, 297183, 296540, 297182 (unseen) + 245684 |
| Main loan types | Type\_1 (90%) | Type\_1 | Type\_3 (3,039, unseen), Type\_2 (449), Type\_8 (37, unseen) |
| Durations | 7 days dominates | 7 days dominates | 7 days (3,039); 150–365 days for the rest |
| Currency | KES | KES | GHS (cedi) |
| Known customers | — | Most | None |
| Disbursement window | Oct 2021–Nov 2024 | Nov 2021–Nov 2024 | Nov 2021–Mar 2023 |

The currency trap: Ghana's average loan looks similar to Kenya's in raw numbers (about 15,500 vs 14,200), but 1 USD was about 118 KES and 8.3 GHS in 2022 (from `economic_indicators.csv`). A GHS 10,000 loan is roughly 14 times larger in real terms than a KES 10,000 loan. Feed raw amounts to a model and it will badly misjudge Ghana.

How we will handle it (Modules 2–3):

- Convert money to USD with the year's exchange rate, and prefer scale-free features (markup ratio, lender share, amount relative to the customer's own history).
- Do not let the model memorise `lender_id` or `loan_type` codes it will never see in Ghana; map them to broader traits (tenor bucket, pricing) or give unseen codes an "unknown" path.
- Build a validation fold that mimics Ghana: hold out whole lenders or loan types the model never saw.
- Accept that we cannot measure Ghana performance directly; we have no Ghana labels. Say this honestly in the model card.

## External data: economic indicators

The macro file describes each country's economy by year; it explains *why* the same borrower profile is riskier in Ghana than in Kenya, but it adds almost nothing within Kenya-only training data. Treat it as a cross-country context tool, not a magic feature.

The file is wide: one row per country × indicator, one column per year (`YR2001`…`YR2023`). Kenya and Ghana, latest years:

| Indicator | Kenya 2022 | Kenya 2023 | Ghana 2022 | Ghana 2023 | Why a lender cares |
| --- | --- | --- | --- | --- | --- |
| Inflation (annual %) | 7.7 | 7.7 | 31.3 | 38.1 | Erodes borrowers' real income |
| Exchange rate (LCU per USD) | 117.9 | 139.8 | 8.3 | 11.0 | Needed to put amounts on one scale |
| Deposit rate (%) | 7.1 | 9.2 | 12.1 | 11.4 | Lender's cost of funds |
| Lending rate (%) | 12.3 | 13.6 | missing | missing | Market price of credit |
| Real interest rate (%) | 6.0 | 6.5 | missing | missing | Lending rate minus inflation |
| Unemployment (%) | 5.8 | 5.7 | 3.1 | 3.1 | Income stability |
| Precipitation, fossil fuel use | constant or missing |  |  |  | Little value; drop |

How it connects to the loans:

1. Reshape it long (`melt`) to columns `country, indicator, year, value`, then pivot to one row per `country, year`.
2. Join on `country_id` and the **year of `disbursement_date`**. Use the year the loan was made, never a later year.
3. Data stops at 2023, but loans run to November 2024. Decide and document a rule: carry 2023 forward, or (better practice) use the *previous* year's value for every loan, since a lender would only know last year's published figure.
4. Fill Ghana's missing lending rate from public sources only if you can cite them; otherwise leave it out.

The honest caveat: within Kenya training data, each macro feature takes only 4 distinct values (2021–2024), so it behaves like a year flag. It may help explain the vintage swings, but it can also overfit. We will test it with and without in Module 3.

## Traps: leakage, starter-notebook bugs, validation

The starter notebook runs, but it would not survive a credit committee review: it uses ID numbers as features, scores the test set on unscaled data and validates on a random split.

Starter notebook issues to find and fix yourself:

1. **IDs as numbers.** `customer_id`, `tbl_loan_id` and `lender_id` are fed to the model as numeric features. An ID's size means nothing, and loan ids rise over time, so the model learns "when" through a back door.
2. **Scaling mismatch.** The model is trained on `StandardScaler` output but `predict` is called on unscaled `test_df`. The submission is effectively garbage.
3. **Label encoding then scaling categories.** Treating encoded `lender_id` as a continuous number imposes a fake order.
4. **Threshold 0.5.** Never tuned for F1.
5. **Random stratified split.** Same customers and loans land on both sides.

How much the leakage flatters you (my quick LightGBM check, 5-fold, same simple features, best-threshold F1):

| Validation | ID columns as features | Out-of-fold F1 | ROC AUC |
| --- | --- | --- | --- |
| Random split | Yes | 0.883 | 0.997 |
| Random split | No | 0.848 | 0.994 |
| Grouped by customer | Yes | 0.846 | 0.995 |
| Grouped by customer | No | 0.818 | 0.992 |

The last row is the honest number for Kenya, and it is still optimistic for Ghana. A 6-point F1 gap is the difference between a model that ships and one that embarrasses you in production.

Other leakage rules for this project:

- **Time travel.** Customer-history features (previous loans, past defaults) must use only loans disbursed *before* the current one. Never compute a customer's default rate over their whole history, including the current loan.
- **Target encoding inside folds only.** Encodings of `loan_type` or `lender_id` are fit on the training fold, applied to the validation fold.
- **Train + test concatenation** is fine for fitting encoders on categories, never for anything that touches `target`.
- **Macro data from the future.** Join by the disbursement year (or the year before), not by the due date year.

## Your Module 1 tasks

Work through `01_data_understanding.ipynb` in your project folder; every cell marked `# TODO` is yours. Bring your answers to our next session.

Hands-on:

- [ ] Set up a clean environment: `python -m venv .venv`, then `pip install pandas numpy matplotlib seaborn scikit-learn lightgbm jupyter`
- [ ] Run the notebook top to bottom and confirm every number in this doc
- [ ] Plot default rate by `duration` bucket and by markup decile; describe the shape in two sentences
- [ ] Reshape `economic_indicators.csv` to one row per country-year and join it to train and test by disbursement year
- [ ] Convert `Total_Amount` to USD and compare Kenya vs Ghana distributions on one chart
- [ ] List the 72 loans where `Total_Amount_to_Repay` < `Total_Amount` and the one row where `Lender_portion_Funded` > 1. Error, discount or refund? Decide how to treat them
- [ ] Find and write down the scaling bug in the starter notebook (cell and line)

Questions to answer in writing:

1. Why is accuracy the wrong metric here, and what does a 0.80 F1 actually mean to the asset manager?
2. Why does a random train/validation split overstate performance for this dataset?
3. Default rates were about 1% in Q3–Q4 2022 and 8–15% in most other quarters. Name two possible business explanations and how you would check each.
4. Ghana has no labels. How will you convince a credit committee your model is safe to use there?
5. Is the markup ratio a legitimate feature or leakage? Defend your answer.

Open questions for the client (a real project would ask these before modelling):

- How many days past due counts as default?
- What do the loan type codes represent (e.g., device loans, merchant float, payroll advances)?
- Are all amounts in local currency, and is the asset manager's exposure the lender portion only?
