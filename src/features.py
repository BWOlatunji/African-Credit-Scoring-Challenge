"""Module 3 feature pipeline for the African Credit Scoring Challenge.

Every feature here obeys two rules:
  1. Timing   - it uses only information available on the loan's disbursement date.
  2. Symmetry - it is computed identically for train and test rows.

Usage:
    from features import build_features
    feats = build_features(train, test, econ)   # one row per input row, train first then test
"""
import numpy as np
import pandas as pd

MACRO_KEEP = {
    'Inflation, consumer prices (annual %)': 'macro_inflation',
    'Official exchange rate (LCU per US$, period average)': 'fx_lcu_per_usd',
    'Deposit interest rate (%)': 'macro_deposit_rate',
    'Unemployment rate': 'macro_unemployment',
}
MARKUP_HINGE = 0.037   # from Module 1: default rate jumps above this markup


# ---------------------------------------------------------------- macro / FX
def macro_table(econ):
    """One row per (country_id, year) with the four usable indicators."""
    long = econ.melt(id_vars=['Country', 'Indicator'], var_name='year', value_name='value')
    long['year'] = long['year'].str.replace('YR', '').astype(int)
    long = long[long['Indicator'].isin(MACRO_KEEP)]
    wide = (long.assign(name=long['Indicator'].map(MACRO_KEEP))
                .pivot_table(index=['Country', 'year'], columns='name', values='value')
                .reset_index().rename(columns={'Country': 'country_id'}))
    return wide


def add_macro(df, macro, lag_years=1):
    """Join each loan to the indicators published `lag_years` before disbursement."""
    out = df[['country_id', 'disbursement_date']].copy()
    out['macro_year'] = (out['disbursement_date'].dt.year - lag_years).clip(upper=macro['year'].max())
    out = out.merge(macro, left_on=['country_id', 'macro_year'], right_on=['country_id', 'year'],
                    how='left').drop(columns=['year'])
    out.index = df.index
    return out.drop(columns=['country_id', 'disbursement_date', 'macro_year'])


# ---------------------------------------------------------------- loan-level basics
def loan_basics(df):
    f = pd.DataFrame(index=df.index)
    amount = df['Total_Amount'].clip(lower=1)
    f['log_amount_lcu'] = np.log1p(df['Total_Amount'])
    f['markup'] = df['Total_Amount_to_Repay'] / amount - 1
    f['duration'] = df['duration']
    f['daily_rate'] = f['markup'] / df['duration'].clip(lower=1)          # price per day of credit
    f['markup_high'] = (f['markup'] > MARKUP_HINGE).astype(int)           # the Module 1 cliff
    f['markup_excess'] = (f['markup'] - MARKUP_HINGE).clip(lower=0)       # how far past the cliff
    f['is_new'] = (df['New_versus_Repeat'] == 'New Loan').astype(int)
    f['lender_share'] = df['Lender_portion_Funded']
    return f


# ---------------------------------------------------------------- customer history
def customer_history(rows, label_lag_days=0):
    """History features from the customer's EARLIER loans only.

    rows: train and test stacked, with columns customer_id, tbl_loan_id, disbursement_date,
          due_date, Total_Amount, target (NaN for test rows).
    A prior loan's outcome counts as known only if due_date + label_lag_days is before
    the current loan's disbursement date, and only if it is a labelled (train) loan.
    Test loans' outcomes are unknown to everyone, so train and test are treated alike.
    """
    loans = (rows.groupby(['customer_id', 'tbl_loan_id'], as_index=False)
                 .agg(disb=('disbursement_date', 'first'), due=('due_date', 'first'),
                      amount=('Total_Amount', 'first'), target=('target', 'max')))
    loans = loans.sort_values(['customer_id', 'disb', 'tbl_loan_id']).reset_index(drop=True)
    lag = pd.Timedelta(days=label_lag_days)

    out = {k: np.full(len(loans), np.nan) for k in
           ['hist_n_prior', 'hist_days_since_last', 'hist_log_amount_vs_prior',
            'hist_n_known', 'hist_n_defaults', 'hist_default_rate']}
    for _, g in loans.groupby('customer_id', sort=False):
        idx = g.index.to_numpy()
        disb = g['disb'].to_numpy(); due = (g['due'] + lag).to_numpy()
        logamt = np.log1p(g['amount'].to_numpy()); tgt = g['target'].to_numpy()
        for j, i in enumerate(idx):
            prior = disb < disb[j]                      # strictly earlier loans only
            n = prior.sum()
            out['hist_n_prior'][i] = n
            if n:
                out['hist_days_since_last'][i] = (disb[j] - disb[prior].max()) / np.timedelta64(1, 'D')
                out['hist_log_amount_vs_prior'][i] = logamt[j] - logamt[prior].mean()
            known = prior & (due < disb[j]) & ~np.isnan(tgt)
            k = known.sum()
            out['hist_n_known'][i] = k
            out['hist_n_defaults'][i] = np.nansum(tgt[known]) if k else 0
    out['hist_default_rate'] = (out['hist_n_defaults'] + 0.018) / (out['hist_n_known'] + 1)  # smoothed toward base rate
    hist = pd.concat([loans[['customer_id', 'tbl_loan_id']], pd.DataFrame(out)], axis=1)
    merged = rows[['customer_id', 'tbl_loan_id']].merge(hist, on=['customer_id', 'tbl_loan_id'], how='left')
    merged.index = rows.index
    return merged.drop(columns=['customer_id', 'tbl_loan_id'])


# ---------------------------------------------------------------- everything
def build_features(train, test, econ, label_lag_days=0, macro_lag_years=1):
    rows = pd.concat([train.assign(_is_train=1), test.assign(_is_train=0, target=np.nan)],
                     ignore_index=True)
    f = loan_basics(rows)
    macro = add_macro(rows, macro_table(econ), lag_years=macro_lag_years)
    f['log_amount_usd'] = np.log1p(rows['Total_Amount'] / macro['fx_lcu_per_usd'])
    f = pd.concat([f, macro.drop(columns=['fx_lcu_per_usd']),
                   customer_history(rows, label_lag_days)], axis=1)
    f['lender_id'] = rows['lender_id'].astype(str).astype('category')
    f['loan_type'] = rows['loan_type'].astype('category')
    f['is_zero_funded'] = (rows['Amount_Funded_By_Lender'] == 0).astype(int)  # diagnostic only, never a feature
    f['_is_train'] = rows['_is_train'].to_numpy()
    return f


FAMILIES = {
    'baseline': ['log_amount_lcu', 'markup', 'duration', 'is_new', 'lender_share'],
    'pricing':  ['daily_rate', 'markup_high', 'markup_excess'],
    'history':  ['hist_n_prior', 'hist_days_since_last', 'hist_log_amount_vs_prior',
                 'hist_n_known', 'hist_n_defaults', 'hist_default_rate'],
    'usd':      ['log_amount_usd'],
    'macro':    ['macro_inflation', 'macro_deposit_rate', 'macro_unemployment'],
    'codes':    ['lender_id', 'loan_type'],
}
