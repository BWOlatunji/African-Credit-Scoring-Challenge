"""Module 2 validation protocol as reusable code (used from Module 3 on).

    from evaluate import Evaluator
    ev = Evaluator(train, test, feats, folds)       # feats from features.build_features
    result = ev.run('R1 baseline + pricing', cols)  # scores B, C, D, E + Ghana flag rate
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import date
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

THRESHOLDS = np.round(np.arange(0.02, 0.95, 0.01), 2)
TIME_CUT = pd.Timestamp('2023-01-01')


def best_threshold(y_true, p):
    scores = [f1_score(y_true, p >= t) for t in THRESHOLDS]
    k = int(np.argmax(scores))
    return THRESHOLDS[k], scores[k]


def make_model():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=0, verbose=-1)


class Evaluator:
    def __init__(self, train, test, feats, folds, model_factory=make_model):
        n = len(train)
        assert (folds['ID'].values == train['ID'].values).all(), 'folds.csv is not in Train.csv row order'
        self.X = feats.iloc[:n].reset_index(drop=True)
        self.X_test = feats.iloc[n:].reset_index(drop=True)
        self.y = train['target'].to_numpy()
        self.fold_loan = folds['fold_loan'].to_numpy()
        self.fold_customer = folds['fold_customer'].to_numpy()
        self.past = (train['disbursement_date'] < TIME_CUT).to_numpy()
        self.lender = train['lender_id'].to_numpy()
        self.test_country = test['country_id'].to_numpy()
        self.model_factory = model_factory

    def _cv(self, X, y, fold_ids, train_mask):
        """Out-of-fold predictions; rows outside train_mask are never trained on but are always scored."""
        oof = np.zeros(len(y))
        for k in np.unique(fold_ids):
            fit_rows = (fold_ids != k) & train_mask
            oof[fold_ids == k] = (self.model_factory().fit(X[fit_rows], y[fit_rows])
                                  .predict_proba(X[fold_ids == k])[:, 1])
        f1s = [f1_score(y[fold_ids == k], oof[fold_ids == k] >= best_threshold(y[fold_ids != k], oof[fold_ids != k])[0])
               for k in np.unique(fold_ids)]
        return oof, f1s

    def run(self, name, cols, train_mask=None, verbose=True):
        X, y = self.X[cols], self.y
        m = np.ones(len(y), bool) if train_mask is None else np.asarray(train_mask)
        r = {'experiment': name, 'n_features': len(cols)}

        oof_B, f1s_B = self._cv(X, y, self.fold_loan, m)                       # scheme B
        thr_B = best_threshold(y, oof_B)[0]
        r.update(f1_B=np.mean(f1s_B), f1_B_sd=np.std(f1s_B), roc_auc_B=roc_auc_score(y, oof_B),
                 pr_auc_B=average_precision_score(y, oof_B), threshold_B=thr_B)
        r['f1_B_folds'] = ' '.join(f'{v:.3f}' for v in f1s_B)

        _, f1s_C = self._cv(X, y, self.fold_customer, m)                        # scheme C
        r['f1_C'] = np.mean(f1s_C)

        past, fut = self.past, ~self.past                                       # scheme D
        oof_past, _ = self._cv(X[past].reset_index(drop=True), y[past], self.fold_loan[past], m[past])
        thr_past = best_threshold(y[past], oof_past)[0]
        p_fut = self.model_factory().fit(X[past & m], y[past & m]).predict_proba(X[fut])[:, 1]
        r.update(f1_D=f1_score(y[fut], p_fut >= thr_past), mean_pred_D=p_fut.mean(), actual_D=y[fut].mean())

        aucs, f1s, w = [], [], []                                               # scheme E
        for L in np.unique(self.lender):
            a, b = (self.lender != L) & m, self.lender == L
            p = self.model_factory().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
            aucs.append(roc_auc_score(y[b], p)); f1s.append(f1_score(y[b], p >= thr_B)); w.append(b.sum())
        r.update(auc_E_mean=np.mean(aucs), auc_E_min=np.min(aucs), f1_E_weighted=np.average(f1s, weights=w))

        p_test = self.model_factory().fit(X[m], y[m]).predict_proba(self.X_test[cols])[:, 1]  # test flag rates
        for c in ['Kenya', 'Ghana']:
            r[f'flag_rate_{c}'] = (p_test[self.test_country == c] >= thr_B).mean()

        self.last_oof_B, self.last_test_prob = oof_B, p_test
        if verbose:
            print(f"{name}: F1 B {r['f1_B']:.3f} ± {r['f1_B_sd']:.3f} | C {r['f1_C']:.3f} | D {r['f1_D']:.3f} | "
                  f"E AUC {r['auc_E_mean']:.3f} | Ghana flag {r['flag_rate_Ghana']:.1%}")
        return r


def log_experiment(row, path=Path('outputs') / 'experiments.csv'):
    """One row per experiment name; re-running replaces the old row."""
    row = {'date': str(date.today()), **row}
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old['experiment'] != row['experiment']], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new
