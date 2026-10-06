"""
=============================================================================
 FILE: services/rl_causal.py
=============================================================================
 Causal inference for intervention effect estimation.

 Problem:
   The observational dataset (UCI) contains no explicit intervention records.
   Naive correlations (graduate vs dropout trajectories) conflate treatment
   effect with self-selection bias — students who would have graduated anyway
   may appear more often in "positive trajectory" groups.

 Approach:
   Use proxy treatment variables already in the dataset:
     - scholarship_holder    → proxy for financial support received
     - tuition_up_to_date    → proxy for no financial distress
     - low grade_change      → proxy for academic support not received
   For each proxy, estimate Average Treatment Effect (ATE) using:
     1. Naive difference  (biased, correlational only)
     2. Inverse Propensity Weighting (IPW)       — Horvitz-Thompson
     3. Doubly Robust (DR) — robust to model misspecification

 Outputs:
   - CATE estimates per intervention (stored in models/causal_ate.json)
   - Sensitivity analysis (E-value bound for unobserved confounding)
   - Debias multiplier applied to RL rewards if |ATE_DR - ATE_naive| > threshold

 Reference:
   - Horvitz & Thompson 1952 (IPW)
   - Bang & Robins 2005 (Doubly Robust)
   - VanderWeele & Ding 2017 (E-value for sensitivity)
=============================================================================
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Proxy treatment definitions — mapping from RL action → observable dataset flag
TREATMENT_PROXIES = {
    'financial_aid': {
        'description': 'Students with scholarship or fully paid tuition receive effective financial support',
        'treatment_fn': lambda row: int(row.get('Scholarship holder', 0) == 1 or row.get('Tuition fees up to date', 0) == 1),
    },
    'mentoring': {
        'description': 'Students with above-median Sem 1 approval rate who improved in Sem 2 received effective mentoring',
        'treatment_fn': lambda row: int(row.get('Grade_change', 0) > 0 and row.get('Approval_rate_sem1', 0) > 0.5),
    },
}

# Confounders to adjust for
CONFOUNDERS = [
    'Age at enrollment',
    'Gender',
    'Marital status',
    'Debtor',
    'Displaced',
    'International',
    "Mother's qualification",
    "Father's qualification",
    'Application mode',
    'Previous qualification',
    'Course',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (grade)',
]


def _propensity_scores(df, treatment, confounders):
    """Fit logistic regression P(T=1 | X)."""
    X = df[confounders].astype(float).fillna(0).values
    T = df[treatment].astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_scaled, T)
    ps = model.predict_proba(X_scaled)[:, 1]
    ps = np.clip(ps, 0.02, 0.98)  # trim extreme weights
    return ps, model, scaler


def estimate_naive_ate(df, treatment_col, outcome_col):
    treated = df[df[treatment_col] == 1][outcome_col].mean()
    control = df[df[treatment_col] == 0][outcome_col].mean()
    return float(treated - control)


def estimate_ipw_ate(df, treatment_col, outcome_col, confounders):
    """Inverse Propensity Weighting ATE estimator."""
    ps, _, _ = _propensity_scores(df, treatment_col, confounders)
    T = df[treatment_col].values
    Y = df[outcome_col].values
    # Horvitz-Thompson
    w1 = T / ps
    w0 = (1 - T) / (1 - ps)
    ate = (w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum()
    # Effective sample size for stability check
    ess = (w1.sum() + w0.sum()) ** 2 / ((w1 ** 2).sum() + (w0 ** 2).sum())
    return float(ate), float(ess)


def estimate_dr_ate(df, treatment_col, outcome_col, confounders):
    """
    Doubly Robust ATE using augmented IPW.
    Requires: outcome model m(X, T) + propensity model e(X).
    Consistent if either model is correct.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    ps, _, _ = _propensity_scores(df, treatment_col, confounders)
    X = df[confounders].astype(float).fillna(0).values
    T = df[treatment_col].astype(int).values
    Y = df[outcome_col].astype(float).values

    # Outcome models m1(X) = E[Y|X,T=1], m0(X) = E[Y|X,T=0]
    X_t1 = X[T == 1]; Y_t1 = Y[T == 1]
    X_t0 = X[T == 0]; Y_t0 = Y[T == 0]
    if len(X_t1) < 20 or len(X_t0) < 20:
        return None, None  # insufficient overlap

    m1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_t1, Y_t1)
    m0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_t0, Y_t0)

    mu1 = m1.predict(X)
    mu0 = m0.predict(X)

    # DR formula
    dr_1 = mu1 + T * (Y - mu1) / ps
    dr_0 = mu0 + (1 - T) * (Y - mu0) / (1 - ps)
    ate = dr_1.mean() - dr_0.mean()

    # Bootstrap SE
    n = len(df)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, n, n)
        boots.append(dr_1[idx].mean() - dr_0[idx].mean())
    se = float(np.std(boots))

    return float(ate), se


def e_value(ate, se, outcome_baseline):
    """
    E-value bound for unobserved confounding (VanderWeele & Ding 2017).
    How strong would an unmeasured confounder need to be to explain the ATE?
    """
    if ate is None or se is None or outcome_baseline <= 0:
        return None
    # Convert ATE (absolute) to risk ratio approximation
    rr = 1 + abs(ate) / max(outcome_baseline, 1e-6)
    # E-value formula
    return float(rr + np.sqrt(rr * (rr - 1))) if rr > 1 else 1.0


def analyze_dataset(df, outcome_col='Risk_binary', confounders=None):
    """
    Run the full causal analysis on the dataset.
    Returns dict with ATE estimates for each proxy treatment.
    """
    if confounders is None:
        confounders = [c for c in CONFOUNDERS if c in df.columns]

    baseline_outcome = float(df[outcome_col].mean())
    results = {
        'baseline_outcome_rate': baseline_outcome,
        'confounders_used': confounders,
        'sample_size': int(len(df)),
        'interventions': {},
    }

    for action, info in TREATMENT_PROXIES.items():
        # Build treatment column
        t_col = f'_T_{action}'
        df_work = df.copy()
        df_work[t_col] = df_work.apply(info['treatment_fn'], axis=1)

        n_treated = int(df_work[t_col].sum())
        n_control = int((1 - df_work[t_col]).sum())
        if n_treated < 30 or n_control < 30:
            results['interventions'][action] = {'error': 'insufficient overlap'}
            continue

        naive = estimate_naive_ate(df_work, t_col, outcome_col)
        ipw, ess = estimate_ipw_ate(df_work, t_col, outcome_col, confounders)
        dr, se = estimate_dr_ate(df_work, t_col, outcome_col, confounders)

        bias = abs(dr - naive) if dr is not None else None
        evalue = e_value(dr, se, baseline_outcome)

        # Interpretation — negative ATE means intervention reduces dropout risk
        direction = 'reduces_risk' if dr is not None and dr < 0 else 'increases_risk'

        results['interventions'][action] = {
            'description': info['description'],
            'n_treated': n_treated,
            'n_control': n_control,
            'ate_naive': round(naive, 4),
            'ate_ipw': round(ipw, 4) if ipw is not None else None,
            'ess': round(ess, 1) if ess is not None else None,
            'ate_dr': round(dr, 4) if dr is not None else None,
            'dr_std_error': round(se, 4) if se is not None else None,
            'ci_95': [round(dr - 1.96 * se, 4), round(dr + 1.96 * se, 4)] if dr is not None and se is not None else None,
            'naive_minus_dr_bias': round(bias, 4) if bias is not None else None,
            'e_value': round(evalue, 2) if evalue is not None else None,
            'direction': direction,
        }

    return results


def compute_causal_reward_multipliers(causal_results):
    """
    Derive per-action reward multipliers from causal estimates.
    If DR ATE is much smaller in magnitude than naive ATE, penalize that action's reward
    (discount for likely self-selection bias).
    """
    multipliers = {'none': 1.0}
    for action, res in causal_results.get('interventions', {}).items():
        if 'error' in res or res.get('ate_dr') is None:
            multipliers[action] = 1.0
            continue
        naive = res['ate_naive']
        dr = res['ate_dr']
        # If naive magnifies the effect, discount reward
        if abs(naive) > 1e-6:
            ratio = abs(dr) / abs(naive)
            multipliers[action] = float(max(0.3, min(1.0, ratio)))
        else:
            multipliers[action] = 1.0
    return multipliers
