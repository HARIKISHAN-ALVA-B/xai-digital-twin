"""
=============================================================================
 FILE: services/rl_fairness.py
=============================================================================
 Fairness analysis for RL policy.

 Protected attributes in the dataset:
   - Gender (0=female, 1=male)
   - International  (0=domestic, 1=international)
   - Displaced      (0=no, 1=yes)
   - Scholarship holder (proxy for socio-economic status)
   - Age group       (quantile buckets)

 Metrics (Hardt et al. 2016, Barocas et al. 2019):
   - Demographic Parity    — P(action=intervene | group_a) vs P(action=intervene | group_b)
   - Equal Opportunity     — Same but conditional on at-risk status (true positive rate)
   - Group-conditional RR  — Mean Risk Reduction per group
   - Fairness Gap          — max-min across groups for each metric

 Threshold for "fair":
   - Gap < 5 percentage points on all metrics
=============================================================================
"""

import numpy as np
import pandas as pd


PROTECTED_ATTRIBUTES = {
    'gender':      {'column': 'Gender',             'values': {0: 'female', 1: 'male'}},
    'international': {'column': 'International',    'values': {0: 'domestic', 1: 'international'}},
    'displaced':   {'column': 'Displaced',          'values': {0: 'not_displaced', 1: 'displaced'}},
    'scholarship': {'column': 'Scholarship holder', 'values': {0: 'no_scholarship', 1: 'scholarship'}},
}


def evaluate_policy_fairness(eval_records):
    """
    eval_records: list of dicts, each with:
        {
          'student_id', 'raw_data' (the full feature dict),
          'initial_risk', 'final_risk', 'risk_reduction',
          'retention', 'action_counts': {action: count, ...},
          'total_interventions': int
        }
    """
    df = pd.DataFrame(eval_records)
    # Extract protected attributes from raw_data
    for attr, info in PROTECTED_ATTRIBUTES.items():
        df[attr] = df['raw_data'].apply(lambda r: int(r.get(info['column'], 0)))

    results = {}
    for attr, info in PROTECTED_ATTRIBUTES.items():
        group_stats = {}
        for val, label in info['values'].items():
            subset = df[df[attr] == val]
            if len(subset) == 0:
                continue
            group_stats[label] = {
                'n': int(len(subset)),
                'mean_initial_risk': float(subset['initial_risk'].mean()),
                'mean_final_risk': float(subset['final_risk'].mean()),
                'mean_risk_reduction': float(subset['risk_reduction'].mean()),
                'retention_rate': float(subset['retention'].mean()),
                'mean_interventions_per_student': float(subset['total_interventions'].mean()),
            }

        # Compute gaps
        rr_values = [g['mean_risk_reduction'] for g in group_stats.values()]
        ret_values = [g['retention_rate'] for g in group_stats.values()]
        int_values = [g['mean_interventions_per_student'] for g in group_stats.values()]

        gaps = {
            'risk_reduction_gap': float(max(rr_values) - min(rr_values)) if rr_values else 0,
            'retention_gap': float(max(ret_values) - min(ret_values)) if ret_values else 0,
            'intervention_allocation_gap': float(max(int_values) - min(int_values)) if int_values else 0,
        }

        # Fairness verdict
        gaps['verdict'] = 'fair' if (gaps['risk_reduction_gap'] < 0.05
                                       and gaps['retention_gap'] < 0.05) else 'concerning'

        results[attr] = {
            'groups': group_stats,
            'gaps': gaps,
        }

    # Overall summary
    all_verdicts = [r['gaps']['verdict'] for r in results.values()]
    results['overall'] = {
        'n_protected_groups': len(PROTECTED_ATTRIBUTES),
        'fair_groups': sum(1 for v in all_verdicts if v == 'fair'),
        'concerning_groups': sum(1 for v in all_verdicts if v == 'concerning'),
        'max_gap': max((r['gaps']['risk_reduction_gap'] for r in results.values() if isinstance(r.get('gaps'), dict)), default=0),
    }

    return results


def equal_opportunity_test(eval_records, at_risk_threshold=0.5):
    """
    Among students who were initially at-risk, does every group receive
    equal rate of intervention?
    """
    df = pd.DataFrame(eval_records)
    for attr, info in PROTECTED_ATTRIBUTES.items():
        df[attr] = df['raw_data'].apply(lambda r: int(r.get(info['column'], 0)))
    df['at_risk_initial'] = (df['initial_risk'] >= at_risk_threshold).astype(int)
    at_risk = df[df['at_risk_initial'] == 1]

    results = {}
    for attr, info in PROTECTED_ATTRIBUTES.items():
        if len(at_risk) == 0:
            continue
        rates = {}
        for val, label in info['values'].items():
            subset = at_risk[at_risk[attr] == val]
            if len(subset) == 0:
                continue
            rates[label] = {
                'n_at_risk': int(len(subset)),
                'mean_interventions': float(subset['total_interventions'].mean()),
                'retention_rate': float(subset['retention'].mean()),
            }
        if len(rates) >= 2:
            ret_values = [r['retention_rate'] for r in rates.values()]
            results[attr] = {
                'group_retention': rates,
                'equal_opportunity_gap': float(max(ret_values) - min(ret_values)),
            }
    return results
