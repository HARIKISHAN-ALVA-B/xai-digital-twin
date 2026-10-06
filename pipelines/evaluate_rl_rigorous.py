"""
=============================================================================
 FILE: pipelines/evaluate_rl_rigorous.py
=============================================================================
 Publication-grade evaluation of the RL policy.

 Produces results for the research paper:

   1. Statistical significance — paired t-test + Wilcoxon signed-rank
   2. Effect size               — Cohen's d (paired)
   3. Bootstrap 95% CIs         — 1000 resamples
   4. Ablation study            — remove each reward component
   5. Robustness                — transition noise sensitivity
   6. Fairness                  — per-group metrics
   7. Causal ATE                — naive vs IPW vs Doubly Robust

 Saves results to: models/rl_evaluation_rigorous.json

 Run: python pipelines/evaluate_rl_rigorous.py
=============================================================================
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.rl_environment import StudentEnvironment, ACTIONS, rollout
from services.rl_policies import (FQIAgent, no_action_policy,
                                      gpa_threshold_policy, random_policy)
from services.rl_fairness import evaluate_policy_fairness, equal_opportunity_test
from services.rl_causal import analyze_dataset, compute_causal_reward_multipliers


MODELS_DIR = 'models'
OUT_FILE = f'{MODELS_DIR}/rl_evaluation_rigorous.json'
N_EVAL = 300       # eval set size
N_BOOTSTRAP = 1000
N_ROBUSTNESS_SEEDS = 5
HORIZON = 12


# =====================================================================
#  Helpers
# =====================================================================

def load_artifacts():
    with open(f'{MODELS_DIR}/xgb_stage2.pkl', 'rb') as f:
        model_s2 = pickle.load(f)
    with open(f'{MODELS_DIR}/features_stage2.json') as f:
        features_s2 = json.load(f)
    effects = None
    if os.path.exists(f'{MODELS_DIR}/intervention_effects.json'):
        with open(f'{MODELS_DIR}/intervention_effects.json') as f:
            effects = json.load(f).get('effects')
    df = pd.read_csv('data/processed_data.csv')
    agent = FQIAgent.load(f'{MODELS_DIR}/rl_policy.pkl')
    return model_s2, features_s2, effects, df, agent


def make_env(row, model_s2, features_s2, effects, horizon=HORIZON,
             reward_weights=None, noise_std=0.0, seed=None):
    return StudentEnvironment(
        initial_raw=row.to_dict(),
        model_s2=model_s2,
        features_s2=features_s2,
        learned_effects=effects,
        horizon=horizon,
        reward_weights=reward_weights,
        noise_std=noise_std,
        seed=seed,
    )


def rollout_per_student(df_eval, policy_fn, **env_kwargs):
    """Return a list of per-student result dicts."""
    records = []
    for _, row in df_eval.iterrows():
        env = make_env(row, **env_kwargs)
        initial_risk = env.current_risk
        transitions = rollout(env, policy_fn)
        final_risk = env.current_risk
        total_reward = float(sum(t[2] for t in transitions))
        actions = [ACTIONS[t[1]] for t in transitions]
        action_counts = {a: actions.count(a) for a in ACTIONS}
        total_interventions = sum(v for a, v in action_counts.items() if a != 'none')
        records.append({
            'student_id': row.name if hasattr(row, 'name') else '',
            'raw_data': row.to_dict(),
            'initial_risk': float(initial_risk),
            'final_risk': float(final_risk),
            'risk_reduction': float(initial_risk - final_risk),
            'retention': int(final_risk < 0.5),
            'total_reward': total_reward,
            'action_counts': action_counts,
            'total_interventions': total_interventions,
        })
    return records


# =====================================================================
#  Statistical tests
# =====================================================================

def paired_stats(rl_values, baseline_values, label):
    """Return paired comparison statistics."""
    rl = np.array(rl_values)
    bl = np.array(baseline_values)
    diff = rl - bl

    # Paired t-test
    t_stat, t_pvalue = scipy_stats.ttest_rel(rl, bl)

    # Wilcoxon signed-rank (non-parametric fallback)
    try:
        w_stat, w_pvalue = scipy_stats.wilcoxon(rl, bl)
    except Exception:
        w_stat, w_pvalue = float('nan'), float('nan')

    # Cohen's d (paired)
    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0

    return {
        'comparison': label,
        'n': int(len(rl)),
        'rl_mean': float(rl.mean()),
        'baseline_mean': float(bl.mean()),
        'difference_mean': float(diff.mean()),
        'difference_std': float(diff.std(ddof=1)),
        't_stat': float(t_stat),
        't_pvalue': float(t_pvalue),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pvalue': float(w_pvalue),
        'cohens_d': float(cohens_d),
        'significant_at_0.01': bool(t_pvalue < 0.01),
        'significant_at_0.05': bool(t_pvalue < 0.05),
        'effect_size_label': (
            'negligible' if abs(cohens_d) < 0.2 else
            'small' if abs(cohens_d) < 0.5 else
            'medium' if abs(cohens_d) < 0.8 else
            'large'
        ),
    }


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=0.95):
    """Percentile bootstrap CI for the mean."""
    values = np.asarray(values)
    rng = np.random.default_rng(42)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# =====================================================================
#  Ablation: reward components
# =====================================================================

def reward_ablation(df_eval, df_train, env_kwargs_base, original_agent):
    """
    Proper ablation: retrain a fresh FQI agent with each ablated reward,
    then evaluate on the same eval set. This measures how each reward
    component contributes to the learned policy.
    """
    from services.rl_policies import FQIAgent
    ablations = {}
    full_weights = {'w_risk': 10.0, 'w_retention': 2.0, 'w_cost': 1.0, 'w_instability': 0.5}

    for remove in [None, 'w_risk', 'w_retention', 'w_cost', 'w_instability']:
        weights = dict(full_weights)
        if remove is not None:
            weights[remove] = 0.0

        # Generate replay buffer with ablated reward
        kwargs = dict(env_kwargs_base)
        kwargs['reward_weights'] = weights
        buffer = []
        for _, row in df_train.iterrows():
            for b_policy in [no_action_policy, random_policy, gpa_threshold_policy]:
                env = make_env(row, **kwargs)
                transitions = rollout(env, b_policy)
                for (s, a, r, s_next, done, _info) in transitions:
                    buffer.append((s, a, r, s_next, done))

        # Retrain agent on ablated reward
        abl_agent = FQIAgent(n_actions=len(ACTIONS), gamma=0.95, n_iterations=10)
        abl_agent.fit(buffer, verbose=False)

        # Evaluate on eval set with FULL reward (for fair comparison)
        eval_kwargs = dict(env_kwargs_base)
        eval_kwargs['reward_weights'] = full_weights
        records = rollout_per_student(df_eval, abl_agent.policy, **eval_kwargs)
        rr = [r['risk_reduction'] for r in records]
        ret = [r['retention'] for r in records]
        ablations[remove or 'full_reward'] = {
            'mean_risk_reduction': float(np.mean(rr)),
            'retention_rate': float(np.mean(ret)),
        }

    base = ablations['full_reward']
    for k, v in ablations.items():
        if k == 'full_reward':
            continue
        v['degradation_rr'] = float(base['mean_risk_reduction'] - v['mean_risk_reduction'])
        v['degradation_retention'] = float(base['retention_rate'] - v['retention_rate'])
    return ablations


# =====================================================================
#  Robustness: transition noise
# =====================================================================

def robustness_analysis(df_eval, env_kwargs_base, agent):
    """Evaluate under increasing transition noise."""
    noise_levels = [0.0, 0.02, 0.05, 0.10]
    results = {}
    for sigma in noise_levels:
        # Average over multiple seeds
        all_rr = []
        all_ret = []
        for seed in range(N_ROBUSTNESS_SEEDS):
            kwargs = dict(env_kwargs_base)
            kwargs['noise_std'] = sigma
            kwargs['seed'] = seed
            records = rollout_per_student(df_eval, agent.policy, **kwargs)
            all_rr.extend(r['risk_reduction'] for r in records)
            all_ret.extend(r['retention'] for r in records)
        results[f'sigma_{sigma}'] = {
            'noise_std': sigma,
            'mean_risk_reduction': float(np.mean(all_rr)),
            'risk_reduction_std': float(np.std(all_rr)),
            'retention_rate': float(np.mean(all_ret)),
        }
    return results


# =====================================================================
#  Main
# =====================================================================

def main():
    print('=' * 75)
    print('  RIGOROUS RL EVALUATION — Statistical Tests + Ablation + Fairness + Causal')
    print('=' * 75)

    model_s2, features_s2, effects, df, agent = load_artifacts()
    print(f'[LOAD] Stage-2 model, RL agent (trained: {agent.trained}), dataset={len(df)}')

    # Stratified eval sample
    np.random.seed(123)
    risky = df[df['Risk_binary'] == 1].sample(min(N_EVAL // 2, 700), random_state=123)
    safe  = df[df['Risk_binary'] == 0].sample(min(N_EVAL // 2, 700), random_state=123)
    df_eval = pd.concat([risky, safe]).sample(frac=1, random_state=123).reset_index(drop=True)
    df_eval = df_eval.head(N_EVAL)
    print(f'[SPLIT] Eval: {len(df_eval)} students '
          f'(at-risk ratio {df_eval["Risk_binary"].mean():.2f})\n')

    env_kwargs_base = dict(
        model_s2=model_s2, features_s2=features_s2, effects=effects, horizon=HORIZON,
    )

    # ---------- 1. Rollouts per policy ----------
    print('[1] Rolling out policies on eval set...')
    records = {}
    for policy, label in [
        (no_action_policy,     'B1_no_action'),
        (random_policy,        'B0_random'),
        (gpa_threshold_policy, 'B2_gpa_rule'),
        (agent.policy,         'RL_FQI'),
    ]:
        records[label] = rollout_per_student(df_eval, policy, **env_kwargs_base)
        rr = np.mean([r['risk_reduction'] for r in records[label]])
        ret = np.mean([r['retention'] for r in records[label]])
        print(f'    {label:20s}  RR={rr:+.4f}  retention={ret:.3f}')

    # ---------- 2. Statistical significance (RL vs each baseline) ----------
    print('\n[2] Statistical significance (paired tests)...')
    rl_rr = [r['risk_reduction'] for r in records['RL_FQI']]
    rl_ret = [r['retention'] for r in records['RL_FQI']]
    stat_tests = {}
    for baseline in ['B1_no_action', 'B0_random', 'B2_gpa_rule']:
        b_rr = [r['risk_reduction'] for r in records[baseline]]
        b_ret = [r['retention'] for r in records[baseline]]
        stat_tests[f'{baseline}_risk_reduction'] = paired_stats(rl_rr, b_rr, f'RL vs {baseline} (RR)')
        stat_tests[f'{baseline}_retention']      = paired_stats(rl_ret, b_ret, f'RL vs {baseline} (retention)')
        s = stat_tests[f'{baseline}_risk_reduction']
        print(f'    RL vs {baseline:15s}  '
              f'diff={s["difference_mean"]:+.4f}  '
              f't={s["t_stat"]:+.2f}  p={s["t_pvalue"]:.4g}  '
              f'd={s["cohens_d"]:+.3f} ({s["effect_size_label"]})  '
              f'{"***" if s["t_pvalue"] < 0.001 else "**" if s["t_pvalue"] < 0.01 else "*" if s["t_pvalue"] < 0.05 else "ns"}')

    # ---------- 3. Bootstrap CIs ----------
    print('\n[3] Bootstrap 95% CIs (1000 resamples)...')
    cis = {}
    for label, recs in records.items():
        rr = [r['risk_reduction'] for r in recs]
        lo, hi = bootstrap_ci(rr)
        cis[label] = {'risk_reduction_mean': float(np.mean(rr)),
                      'ci_95': [lo, hi]}
        print(f'    {label:20s}  RR={np.mean(rr):+.4f}  [95% CI: {lo:+.4f}, {hi:+.4f}]')

    # ---------- 4. Ablation ----------
    print('\n[4] Reward ablation (retraining with each component removed)...')
    # Use a smaller training sample for ablation (speed)
    train_risky = df[df['Risk_binary'] == 1].sample(min(100, 500), random_state=42)
    train_safe  = df[df['Risk_binary'] == 0].sample(min(100, 500), random_state=42)
    df_train_abl = pd.concat([train_risky, train_safe]).sample(frac=1, random_state=42).head(200)
    abl = reward_ablation(df_eval, df_train_abl, env_kwargs_base, agent)
    for k, v in abl.items():
        deg_rr = v.get('degradation_rr', 0)
        deg_ret = v.get('degradation_retention', 0)
        print(f'    remove {k:20s}  RR={v["mean_risk_reduction"]:+.4f}  '
              f'retention={v["retention_rate"]:.3f}  '
              f'(delta_RR={deg_rr:+.4f}, delta_ret={deg_ret:+.4f})')

    # ---------- 5. Robustness ----------
    print('\n[5] Robustness under transition noise...')
    robust = robustness_analysis(df_eval, env_kwargs_base, agent)
    for k, v in robust.items():
        print(f'    sigma={v["noise_std"]:.2f}  RR={v["mean_risk_reduction"]:+.4f}  '
              f'std={v["risk_reduction_std"]:.4f}  retention={v["retention_rate"]:.3f}')

    # ---------- 6. Fairness ----------
    print('\n[6] Fairness analysis...')
    fairness = evaluate_policy_fairness(records['RL_FQI'])
    for attr, res in fairness.items():
        if attr == 'overall':
            print(f'    [overall] fair_groups={res["fair_groups"]}/{res["n_protected_groups"]}  '
                  f'max_gap={res["max_gap"]:.4f}')
            continue
        gaps = res['gaps']
        print(f'    {attr:15s}  RR_gap={gaps["risk_reduction_gap"]:.4f}  '
              f'ret_gap={gaps["retention_gap"]:.4f}  '
              f'intervention_gap={gaps["intervention_allocation_gap"]:.3f}  '
              f'[{gaps["verdict"]}]')
    equal_opp = equal_opportunity_test(records['RL_FQI'])

    # ---------- 7. Causal analysis ----------
    print('\n[7] Causal ATE estimation (naive vs IPW vs Doubly Robust)...')
    causal = analyze_dataset(df)
    for action, res in causal.get('interventions', {}).items():
        if 'error' in res:
            print(f'    {action}: {res["error"]}')
            continue
        print(f'    {action:15s}  naive={res["ate_naive"]:+.4f}  '
              f'IPW={res["ate_ipw"]:+.4f}  '
              f'DR={res["ate_dr"]:+.4f} [±{res["dr_std_error"]:.4f}]  '
              f'bias={res["naive_minus_dr_bias"]:+.4f}  '
              f'E-value={res["e_value"]}')

    reward_multipliers = compute_causal_reward_multipliers(causal)
    print(f'    Suggested causal reward multipliers: {reward_multipliers}')

    # ---------- Save ----------
    full_results = {
        'summary': {
            'n_eval': int(len(df_eval)),
            'horizon_weeks': HORIZON,
            'rl_mean_risk_reduction': float(np.mean(rl_rr)),
            'rl_retention_rate': float(np.mean(rl_ret)),
        },
        'statistical_tests': stat_tests,
        'bootstrap_cis': cis,
        'reward_ablation': abl,
        'robustness': robust,
        'fairness': fairness,
        'equal_opportunity': equal_opp,
        'causal_ate': causal,
        'causal_reward_multipliers': reward_multipliers,
    }

    with open(OUT_FILE, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f'\n[SAVE] Full rigorous evaluation -> {OUT_FILE}')
    print('=' * 75)


if __name__ == '__main__':
    main()
