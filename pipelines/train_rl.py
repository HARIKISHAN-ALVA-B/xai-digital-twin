"""
=============================================================================
 FILE: pipelines/train_rl.py
=============================================================================
 Offline training + evaluation of the RL policy for intervention optimization.

 Procedure:
   1. Load trained Stage-2 model + digital twin effects from models/
   2. Generate a replay buffer by rolling out a mixed-behavior policy
      through the twin for each student (or a sample)
   3. Train FQI agent on the replay buffer
   4. Evaluate learned policy vs baselines (no-action, GPA-rule, random)
   5. Save agent to models/rl_policy.pkl

 Run: python pipelines/train_rl.py
=============================================================================
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.rl_environment import StudentEnvironment, rollout, ACTIONS, ACTION_TO_IDX
from services.rl_policies import (FQIAgent, no_action_policy, gpa_threshold_policy,
                                      random_policy, financial_targeted_policy)


MODELS_DIR = 'models'
OUT_POLICY = 'models/rl_policy.pkl'
OUT_METRICS = 'models/rl_metrics.json'

HORIZON = 12         # weeks per episode (shorter = faster training)
N_TRAIN_STUDENTS = 400   # sample size for training trajectories
N_EVAL_STUDENTS = 200    # sample size for evaluation
N_ROLLOUTS_PER_STUDENT = 4  # different behavior policies per student (includes financial_targeted)
FQI_ITERATIONS = 15


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
    return model_s2, features_s2, effects, df


def sample_behavior_policy(idx):
    """Return a different behavior policy for each index (mixing = diverse data).
    Including financial_targeted_policy ensures financial_aid action has enough
    coverage in the replay buffer (otherwise Q-values for it are poorly calibrated)."""
    if idx == 0:
        return no_action_policy, 'none'
    elif idx == 1:
        return random_policy, 'random'
    elif idx == 2:
        return gpa_threshold_policy, 'gpa_rule'
    else:
        return financial_targeted_policy, 'financial_targeted'


def make_env(row, model_s2, features_s2, effects, horizon=HORIZON):
    return StudentEnvironment(
        initial_raw=row.to_dict(),
        model_s2=model_s2,
        features_s2=features_s2,
        learned_effects=effects,
        horizon=horizon,
    )


def generate_replay_buffer(df_sample, model_s2, features_s2, effects):
    """Roll out behavior policies across students; collect (s, a, r, s', done) tuples."""
    buffer = []
    n_students = len(df_sample)
    for i, (_, row) in enumerate(df_sample.iterrows()):
        for b in range(N_ROLLOUTS_PER_STUDENT):
            policy, label = sample_behavior_policy(b)
            env = make_env(row, model_s2, features_s2, effects)
            transitions = rollout(env, policy, seed=i * 10 + b)
            for (s, a, r, s_next, done, _info) in transitions:
                buffer.append((s, a, r, s_next, done))
        if (i + 1) % 50 == 0 or i == n_students - 1:
            print(f"  [replay] {i+1}/{n_students} students   buffer={len(buffer)} transitions")
    return buffer


def evaluate_policy(df_sample, model_s2, features_s2, effects, policy_fn, label):
    """Return mean metrics over the sample."""
    risks_init = []
    risks_final = []
    returns = []
    action_counts = {a: 0 for a in ACTIONS}
    for _, row in df_sample.iterrows():
        env = make_env(row, model_s2, features_s2, effects)
        initial_risk = env.current_risk
        transitions = rollout(env, policy_fn)
        final_risk = env.current_risk
        total_reward = sum(t[2] for t in transitions)
        for (_, a, _, _, _, _) in transitions:
            action_counts[ACTIONS[a]] += 1
        risks_init.append(initial_risk)
        risks_final.append(final_risk)
        returns.append(total_reward)

    risks_init = np.array(risks_init)
    risks_final = np.array(risks_final)
    returns = np.array(returns)

    metrics = {
        'policy': label,
        'mean_initial_risk': float(risks_init.mean()),
        'mean_final_risk': float(risks_final.mean()),
        'mean_risk_reduction': float((risks_init - risks_final).mean()),
        'retention_rate': float((risks_final < 0.5).mean()),
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'action_distribution': {a: int(v) for a, v in action_counts.items()},
    }
    return metrics


def main():
    print('=' * 70)
    print('  RL Training Pipeline — Fitted Q-Iteration on Digital Twin')
    print('=' * 70)

    model_s2, features_s2, effects, df = load_artifacts()
    print(f"[LOAD] Stage-2 model, {len(features_s2)} features, "
          f"{'learned' if effects else 'default'} intervention effects")
    print(f"[LOAD] Dataset: {len(df)} students\n")

    # Sample train/eval students (stratify on binary risk)
    np.random.seed(42)
    risky = df[df['Risk_binary'] == 1].sample(min(N_TRAIN_STUDENTS // 2, 700), random_state=42)
    safe  = df[df['Risk_binary'] == 0].sample(min(N_TRAIN_STUDENTS // 2, 700), random_state=42)
    df_train = pd.concat([risky, safe]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = df_train.head(N_TRAIN_STUDENTS)

    eval_risky = df[df['Risk_binary'] == 1].sample(min(N_EVAL_STUDENTS // 2, 300), random_state=123)
    eval_safe  = df[df['Risk_binary'] == 0].sample(min(N_EVAL_STUDENTS // 2, 300), random_state=123)
    df_eval = pd.concat([eval_risky, eval_safe]).sample(frac=1, random_state=123).reset_index(drop=True)
    df_eval = df_eval.head(N_EVAL_STUDENTS)

    print(f"[SPLIT] Train: {len(df_train)} students, Eval: {len(df_eval)} students")
    print(f"[SPLIT] Train at-risk ratio: {df_train['Risk_binary'].mean():.2f}\n")

    # Generate replay buffer
    print('[PHASE 1] Generating replay buffer...')
    buffer = generate_replay_buffer(df_train, model_s2, features_s2, effects)
    print(f"[PHASE 1] Buffer complete: {len(buffer)} transitions\n")

    # Train FQI
    print('[PHASE 2] Training FQI agent...')
    agent = FQIAgent(n_actions=len(ACTIONS), gamma=0.95, n_iterations=FQI_ITERATIONS)
    agent.fit(buffer, verbose=True)
    print('[PHASE 2] Training complete\n')

    # Evaluate policies
    print('[PHASE 3] Evaluating policies on eval set...')
    print('-' * 70)
    all_metrics = {}
    for policy, label in [
        (no_action_policy,     'B1_no_action'),
        (random_policy,        'B0_random'),
        (gpa_threshold_policy, 'B2_gpa_rule'),
        (agent.policy,         'RL_FQI'),
    ]:
        m = evaluate_policy(df_eval, model_s2, features_s2, effects, policy, label)
        all_metrics[label] = m
        print(f"  {label:20s}  "
              f"init_risk={m['mean_initial_risk']:.3f}  "
              f"final={m['mean_final_risk']:.3f}  "
              f"reduction={m['mean_risk_reduction']:+.3f}  "
              f"retention={m['retention_rate']:.3f}  "
              f"return={m['mean_return']:+.2f}")
        print(f"      action_dist: {m['action_distribution']}")
    print('-' * 70)

    # Compute RL improvement over best baseline
    rl_m = all_metrics['RL_FQI']
    best_baseline_reduction = max(
        all_metrics['B1_no_action']['mean_risk_reduction'],
        all_metrics['B2_gpa_rule']['mean_risk_reduction'],
    )
    gain = rl_m['mean_risk_reduction'] - best_baseline_reduction
    print(f"\n[RESULT] RL risk reduction gain over best baseline: {gain*100:+.1f} pp")

    # Save
    agent.save(OUT_POLICY)
    with open(OUT_METRICS, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n[SAVE] Policy saved to {OUT_POLICY}")
    print(f"[SAVE] Metrics saved to {OUT_METRICS}")
    print('=' * 70)


if __name__ == '__main__':
    main()
