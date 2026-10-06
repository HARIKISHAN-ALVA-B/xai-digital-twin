"""
=============================================================================
 FILE: services/rl_explainer.py
=============================================================================
 Explainable Reinforcement Learning (XRL) module.

 Provides three layers of explanation for RL policy decisions:

   1. SHAP attribution on Q-values — which features drove the action choice
   2. Counterfactual analysis     — "what if we'd chosen the 2nd-best action"
   3. Natural-language template    — human-readable explanation

 Used by:
   - /api/student/{id}/optimal-plan/explain
   - Counselor UI for transparency
   - Research paper policy analysis
=============================================================================
"""

import numpy as np
import shap
from copy import deepcopy
from .rl_environment import ACTIONS, STATE_FEATURES, StudentEnvironment


# State feature names in the same order as StudentEnvironment._get_state_vector()
FULL_FEATURE_NAMES = (
    list(STATE_FEATURES)
    + ['Predicted_risk', 'Week_progress']
    + [f'History_{a}' for a in ACTIONS]
)

# Plain-language mapping for NL explanations
PLAIN = {
    'Curricular units 1st sem (grade)': 'Semester 1 grade',
    'Curricular units 2nd sem (grade)': 'Semester 2 grade',
    'Curricular units 1st sem (approved)': 'Semester 1 subjects passed',
    'Curricular units 2nd sem (approved)': 'Semester 2 subjects passed',
    'Curricular units 1st sem (without evaluations)': 'missed Semester 1 exams',
    'Curricular units 2nd sem (without evaluations)': 'missed Semester 2 exams',
    'Approval_rate_sem1': 'Semester 1 pass rate',
    'Approval_rate_sem2': 'Semester 2 pass rate',
    'Failure_rate_sem1': 'Semester 1 failure rate',
    'Failure_rate_sem2': 'Semester 2 failure rate',
    'Stress_indicator': 'academic stress level',
    'Grade_change': 'grade change between semesters',
    'Approved_change': 'change in subjects passed',
    'Debtor': 'debt status',
    'Tuition fees up to date': 'tuition payment status',
    'Scholarship holder': 'scholarship status',
    'Age at enrollment': 'age at enrollment',
    'Gender': 'demographic profile',
    'Predicted_risk': 'current predicted risk',
    'Week_progress': 'point in the semester',
}

ACTION_PLAIN = {
    'none': 'no intervention',
    'mentoring': 'mentoring',
    'financial_aid': 'financial aid',
}


def plain_name(feature):
    return PLAIN.get(feature, feature.replace('_', ' '))


class RLExplainer:
    """Post-hoc explanations for a trained FQIAgent."""

    def __init__(self, agent):
        self.agent = agent
        self._shap_explainers = {}  # per-action SHAP explainer cache

    def _get_explainer(self, action_idx):
        """Lazily build a SHAP TreeExplainer for a given action's Q-model."""
        if action_idx not in self._shap_explainers:
            model = self.agent.q_models[action_idx]
            if model is None:
                return None
            self._shap_explainers[action_idx] = shap.TreeExplainer(model)
        return self._shap_explainers[action_idx]

    # ---------------------------------------------------------------
    #  Layer 1 — SHAP on Q-values
    # ---------------------------------------------------------------

    def shap_attribution(self, state_vec, action_idx):
        """Return SHAP values for the chosen action's Q-function."""
        explainer = self._get_explainer(action_idx)
        if explainer is None:
            return None
        state_2d = state_vec.reshape(1, -1)
        shap_vals = explainer.shap_values(state_2d)[0]
        return {
            'base_value': float(explainer.expected_value),
            'contributions': [
                {'feature': FULL_FEATURE_NAMES[i],
                 'value': float(state_vec[i]),
                 'shap': float(shap_vals[i])}
                for i in range(len(FULL_FEATURE_NAMES))
            ],
        }

    # ---------------------------------------------------------------
    #  Layer 2 — Counterfactual rollout
    # ---------------------------------------------------------------

    def counterfactual_action(self, env_factory, state_vec, alt_action_idx,
                               remaining_weeks):
        """
        Simulate: "what if we'd chosen alt_action at this step and followed
        the agent's policy afterwards?"
        Returns projected final risk under the counterfactual.
        """
        env = env_factory()
        # Fast-forward env to current state... since env is stateful,
        # we assume env_factory returns an env already at the right state.
        # For simplicity, we instantiate a fresh env and use alt_action once,
        # then greedy policy for remaining weeks.
        env.reset()
        # Apply alt action for 1 step, then greedy for the rest
        alt_state, alt_reward, done, info = env.step(alt_action_idx)
        trajectory = [{'week': info['week'], 'action': info['action'],
                       'risk': info['risk']}]
        while not done and env.t < env.horizon:
            a_idx = self.agent.policy(alt_state)
            alt_state, _, done, info = env.step(a_idx)
            trajectory.append({'week': info['week'], 'action': info['action'],
                               'risk': info['risk']})
        return {
            'alt_action': ACTIONS[alt_action_idx],
            'final_risk': float(env.current_risk),
            'trajectory': trajectory,
        }

    # ---------------------------------------------------------------
    #  Layer 3 — Natural-language explanation
    # ---------------------------------------------------------------

    def explain_action(self, state_vec, env_factory, top_k=4):
        """
        Full explanation bundle for one decision.
        """
        q_vals = self.agent.q_values(state_vec)
        chosen = int(np.argmax(q_vals))
        chosen_name = ACTIONS[chosen]
        sorted_indices = np.argsort(q_vals)[::-1]
        second = int(sorted_indices[1]) if len(sorted_indices) > 1 else chosen
        margin = float(q_vals[chosen] - q_vals[second])

        # SHAP attribution
        shap_result = self.shap_attribution(state_vec, chosen)
        top_factors = []
        if shap_result:
            ranked = sorted(shap_result['contributions'],
                            key=lambda x: abs(x['shap']), reverse=True)
            top_factors = ranked[:top_k]

        # Counterfactual for second-best action
        cf = None
        if second != chosen and env_factory is not None:
            try:
                cf = self.counterfactual_action(env_factory, state_vec, second,
                                                 env_factory().horizon)
            except Exception:
                cf = None

        nl = self._natural_language(chosen_name, top_factors, second,
                                     ACTIONS[second], margin, cf)

        return {
            'chosen_action': chosen_name,
            'q_values': {ACTIONS[i]: float(q_vals[i]) for i in range(len(ACTIONS))},
            'decision_margin': margin,
            'second_best': ACTIONS[second],
            'top_factors': [
                {'feature': plain_name(f['feature']),
                 'raw_feature': f['feature'],
                 'value': round(f['value'], 4),
                 'shap': round(f['shap'], 4),
                 'direction': 'supports this action' if f['shap'] > 0 else 'discourages this action'}
                for f in top_factors
            ],
            'counterfactual': cf,
            'explanation': nl,
        }

    def _natural_language(self, action, top_factors, second_idx, second_name,
                          margin, counterfactual):
        parts = []
        parts.append(f"The system recommends {ACTION_PLAIN.get(action, action)} for this week.")

        if top_factors:
            pos = [f for f in top_factors if f['shap'] > 0][:2]
            neg = [f for f in top_factors if f['shap'] < 0][:2]
            if pos:
                reasons = ', '.join(plain_name(f['feature']) for f in pos)
                parts.append(f"Key drivers: {reasons}.")
            if neg:
                reasons = ', '.join(plain_name(f['feature']) for f in neg)
                parts.append(f"Also considered: {reasons}.")

        if margin > 0.5:
            parts.append(f"The decision margin over the second-best option ({ACTION_PLAIN.get(second_name, second_name)}) is strong.")
        elif margin > 0.1:
            parts.append(f"The second-best option ({ACTION_PLAIN.get(second_name, second_name)}) is a close alternative.")
        else:
            parts.append(f"The decision is borderline — {ACTION_PLAIN.get(second_name, second_name)} would be nearly equivalent.")

        if counterfactual:
            parts.append(
                f"If {ACTION_PLAIN.get(second_name, second_name)} had been chosen instead, "
                f"projected final risk would be {counterfactual['final_risk']*100:.1f}%."
            )

        return ' '.join(parts)

    # ---------------------------------------------------------------
    #  Sequence-level explanation
    # ---------------------------------------------------------------

    def explain_sequence(self, plan):
        """
        Aggregate-level explanation of a full intervention plan.
        plan: list of {week, action, projected_risk} dicts from optimal-plan.
        """
        action_counts = {a: 0 for a in ACTIONS}
        for step in plan:
            action_counts[step['action']] += 1

        dominant = max(action_counts, key=action_counts.get)
        initial_risk = plan[0].get('projected_risk_before', plan[0]['projected_risk'])
        final_risk = plan[-1]['projected_risk']
        reduction = initial_risk - final_risk

        # Detect strategy pattern
        if action_counts['none'] == len(plan):
            strategy = 'observational — no active intervention'
        elif action_counts[dominant] == len(plan):
            strategy = f'single-intervention ({ACTION_PLAIN[dominant]} throughout)'
        else:
            mix = ', '.join(f"{v} weeks of {ACTION_PLAIN[a]}" for a, v in action_counts.items() if v > 0)
            strategy = f'mixed strategy ({mix})'

        # Natural phrasing based on reduction direction
        reduction_pp = reduction * 100
        if reduction_pp > 0.5:
            change_text = f"Projected risk reduction: {reduction_pp:.1f} percentage points"
        elif reduction_pp < -0.5:
            change_text = f"Projected risk increase: {abs(reduction_pp):.1f} percentage points (no intervention recommended)"
        else:
            change_text = "Risk expected to remain stable"

        summary = (
            f"Plan strategy: {strategy}. "
            f"{change_text} over {len(plan)} weeks."
        )

        return {
            'strategy': strategy,
            'summary': summary,
            'action_distribution': action_counts,
            'dominant_action': dominant,
            'projected_reduction_pp': round(reduction * 100, 1),
        }
