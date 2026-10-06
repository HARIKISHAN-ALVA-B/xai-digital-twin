"""
=============================================================================
 FILE: modules/transition_model.py
=============================================================================
 Unified Transition Dynamics:  Sₜ + Aₜ  →  Sₜ₊₁

 This module is the SINGLE SOURCE OF TRUTH for how a student's state
 evolves under intervention actions.  Both the RL environment and the
 Digital Twin's week-by-week simulation delegate here.

 Architecture:
     ┌─────────────────────────────────┐
     │        TransitionModel          │
     │                                 │
     │  state_dict + action            │
     │       │                         │
     │       ▼                         │
     │  accumulate deltas (± noise)    │
     │  apply caps                     │
     │  update academic features       │
     │  update stress / financial      │
     │  recompute derived features     │
     │  predict risk via ML model      │
     │       │                         │
     │       ▼                         │
     │  (new_state, risk, grade, appr) │
     └─────────────────────────────────┘

 Consumers:
   - services/rl_environment.py  (RL training + rollouts)
   - modules/digital_twin.py     (dashboard simulation)
=============================================================================
"""

import numpy as np
import pandas as pd


# ── Default weekly effects (single source of truth) ──
# Used when data-driven learned_effects are not available.
# Keys: action name → per-week feature deltas
DEFAULT_EFFECTS = {
    'none':          {'grade_delta': -0.08, 'approved_delta': -0.04, 'eval_delta': 0.0,   'stress_delta': +0.10, 'tuition_paid_delta': 0.0},
    'mentoring':     {'grade_delta': +0.18, 'approved_delta': +0.12, 'eval_delta': +0.10, 'stress_delta': -0.04, 'tuition_paid_delta': 0.0},
    'financial_aid': {'grade_delta': +0.02, 'approved_delta': +0.01, 'eval_delta': +0.01, 'stress_delta': -0.20, 'tuition_paid_delta': 0.20},
}


class TransitionModel:
    """
    Stateful transition model:  Sₜ + Aₜ → Sₜ₊₁

    Tracks cumulative intervention effects relative to frozen baselines,
    applies domain constraints, updates all features (academic, stress,
    financial, derived), and re-evaluates risk via the ML model.

    Usage:
        tm = TransitionModel(initial_state, model, features, effects)
        new_state, risk, grade, approved = tm.step(state, 'mentoring')
    """

    def __init__(self, initial_state, risk_model, model_features,
                 effects, max_shift=None):
        """
        Args:
            initial_state: dict of raw feature values for the student
            risk_model:    trained classifier with .predict_proba()
            model_features: list of feature names expected by risk_model
            effects:       dict mapping action names → per-week delta dicts
            max_shift:     dict with cumulative caps {'grade', 'approved', 'eval'}
        """
        self.risk_model = risk_model
        self.model_features = model_features
        self.effects = effects
        self.max_shift = max_shift or {
            'grade': 3.0, 'approved': 3.0, 'eval': 3.0,
        }

        # ── Freeze baselines (all deltas measured relative to these) ──
        self.base_grade    = initial_state.get('Curricular units 2nd sem (grade)', 0)
        self.base_approved = initial_state.get('Curricular units 2nd sem (approved)', 0)
        self.base_evals    = initial_state.get('Curricular units 2nd sem (evaluations)', 0)
        self.base_no_eval  = initial_state.get('Curricular units 2nd sem (without evaluations)', 0)
        self.enrolled      = max(initial_state.get('Curricular units 2nd sem (enrolled)', 6), 1)
        self.sem1_grade    = initial_state.get('Curricular units 1st sem (grade)', 0)
        self.sem1_approved = initial_state.get('Curricular units 1st sem (approved)', 0)

        # ── Cumulative deltas (reset per episode / simulation) ──
        self.cum_grade    = 0.0
        self.cum_approved = 0.0
        self.cum_eval     = 0.0

    def reset(self):
        """Reset cumulative deltas for a new simulation episode."""
        self.cum_grade    = 0.0
        self.cum_approved = 0.0
        self.cum_eval     = 0.0

    # ─────────────────────────────────────────────────────────────
    #  Core transition:  Sₜ + Aₜ  →  Sₜ₊₁
    # ─────────────────────────────────────────────────────────────

    def step(self, state_dict, action, noise_std=0.0, rng=None):
        """
        Apply one week of transition dynamics.

        Args:
            state_dict: mutable dict of feature values (modified in-place)
            action:     action name string (e.g. 'mentoring')
            noise_std:  std-dev of Gaussian noise on deltas (for robustness)
            rng:        numpy random generator (required if noise_std > 0)

        Returns:
            (state_dict, risk, grade, approved)
        """
        fx = self.effects.get(action, self.effects.get('none', {}))

        # Optional noise for stochastic transitions during RL training
        noise = (lambda: rng.normal(0, noise_std)) if (noise_std > 0 and rng) else (lambda: 0.0)

        # ── 1. Accumulate action effects ──
        self.cum_grade    += fx.get('grade_delta', 0)    + noise()
        self.cum_approved += fx.get('approved_delta', 0) + noise()
        self.cum_eval     += fx.get('eval_delta', 0)     + noise()

        # ── 2. Apply cumulative caps ──
        ms = self.max_shift
        self.cum_grade    = max(-ms['grade'],    min(ms['grade'],    self.cum_grade))
        self.cum_approved = max(-ms['approved'], min(ms['approved'], self.cum_approved))
        self.cum_eval     = max(-ms['eval'],     min(ms['eval'],     self.cum_eval))

        # ── 3. Compute new academic features from baseline + deltas ──
        grade    = max(0, min(20, self.base_grade + self.cum_grade))
        evals    = max(1, min(self.enrolled * 2, self.base_evals + self.cum_eval))
        approved = max(0, min(self.enrolled, self.base_approved + self.cum_approved))
        approved = min(approved, evals)   # can't approve more than evaluated
        no_eval  = max(0, self.base_no_eval - self.cum_eval)

        state_dict['Curricular units 2nd sem (grade)']                = grade
        state_dict['Curricular units 2nd sem (approved)']             = approved
        state_dict['Curricular units 2nd sem (evaluations)']          = evals
        state_dict['Curricular units 2nd sem (without evaluations)']  = no_eval

        # ── 4. Update stress indicator ──
        stress_delta = fx.get('stress_delta', 0)
        if stress_delta != 0:
            cur = state_dict.get('Stress_indicator', 0)
            state_dict['Stress_indicator'] = max(0, min(10, cur + stress_delta))

        # ── 5. Update financial features ──
        tuition_delta = fx.get('tuition_paid_delta', 0)
        if tuition_delta != 0:
            cur = state_dict.get('Tuition fees up to date', 0)
            state_dict['Tuition fees up to date'] = max(0, min(1, cur + tuition_delta))
            if state_dict['Tuition fees up to date'] >= 1.0:
                state_dict['Debtor'] = 0

        # ── 6. Recompute derived features ──
        enrolled_s2 = max(state_dict.get('Curricular units 2nd sem (enrolled)', 1), 1)
        state_dict['Approval_rate_sem2'] = max(0, min(1, approved / enrolled_s2))
        state_dict['Failure_rate_sem2']  = 1 - state_dict['Approval_rate_sem2']
        state_dict['Grade_change']       = grade - self.sem1_grade
        state_dict['Approved_change']    = approved - self.sem1_approved

        # ── 7. Predict risk via ML model ──
        risk = self.predict_risk(state_dict)

        return state_dict, risk, grade, approved

    def predict_risk(self, state_dict):
        """Evaluate dropout risk: P(dropout | state)."""
        X = pd.DataFrame([state_dict])[self.model_features].astype(float)
        return float(self.risk_model.predict_proba(X)[0][1])
