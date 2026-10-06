"""
=============================================================================
 FILE: services/rl_environment.py
=============================================================================
 Gym-style RL environment that wraps the Digital Twin.

 Implements:
   - State representation (academic + financial + engagement + stress + history)
   - Discrete action space (interventions)
   - Multi-objective reward (risk reduction + retention - cost - instability)
   - Transition dynamics via unified TransitionModel

 Architecture (closed-loop MDP):
     State Sₜ → TransitionModel(Sₜ, Aₜ) → Sₜ₊₁ → Risk Model → Rₜ₊₁
                                                         │
                                                         ▼
                                                    Reward Rₜ
                                                         │
                                                         ▼
                                                    FQI Agent → Aₜ₊₁
                                                         │
                                                         └──► repeat

 Used for:
   - Training offline RL agents (Fitted Q-Iteration)
   - Evaluating policies by rolling out trajectories
=============================================================================
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from modules.transition_model import TransitionModel, DEFAULT_EFFECTS


# State feature list used by the RL agent (subset of full twin features,
# chosen for compactness and predictive power)
STATE_FEATURES = [
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (approved)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (without evaluations)',
    'Approval_rate_sem1',
    'Approval_rate_sem2',
    'Failure_rate_sem1',
    'Failure_rate_sem2',
    'Stress_indicator',
    'Grade_change',
    'Approved_change',
    'Debtor',
    'Tuition fees up to date',
    'Scholarship holder',
    'Age at enrollment',
    'Gender',
]

# Action space — start with the same 3 used by the current system.
# Additional actions (intensive, peer_support, etc.) can be added incrementally.
ACTIONS = ['none', 'mentoring', 'financial_aid']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Normalized cost per action (for reward computation)
ACTION_COST = {
    'none':           0.00,
    'mentoring':      0.30,
    'financial_aid':  0.50,
}


class StudentEnvironment:
    """
    Gym-compatible environment.

    reset() -> state vector (numpy array)
    step(action_idx) -> (next_state, reward, done, info)

    The environment is single-agent, single-student. For batch training,
    instantiate multiple environments in parallel.

    Internally delegates state transitions to the unified TransitionModel,
    ensuring identical dynamics with the Digital Twin simulation.
    """

    def __init__(self, initial_raw, model_s2, features_s2,
                 learned_effects=None, horizon=24,
                 reward_weights=None, max_shift=None,
                 noise_std=0.0, seed=None):
        """
        Args:
            initial_raw: dict of raw feature values for the student (from twin.raw_data)
            model_s2: trained XGBoost Stage-2 classifier
            features_s2: feature list expected by model_s2
            learned_effects: dict of per-week deltas per action (from pipeline)
            horizon: number of weeks to simulate
            reward_weights: dict with keys w_risk, w_retention, w_cost, w_instability
            max_shift: cumulative caps for each feature (same as digital twin)
        """
        self.initial_raw = deepcopy(initial_raw)
        self.model_s2 = model_s2
        self.features_s2 = features_s2
        self.effects = learned_effects if learned_effects else DEFAULT_EFFECTS
        self.horizon = horizon

        self.reward_weights = reward_weights or {
            'w_risk': 10.0,
            'w_retention': 2.0,
            'w_cost': 1.0,
            'w_instability': 0.5,
        }
        self.max_shift = max_shift or {
            'grade': 3.0, 'approved': 3.0, 'eval': 3.0,
        }

        # Stochasticity: noise on transitions and observations
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Action history — counts per action taken so far (for state)
        self.history = None
        self.transition = None
        self._reset_state()

    def _reset_state(self):
        """Initialize simulation state and create fresh TransitionModel."""
        self.t = 0
        self.state = deepcopy(self.initial_raw)
        self.history = {a: 0 for a in ACTIONS}

        # Create unified transition model (shared dynamics with Digital Twin)
        self.transition = TransitionModel(
            self.initial_raw, self.model_s2, self.features_s2,
            self.effects, self.max_shift
        )
        self.current_risk = self.transition.predict_risk(self.state)

    def reset(self):
        self._reset_state()
        return self._get_state_vector()

    def step(self, action_idx):
        """
        Apply action for one week, return (next_state, reward, done, info).

        Delegates the core transition Sₜ + Aₜ → Sₜ₊₁ to TransitionModel.
        """
        action = ACTIONS[action_idx] if isinstance(action_idx, (int, np.integer)) else action_idx
        risk_before = self.current_risk

        # ── Core transition: Sₜ + Aₜ → Sₜ₊₁  (unified with Digital Twin) ──
        self.state, risk_after, grade, approved = self.transition.step(
            self.state, action, noise_std=self.noise_std, rng=self.rng
        )

        self.current_risk = risk_after

        # Update history & timestep
        self.history[action] += 1
        self.t += 1

        # Compute reward
        reward = self._compute_reward(risk_before, risk_after, action)
        done = self.t >= self.horizon

        info = {
            'risk': risk_after,
            'action': action,
            'week': self.t,
            'grade': grade,
            'approved': approved,
        }
        return self._get_state_vector(), reward, done, info

    def _compute_reward(self, risk_before, risk_after, action):
        w = self.reward_weights
        delta_risk = risk_before - risk_after  # positive = good
        retention = 1.0 if risk_after < 0.5 else 0.0
        cost = ACTION_COST.get(action, 0)
        instability = abs(delta_risk) if delta_risk < -0.15 else 0  # only penalize risk spikes, not improvements
        return (w['w_risk'] * delta_risk
                + w['w_retention'] * retention
                - w['w_cost'] * cost
                - w['w_instability'] * instability)

    def _get_state_vector(self):
        """Return compact feature vector for RL agent."""
        feats = [float(self.state.get(f, 0)) for f in STATE_FEATURES]
        feats.append(self.current_risk)  # model's current risk estimate
        feats.append(float(self.t) / self.horizon)  # normalized time
        # Action history (fraction of weeks each action was used)
        total = max(self.t, 1)
        for a in ACTIONS:
            feats.append(self.history[a] / total)
        return np.array(feats, dtype=np.float32)

    @property
    def state_dim(self):
        return len(STATE_FEATURES) + 2 + len(ACTIONS)

    @property
    def n_actions(self):
        return len(ACTIONS)


def rollout(env, policy, seed=None):
    """
    Run one full episode under the given policy.
    Returns list of (state, action_idx, reward, next_state, done, info) tuples.
    policy: callable(state_vector) -> action_idx
    """
    if seed is not None:
        np.random.seed(seed)
    transitions = []
    state = env.reset()
    done = False
    while not done:
        action_idx = policy(state)
        next_state, reward, done, info = env.step(action_idx)
        transitions.append((state, action_idx, reward, next_state, done, info))
        state = next_state
    return transitions
