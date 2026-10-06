"""
=============================================================================
 FILE: services/rl_policies.py
=============================================================================
 Policies and the Fitted Q-Iteration (FQI) agent.

 Baselines:
   - NoActionPolicy (B1)
   - GPAThresholdPolicy (B2) -- rule-based
   - GreedySHAPPolicy (B3) -- pick action based on highest risk category
   - RandomPolicy (B0)

 Learned:
   - FQIAgent -- offline RL via Fitted Q-Iteration with XGBoost regressors.
     Trains one Q_a(s) regressor per action.
     policy(s) = argmax_a Q_a(s)

 Why FQI + XGBoost:
   - Offline (no exploration needed -- works on replay buffer)
   - No deep-learning dependency
   - Matches existing XGBoost stack
   - Interpretable via SHAP
=============================================================================
"""

import numpy as np
import pickle
from xgboost import XGBRegressor
from .rl_environment import ACTIONS, ACTION_TO_IDX, STATE_FEATURES


# ============================================================
#  Baseline Policies
# ============================================================

def no_action_policy(state_vec):
    return ACTION_TO_IDX['none']


def random_policy(state_vec):
    return np.random.randint(len(ACTIONS))


def gpa_threshold_policy(state_vec):
    """B2: if Sem2 grade < 10 -> mentoring, elif tuition unpaid -> financial_aid, else none."""
    sem2_grade_idx = STATE_FEATURES.index('Curricular units 2nd sem (grade)')
    tuition_idx = STATE_FEATURES.index('Tuition fees up to date')
    sem2_grade = state_vec[sem2_grade_idx]
    tuition_paid = state_vec[tuition_idx]
    if sem2_grade < 10:
        return ACTION_TO_IDX['mentoring']
    if tuition_paid < 0.5:
        return ACTION_TO_IDX['financial_aid']
    return ACTION_TO_IDX['none']


def financial_targeted_policy(state_vec):
    """B4: prefer financial_aid for financially stressed students (covers
    underrepresented action in training data — improves Q-function calibration)."""
    tuition_idx = STATE_FEATURES.index('Tuition fees up to date')
    debtor_idx = STATE_FEATURES.index('Debtor')
    stress_idx = STATE_FEATURES.index('Stress_indicator')
    tuition_paid = state_vec[tuition_idx]
    is_debtor = state_vec[debtor_idx]
    stress = state_vec[stress_idx]
    if tuition_paid < 0.5 or is_debtor > 0.5 or stress > 5.0:
        return ACTION_TO_IDX['financial_aid']
    if stress > 3.0:
        return ACTION_TO_IDX['mentoring']
    return ACTION_TO_IDX['none']


def greedy_shap_policy_factory(shap_categorizer):
    """
    B3: Greedy SHAP policy.
    Given a function that returns the top category from SHAP, map to action.
    """
    cat_to_action = {
        'academic':   'mentoring',
        'stress':     'mentoring',
        'engagement': 'mentoring',
        'financial':  'financial_aid',
    }
    def policy(state_vec):
        cat = shap_categorizer(state_vec)
        return ACTION_TO_IDX[cat_to_action.get(cat, 'none')]
    return policy


# ============================================================
#  Fitted Q-Iteration Agent
# ============================================================

class FQIAgent:
    """
    Fitted Q-Iteration with XGBoost regressors (one per action).

    Q_a(s) approximated by an XGBRegressor.
    Training loop (Bellman iteration):
      target(s, a) = r + gamma * max_{a'} Q_{k-1}(s', a')
      Fit Q_k_a(s) = target for each action a

    Policy:
      pi(s) = argmax_a Q_a(s)
    """

    def __init__(self, n_actions, gamma=0.95, n_iterations=20,
                 xgb_params=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.xgb_params = xgb_params or {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42,
        }
        # One regressor per action (only populated after training)
        self.q_models = [None] * n_actions
        self.trained = False

    def _predict_q(self, states):
        """Predict Q-values for all actions. Returns (n_states, n_actions) array."""
        n = states.shape[0]
        Q = np.zeros((n, self.n_actions))
        for a in range(self.n_actions):
            if self.q_models[a] is not None:
                Q[:, a] = self.q_models[a].predict(states)
        return Q

    def fit(self, transitions, verbose=True):
        """
        transitions: list of (s, a, r, s', done) tuples (flat, from many episodes)
        """
        states      = np.stack([t[0] for t in transitions])
        actions     = np.array([t[1] for t in transitions])
        rewards     = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.stack([t[3] for t in transitions])
        dones       = np.array([t[4] for t in transitions], dtype=np.float32)

        n = len(transitions)
        if verbose:
            print(f"[FQI] Training on {n} transitions, {self.n_iterations} iterations, gamma={self.gamma}")

        # Iteration 0: initialize Q as reward only
        for a in range(self.n_actions):
            mask = actions == a
            if mask.sum() > 0:
                self.q_models[a] = XGBRegressor(**self.xgb_params)
                self.q_models[a].fit(states[mask], rewards[mask])

        # Iterations 1..K: Bellman updates
        for k in range(1, self.n_iterations + 1):
            # Compute max_a' Q(s', a') for all next_states
            Q_next = self._predict_q(next_states)
            max_Q_next = Q_next.max(axis=1)
            # Terminal states contribute only reward
            targets = rewards + self.gamma * (1 - dones) * max_Q_next

            # Refit Q_a(s) for each action using transitions where action == a
            for a in range(self.n_actions):
                mask = actions == a
                if mask.sum() > 0:
                    self.q_models[a] = XGBRegressor(**self.xgb_params)
                    self.q_models[a].fit(states[mask], targets[mask])

            if verbose and (k == 1 or k % 5 == 0 or k == self.n_iterations):
                mean_q = max_Q_next.mean()
                print(f"[FQI] Iter {k:2d}/{self.n_iterations}  mean_max_Q={mean_q:.3f}  "
                      f"target_range=[{targets.min():.2f}, {targets.max():.2f}]")

        self.trained = True

    def policy(self, state_vec):
        """Greedy policy: argmax_a Q_a(s)."""
        if not self.trained:
            return 0  # default to no-action
        s = state_vec.reshape(1, -1)
        q_vals = self._predict_q(s)[0]
        return int(np.argmax(q_vals))

    def q_values(self, state_vec):
        s = state_vec.reshape(1, -1)
        return self._predict_q(s)[0]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'n_actions': self.n_actions,
                'gamma': self.gamma,
                'n_iterations': self.n_iterations,
                'xgb_params': self.xgb_params,
                'q_models': self.q_models,
                'trained': self.trained,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        agent = cls(n_actions=d['n_actions'], gamma=d['gamma'],
                    n_iterations=d['n_iterations'], xgb_params=d['xgb_params'])
        agent.q_models = d['q_models']
        agent.trained = d['trained']
        return agent
