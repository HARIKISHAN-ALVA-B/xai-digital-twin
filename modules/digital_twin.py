"""
=============================================================================
 FILE: modules/digital_twin.py
=============================================================================
 PURPOSE:
   Implements the Digital Twin framework where each student is modeled as
   a dynamic computational object that:
     1. Holds a "Stage 1" state (after Semester 1 data is known)
     2. Can be "updated" to "Stage 2" when Semester 2 data arrives
     3. Tracks risk score, stress indicator, and prediction history
     4. Supports what-if simulation of counseling interventions

 DESIGN:
   - StudentDigitalTwin : represents one student
   - DigitalTwinManager : manages the collection of all twins
   - Intervention scenarios: none, immediate_counseling, delayed_counseling,
                             intensive_support, peer_mentoring
=============================================================================
"""

import numpy as np
import pandas as pd
import pickle, json
from copy import deepcopy
from datetime import datetime
from modules.transition_model import TransitionModel, DEFAULT_EFFECTS


class StudentDigitalTwin:
    """
    A Digital Twin for a single student.
    
    The twin is initialized with Stage 1 data (demographics + Sem 1)
    and can be updated with Stage 2 data (Sem 2 results) to show
    how the student's risk evolves over time.
    """

    def __init__(self, student_id, row_data, model_s1, features_s1):
        self.student_id = student_id
        self.created_at = datetime.now().isoformat()

        # Store all raw data for this student
        self.raw_data = row_data.to_dict() if hasattr(row_data, 'to_dict') else dict(row_data)

        # ── Stage 1 prediction (Sem 1 only) ──
        feat_vals = pd.DataFrame([self.raw_data])[features_s1].astype(float)
        self.stage1_risk_prob = float(model_s1.predict_proba(feat_vals)[0][1])
        self.stage1_risk_label = int(self.stage1_risk_prob > 0.5)

        # ── Current state ──
        self.current_stage = 1
        self.states = [{
            'stage': 1,
            'risk_prob': round(self.stage1_risk_prob, 4),
            'risk_label': self.stage1_risk_label,
            'stress': round(self.raw_data.get('Stress_indicator', 0), 2),
            'sem1_grade': round(self.raw_data.get('Curricular units 1st sem (grade)', 0), 2),
            'sem1_approved': int(self.raw_data.get('Curricular units 1st sem (approved)', 0)),
            'sem2_grade': None,
            'sem2_approved': None,
        }]

        # Intervention history
        self.interventions = []

    def update_to_stage2(self, model_s2, features_s2, sem2_overrides=None):
        """
        Update the twin to Stage 2 using 2nd-semester data.
        
        If sem2_overrides is provided (dict), those values replace the original
        Sem 2 data — this is how counselors can manually input updated marks.
        """
        data = deepcopy(self.raw_data)

        # Apply overrides if provided (simulating a manual data entry)
        if sem2_overrides:
            for key, val in sem2_overrides.items():
                data[key] = val
            # Recompute derived features
            enrolled_s2 = max(data.get('Curricular units 2nd sem (enrolled)', 1), 1)
            data['Approval_rate_sem2'] = data.get('Curricular units 2nd sem (approved)', 0) / enrolled_s2
            data['Failure_rate_sem2']  = 1 - data['Approval_rate_sem2']
            data['Grade_change'] = (data.get('Curricular units 2nd sem (grade)', 0)
                                    - data.get('Curricular units 1st sem (grade)', 0))
            data['Approved_change'] = (data.get('Curricular units 2nd sem (approved)', 0)
                                       - data.get('Curricular units 1st sem (approved)', 0))

        # Stage 2 prediction
        feat_vals = pd.DataFrame([data])[features_s2].astype(float)
        risk_prob = float(model_s2.predict_proba(feat_vals)[0][1])
        risk_label = int(risk_prob > 0.5)

        # Recompute stress for Stage 2 (now includes Sem 2 failure)
        evals_s2 = max(data.get('Curricular units 2nd sem (evaluations)', 1), 1)
        failure_s2 = (evals_s2 - data.get('Curricular units 2nd sem (approved)', 0)) / evals_s2
        stress_s2 = (
            0.20 * (data.get('Debtor', 0) * 0.5 + (1 - data.get('Tuition fees up to date', 1)) * 0.5) +
            0.25 * max(0, min(1, failure_s2)) +
            0.25 * max(0, min(1, data.get('Failure_rate_sem1', 0))) +
            0.15 * max(0, min(1, data.get('Curricular units 2nd sem (without evaluations)', 0) /
                               max(data.get('Curricular units 2nd sem (enrolled)', 1), 1))) +
            0.15 * max(0, min(1, (10 - data.get('Curricular units 2nd sem (grade)', 0)) / 10))
        ) * 10

        self.current_stage = 2
        self.raw_data = data
        self.states.append({
            'stage': 2,
            'risk_prob': round(risk_prob, 4),
            'risk_label': risk_label,
            'stress': round(stress_s2, 2),
            'sem1_grade': round(data.get('Curricular units 1st sem (grade)', 0), 2),
            'sem1_approved': int(data.get('Curricular units 1st sem (approved)', 0)),
            'sem2_grade': round(data.get('Curricular units 2nd sem (grade)', 0), 2),
            'sem2_approved': int(data.get('Curricular units 2nd sem (approved)', 0)),
        })

        return self.states[-1]

    def simulate_intervention(self, intervention_type, model_s2, features_s2,
                              learned_effects=None):
        """
        Simulate how Sem 2 results might change under an intervention.
        Returns the projected Stage 2 state.

        Uses data-driven effects when available (from intervention_effects.json),
        falling back to heuristic defaults otherwise.
        """
        # Map legacy intervention names to data-driven effect keys
        _effect_key_map = {
            'none': 'none',
            'immediate_counseling': 'mentoring',
            'delayed_counseling': 'mentoring',
            'intensive_support': 'mentoring',
            'peer_mentoring': 'mentoring',
            'mentoring': 'mentoring',
            'financial_aid': 'financial_aid',
        }
        # Intensity scaling for legacy names
        _intensity = {
            'delayed_counseling': 0.5,
            'peer_mentoring': 0.7,
            'intensive_support': 1.5,
        }

        SIMULATION_WEEKS = 6
        fx_key = _effect_key_map.get(intervention_type, 'none')
        intensity = _intensity.get(intervention_type, 1.0)

        if learned_effects and fx_key in learned_effects:
            fx = learned_effects[fx_key]
            effect = {
                'approved_boost': round(fx.get('approved_delta', 0) * SIMULATION_WEEKS * intensity, 1),
                'grade_boost': round(fx.get('grade_delta', 0) * SIMULATION_WEEKS * intensity, 1),
                'eval_boost': round(fx.get('eval_delta', 0) * SIMULATION_WEEKS * intensity, 1),
            }
        else:
            _fallback = {
                'none': {'approved_boost': 0, 'grade_boost': 0, 'eval_boost': 0},
                'immediate_counseling': {'approved_boost': 2, 'grade_boost': 1.5, 'eval_boost': 1},
                'delayed_counseling': {'approved_boost': 1, 'grade_boost': 0.8, 'eval_boost': 0},
                'intensive_support': {'approved_boost': 3, 'grade_boost': 2.5, 'eval_boost': 2},
                'peer_mentoring': {'approved_boost': 1.5, 'grade_boost': 1.2, 'eval_boost': 1},
            }
            effect = _fallback.get(intervention_type, _fallback['none'])

        enrolled = max(self.raw_data.get('Curricular units 2nd sem (enrolled)', 6), 1)

        overrides = {
            'Curricular units 2nd sem (approved)': min(
                int(self.raw_data.get('Curricular units 2nd sem (approved)', 0) + effect['approved_boost']),
                enrolled
            ),
            'Curricular units 2nd sem (grade)': min(
                self.raw_data.get('Curricular units 2nd sem (grade)', 0) + effect['grade_boost'],
                20.0
            ),
            'Curricular units 2nd sem (evaluations)': min(
                int(self.raw_data.get('Curricular units 2nd sem (evaluations)', 0) + effect['eval_boost']),
                enrolled * 2
            ),
            'Curricular units 2nd sem (without evaluations)': max(
                int(self.raw_data.get('Curricular units 2nd sem (without evaluations)', 0) - effect['eval_boost']),
                0
            ),
        }

        # Temporarily compute the projected state
        data = deepcopy(self.raw_data)
        for k, v in overrides.items():
            data[k] = v

        enrolled_s2 = max(data.get('Curricular units 2nd sem (enrolled)', 1), 1)
        data['Approval_rate_sem2'] = data['Curricular units 2nd sem (approved)'] / enrolled_s2
        data['Failure_rate_sem2']  = 1 - data['Approval_rate_sem2']
        data['Grade_change'] = data['Curricular units 2nd sem (grade)'] - data.get('Curricular units 1st sem (grade)', 0)
        data['Approved_change'] = data['Curricular units 2nd sem (approved)'] - data.get('Curricular units 1st sem (approved)', 0)

        feat_vals = pd.DataFrame([data])[features_s2].astype(float)
        risk_prob = float(model_s2.predict_proba(feat_vals)[0][1])

        return {
            'intervention': intervention_type,
            'projected_risk_prob': round(risk_prob, 4),
            'projected_risk_label': int(risk_prob > 0.5),
            'projected_sem2_approved': data['Curricular units 2nd sem (approved)'],
            'projected_sem2_grade': round(data['Curricular units 2nd sem (grade)'], 2),
        }

    def compare_all_interventions(self, model_s2, features_s2, learned_effects=None):
        """Compare all intervention scenarios side by side."""
        scenarios = ['none', 'immediate_counseling', 'delayed_counseling',
                     'intensive_support', 'peer_mentoring']
        return {s: self.simulate_intervention(s, model_s2, features_s2, learned_effects=learned_effects) for s in scenarios}

    # ==================================================================
    #  Time-Based Simulation (week-by-week Digital Twin evolution)
    # ==================================================================

    def simulate_weeks(self, weeks, intervention, model_s2, features_s2,
                        learned_effects=None, sem2_predictor=None, sem1_features=None):
        """
        Simulate week-by-week progression of the student's digital twin.

        Delegates state transitions to the shared TransitionModel, ensuring
        identical dynamics with the RL environment.

        Architecture:
            for each week:
                Sₜ + Aₜ → TransitionModel.step() → Sₜ₊₁
                Sₜ₊₁ → Risk Model → Riskₜ₊₁ (with per-week clamping)

        Args:
            weeks: number of weeks to simulate (1-12)
            intervention: "none", "mentoring", or "financial_aid"
            model_s2: trained Stage 2 XGBoost classifier
            features_s2: list of Stage 2 feature names
            learned_effects: dict of data-driven per-week deltas (from pipeline)
            sem2_predictor: trained Sem1->Sem2 regression model (optional)
            sem1_features: feature list for sem2_predictor (optional)

        Returns:
            list of weekly state dicts
        """
        # Resolve effects: learned > default
        effects = learned_effects if learned_effects else DEFAULT_EFFECTS

        data = deepcopy(self.raw_data)

        # If Sem2 predictor available, blend predicted baseline for interventions
        if sem2_predictor is not None and sem1_features is not None and intervention != 'none':
            try:
                s1_vals = pd.DataFrame([data])[sem1_features].astype(float)
                pred = [float(v) for v in sem2_predictor.predict(s1_vals)[0]]
                actual_grade = data.get('Curricular units 2nd sem (grade)', 0)
                if actual_grade > 0:
                    data['Curricular units 2nd sem (grade)'] = 0.7 * pred[0] + 0.3 * actual_grade
                    data['Curricular units 2nd sem (approved)'] = 0.7 * pred[1] + 0.3 * data.get('Curricular units 2nd sem (approved)', 0)
                    data['Curricular units 2nd sem (evaluations)'] = 0.7 * pred[2] + 0.3 * data.get('Curricular units 2nd sem (evaluations)', 0)
                    data['Curricular units 2nd sem (without evaluations)'] = 0.7 * pred[3] + 0.3 * data.get('Curricular units 2nd sem (without evaluations)', 0)
            except Exception:
                pass

        # ── Create unified transition model (shared with RL environment) ──
        transition = TransitionModel(
            data, model_s2, features_s2, effects,
            max_shift={'grade': 2.0, 'approved': 2.0, 'eval': 2.0}
        )

        # Current risk for clamping
        MAX_WEEK_CHANGE = 0.20  # max +-20% per-step risk swing
        prev_risk = transition.predict_risk(data)
        timeline = []

        for w in range(1, weeks + 1):
            # ── Core transition: Sₜ + Aₜ → Sₜ₊₁  (unified dynamics) ──
            data, risk_raw, grade, approved = transition.step(data, intervention)

            # Clamp: no more than +-20% swing from previous week (smooth trajectory)
            risk_prob = max(prev_risk - MAX_WEEK_CHANGE, min(prev_risk + MAX_WEEK_CHANGE, risk_raw))

            prev_risk = risk_prob
            cur_stress = data.get('Stress_indicator', 0)

            timeline.append({
                'week': w,
                'risk': round(risk_prob, 4),
                'risk_label': int(risk_prob > 0.5),
                'stress': round(cur_stress, 2),
                'grade': round(grade, 2),
                'approved': round(approved, 1),
            })

        return timeline

    def get_best_intervention(self, model_s2, features_s2, weeks=6,
                              learned_effects=None, sem2_predictor=None, sem1_features=None):
        """
        Simulate all intervention types and return the one with lowest final risk.
        Uses data-driven effects when available.
        """
        candidates = ['none', 'mentoring', 'financial_aid']
        results = {}
        current_risk = self.to_dict()['risk_prob']

        for intervention in candidates:
            timeline = self.simulate_weeks(
                weeks, intervention, model_s2, features_s2,
                learned_effects=learned_effects,
                sem2_predictor=sem2_predictor,
                sem1_features=sem1_features,
            )
            final = timeline[-1]
            results[intervention] = {
                'final_risk': final['risk'],
                'final_stress': final['stress'],
                'timeline': timeline,
            }

        none_final = results['none']['final_risk']
        for intervention in candidates:
            reduction = none_final - results[intervention]['final_risk']
            results[intervention]['risk_reduction_vs_none'] = round(reduction, 4)
            results[intervention]['risk_reduction_vs_current'] = round(current_risk - results[intervention]['final_risk'], 4)
            # Mark ineffective if intervention risk >= no-intervention risk
            results[intervention]['effective'] = (intervention == 'none') or (reduction > 0.001)

        # Best = lowest final risk among effective interventions only
        effective = [k for k in candidates if results[k]['effective']]
        best = min(effective, key=lambda k: results[k]['final_risk']) if effective else 'none'

        return {
            'best_intervention': best,
            'best_final_risk': results[best]['final_risk'],
            'best_risk_reduction': results[best]['risk_reduction_vs_none'],
            'current_risk': round(current_risk, 4),
            'none_final_risk': round(none_final, 4),
            'weeks_simulated': weeks,
            'data_driven': learned_effects is not None,
            'comparison': results,
        }

    def get_progression(self):
        """Return the risk progression across stages for visualization."""
        return {
            'student_id': self.student_id,
            'stages': self.states,
            'current_stage': self.current_stage,
            'actual_target': self.raw_data.get('Target', 'Unknown'),
        }

    def to_dict(self):
        """Full serialization for API responses."""
        s = self.states[-1]  # latest state
        return {
            'student_id': self.student_id,
            'current_stage': self.current_stage,
            'risk_prob': s['risk_prob'],
            'risk_label': s['risk_label'],
            'risk_category': 'Critical' if s['risk_prob'] > 0.75 else
                             'High' if s['risk_prob'] > 0.5 else
                             'Moderate' if s['risk_prob'] > 0.3 else 'Low',
            'stress': s['stress'],
            'stress_level': 'Critical' if s['stress'] > 7.5 else
                            'High' if s['stress'] > 5 else
                            'Moderate' if s['stress'] > 2.5 else 'Low',
            'sem1_grade': s['sem1_grade'],
            'sem1_approved': s['sem1_approved'],
            'sem2_grade': s.get('sem2_grade'),
            'sem2_approved': s.get('sem2_approved'),
            'actual_target': self.raw_data.get('Target', 'Unknown'),
            'states': self.states,
            'raw_data': self.raw_data,
        }


class DigitalTwinManager:
    """Manages a collection of student digital twins."""

    def __init__(self):
        self.twins = {}

    def build_from_dataframe(self, df, model_s1, model_s2, features_s1, features_s2):
        """Build twins for all students using BATCH prediction for speed."""
        # Batch predict Stage 1 and Stage 2 risk probabilities
        X_s1 = df[features_s1].astype(float)
        X_s2 = df[features_s2].astype(float)
        probs_s1 = model_s1.predict_proba(X_s1)[:, 1]
        probs_s2 = model_s2.predict_proba(X_s2)[:, 1]

        for i, (idx, row) in enumerate(df.iterrows()):
            sid = f"STU{idx:04d}"
            twin = StudentDigitalTwin.__new__(StudentDigitalTwin)
            twin.student_id = sid
            twin.created_at = datetime.now().isoformat()
            twin.raw_data = row.to_dict()
            twin.interventions = []

            p1 = float(probs_s1[i])
            stress1 = row.get('Stress_indicator', 0)
            twin.stage1_risk_prob = p1
            twin.stage1_risk_label = int(p1 > 0.5)

            p2 = float(probs_s2[i])
            # Compute stage 2 stress
            evals_s2 = max(row.get('Curricular units 2nd sem (evaluations)', 1), 1)
            fail_s2 = (evals_s2 - row.get('Curricular units 2nd sem (approved)', 0)) / evals_s2
            enr2 = max(row.get('Curricular units 2nd sem (enrolled)', 1), 1)
            stress2 = (
                0.20 * (row.get('Debtor', 0) * 0.5 + (1 - row.get('Tuition fees up to date', 1)) * 0.5) +
                0.25 * max(0, min(1, fail_s2)) +
                0.25 * max(0, min(1, row.get('Failure_rate_sem1', 0))) +
                0.15 * max(0, min(1, row.get('Curricular units 2nd sem (without evaluations)', 0) / enr2)) +
                0.15 * max(0, min(1, (10 - row.get('Curricular units 2nd sem (grade)', 0)) / 10))
            ) * 10

            twin.states = [
                {'stage': 1, 'risk_prob': round(p1, 4), 'risk_label': int(p1 > 0.5),
                 'stress': round(stress1, 2),
                 'sem1_grade': round(row.get('Curricular units 1st sem (grade)', 0), 2),
                 'sem1_approved': int(row.get('Curricular units 1st sem (approved)', 0)),
                 'sem2_grade': None, 'sem2_approved': None},
                {'stage': 2, 'risk_prob': round(p2, 4), 'risk_label': int(p2 > 0.5),
                 'stress': round(stress2, 2),
                 'sem1_grade': round(row.get('Curricular units 1st sem (grade)', 0), 2),
                 'sem1_approved': int(row.get('Curricular units 1st sem (approved)', 0)),
                 'sem2_grade': round(row.get('Curricular units 2nd sem (grade)', 0), 2),
                 'sem2_approved': int(row.get('Curricular units 2nd sem (approved)', 0))},
            ]
            twin.current_stage = 2
            self.twins[sid] = twin

        return len(self.twins)

    def add_student(self, student_id, data_dict, model_s1, model_s2, features_s1, features_s2):
        """Add a new student (CRUD — Create)."""
        row = pd.Series(data_dict)
        twin = StudentDigitalTwin(student_id, row, model_s1, features_s1)
        # If Sem 2 data exists, update
        if data_dict.get('Curricular units 2nd sem (enrolled)', 0) > 0:
            twin.update_to_stage2(model_s2, features_s2)
        self.twins[student_id] = twin
        return twin

    def get_twin(self, student_id):
        return self.twins.get(student_id)

    def delete_twin(self, student_id):
        """CRUD — Delete."""
        if student_id in self.twins:
            del self.twins[student_id]
            return True
        return False

    def get_all_summaries(self):
        """Get summary of all twins for the dashboard table."""
        return [t.to_dict() for t in self.twins.values()]

    def get_at_risk(self):
        return sorted(
            [t.to_dict() for t in self.twins.values() if t.to_dict()['risk_label'] == 1],
            key=lambda x: x['risk_prob'], reverse=True
        )

    def get_dashboard_stats(self):
        all_t = self.get_all_summaries()
        total = len(all_t)
        if total == 0:
            return {'total': 0, 'at_risk': 0, 'safe': 0}
        at_risk = sum(1 for t in all_t if t['risk_label'] == 1)
        risks = [t['risk_prob'] for t in all_t]
        stresses = [t['stress'] for t in all_t]
        return {
            'total': total,
            'at_risk': at_risk,
            'safe': total - at_risk,
            'at_risk_pct': round(at_risk / total * 100, 1),
            'avg_risk': round(np.mean(risks), 4),
            'avg_stress': round(np.mean(stresses), 2),
            'risk_dist': {
                'Low': sum(1 for t in all_t if t['risk_category'] == 'Low'),
                'Moderate': sum(1 for t in all_t if t['risk_category'] == 'Moderate'),
                'High': sum(1 for t in all_t if t['risk_category'] == 'High'),
                'Critical': sum(1 for t in all_t if t['risk_category'] == 'Critical'),
            },
            'stress_dist': {
                'Low': sum(1 for t in all_t if t['stress_level'] == 'Low'),
                'Moderate': sum(1 for t in all_t if t['stress_level'] == 'Moderate'),
                'High': sum(1 for t in all_t if t['stress_level'] == 'High'),
                'Critical': sum(1 for t in all_t if t['stress_level'] == 'Critical'),
            }
        }
