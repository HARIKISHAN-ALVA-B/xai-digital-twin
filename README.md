# Explainable AI Digital Twin Framework for Early Academic Risk Prediction and Prescriptive Intervention

## Team CS44 — B.E. CSE Final Year Project 

---

## Problem Statement

Every year, many students drop out of higher education. By the time institutions notice, it is often too late for effective intervention. This system:

1. **Predicts** dropout risk early (after Semester 1)
2. **Explains** why a student is at risk (SHAP)
3. **Simulates** how risk evolves under different interventions (Digital Twin)
4. **Recommends** category-based counseling strategies (SHAP aggregation)
5. **Prescribes** optimal week-by-week intervention sequences (Reinforcement Learning)
6. **Justifies** each recommendation with explanations + counterfactuals (XRL)
7. **Audits** the recommendations for causal validity and fairness

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Dropout Risk Prediction** | XGBoost classifier (Stage 1: 89.8% AUC, Stage 2: 93.0% AUC) |
| **SHAP Explainability** | Global + local feature importance via TreeExplainer |
| **Digital Twin Simulation** | Each student evolves week-by-week under learned intervention effects |
| **Intervention Analysis** | Compare multiple support strategies with risk trajectories |
| **Best Intervention Selection** | Identifies the intervention with lowest projected risk |
| **SHAP-Based Recommendations** | Category-aggregated counseling recommendations |
| **RL Optimal Plan (NEW)** | Fitted Q-Iteration policy for multi-week intervention sequences |
| **Explainable RL (NEW)** | SHAP on Q-values + counterfactual + natural language |
| **Causal Inference (NEW)** | IPW + Doubly Robust ATE estimation with E-value sensitivity |
| **Fairness Audit (NEW)** | Per-protected-group metrics and gap analysis |
| **Rigorous Evaluation (NEW)** | Paired t-test, Wilcoxon, bootstrap CIs, Cohen's d, ablation, robustness |
| **Interactive Dashboard** | Role-based UI: counselor, student, researcher, admin |
| **Role-Based Access Control** | FastAPI middleware with session authentication |
| **Data Persistence** | CRUD operations survive server restarts |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train classification models (once)
python pipelines/ml_pipeline.py

# 3. Train RL policy (once)
python pipelines/train_rl.py

# 4. (Optional) Run rigorous evaluation for research
python pipelines/evaluate_rl_rigorous.py

# 5. Start dashboard
python main.py
```

Open **http://localhost:5000** in your browser.

Demo accounts (password = username): `admin`, `counselor`, `researcher`, `student1`, `student2`

---

## System Architecture

```
                  Browser UI (Role-Based)
                         |
                    FastAPI Server
                         |
     +-------------------+------------------+
     |                   |                  |
Student Twin      SHAP Explainer       RL Policy (FQI)
     |                   |                  |
     |            Recommendations    XRL Explainer
     |                   |                  |
     +--- Digital Twin Environment ---+
                         |
            Trained Models (models/)
                         |
     Classification + SHAP + Intervention + RL + Causal
```

### Module Responsibilities

| Folder | Purpose |
|--------|---------|
| `main.py` | FastAPI server with 20+ REST endpoints |
| `modules/digital_twin.py` | StudentDigitalTwin + DigitalTwinManager + simulate_weeks |
| `services/recommendations.py` | SHAP-based category recommendation engine |
| `services/auth.py` | Role-based access control |
| `services/rl_environment.py` | Gym-style RL environment wrapping the digital twin |
| `services/rl_policies.py` | FQI agent + baseline policies (no-action, random, GPA rule) |
| `services/rl_explainer.py` | XRL: SHAP on Q-values + counterfactual + natural language |
| `services/rl_causal.py` | IPW, Doubly Robust ATE, E-value sensitivity analysis |
| `services/rl_fairness.py` | Per-protected-group metrics + equal opportunity |
| `pipelines/ml_pipeline.py` | Data loading + feature engineering + classifier training + SHAP |
| `pipelines/train_rl.py` | Generate replay buffer + train FQI + evaluate |
| `pipelines/evaluate_rl_rigorous.py` | Paired tests + bootstrap + ablation + fairness + causal |
| `models/` | Trained artifacts (pickled classifiers, RL policy, metrics JSON) |
| `data/` | UCI dataset, processed data, CRUD persistence |
| `templates/dashboard.html` | Single-page role-aware UI |
| `static/plots/` | SHAP summary + force plots |
| `docs/RL_INTERVENTION_DESIGN.md` | Research design document |

---

## Role-Based UI

### Counselor (primary user)
- Dashboard: institution-wide analytics
- Students tab: list, search, filter, CRUD, detail view
- Student detail panel:
  - Risk assessment + SHAP explanation
  - Contextual stats (percentile, similar students)
  - Digital Twin progression
  - Timeline simulation (No Intervention / Mentoring / Financial Aid)
  - **AI-Optimized Intervention Plan (RL + XRL)** — new section with week-by-week plan, SHAP factors, counterfactual, natural language explanation
- Export CSV

### Student (end user)
- Personal dashboard only (own data)
- Profile card + academic snapshot
- Risk + positives + areas to improve + recommendations + stress + 6-week simulation
- Simple, non-technical language

### Researcher
- **Overview**: dataset, findings, two-stage design
- **Model**: metrics, ROC, confusion matrix
- **Analysis**: hyperparameters, SHAP plots, interpretation guide
- **RL Results (NEW)**: statistical tests, bootstrap CIs, ablation, robustness, fairness, causal ATE
- **About**: formal methodology document

### Admin
- System Control Panel only
- User management, dataset info, model file list, retrain trigger, reset, change log

---

## Model Performance

| Stage | Model | Accuracy | AUC-ROC | 5-Fold CV AUC |
|-------|-------|----------|---------|---------------|
| Stage 1 (Sem 1 only) | XGBoost | 86.33% | 89.82% | 89.63% ± 0.77% |
| Stage 2 (Sem 1 + 2) | XGBoost | 88.25% | 93.04% | 92.10% ± 0.77% |

## RL Policy Performance (300 students, 12-week horizon)

| Policy | Risk Reduction | Retention Rate | 95% CI |
|--------|:---:|:---:|:---:|
| B1 No-Action | -1.65% | 51.3% | [-2.79, -0.50] |
| B0 Random | +1.82% | 52.3% | [+0.85, +2.86] |
| B2 GPA Rule | +1.06% | 54.0% | [-0.52, +2.61] |
| **RL FQI** | **+5.24%** | **62.0%** | **[+3.37, +7.24]** |

### Statistical significance (paired t-test)
- RL vs No-Action: t = 7.79, **p = 1.1×10⁻¹³**, Cohen's d = 0.45
- RL vs Random: t = 4.53, **p = 8.6×10⁻⁶**, Cohen's d = 0.26
- RL vs GPA Rule: t = 4.57, **p = 7.1×10⁻⁶**, Cohen's d = 0.26

All three baselines rejected at **p < 0.001**.

---

## Causal Analysis (Doubly Robust ATE)

| Intervention | Naive ATE | IPW ATE | Doubly Robust ATE | Selection Bias | E-value |
|---|---:|---:|---:|---:|:---:|
| Financial aid | -0.64 | -0.51 | **-0.47 ± 0.01** | +0.16 | 4.39 |
| Mentoring | -0.25 | -0.19 | **-0.17 ± 0.01** | +0.08 | 2.40 |

**Interpretation**: naive correlations overestimate effect magnitude by ~25-33% due to self-selection. The doubly robust estimator is unbiased under either correct propensity or correct outcome model. E-value = 4.39 means an unobserved confounder would need to be 4.39× stronger than any measured confounder to explain away the financial aid effect.

---

## Fairness Audit

All 4 protected groups (gender, international, displaced, scholarship) currently show gaps above the 5pp fairness threshold — this is reported honestly as future-work motivation for fairness-constrained RL training (CPO / Lagrangian methods).

---

## API Endpoints

### Authentication
| Method | Endpoint | Roles |
|--------|----------|-------|
| POST | `/api/auth/login` | All |
| POST | `/api/auth/logout` | All |
| GET | `/api/auth/me` | Authenticated |
| GET | `/api/auth/users` | Admin |
| POST | `/api/auth/users` | Admin |

### Students (Counselor + Student-own-data)
| Method | Endpoint |
|--------|----------|
| GET | `/api/dashboard` |
| GET | `/api/students` |
| GET | `/api/student/{id}` |
| POST/PUT/DELETE | `/api/student/{id}` |
| GET | `/api/student/{id}/predict` |
| GET | `/api/student/{id}/recommendations` |
| GET | `/api/student/{id}/progression` |
| POST | `/api/student/{id}/simulate` |
| GET | `/api/student/{id}/compare` |
| GET | `/api/student/{id}/timeline` |
| GET | `/api/student/{id}/best-intervention` |
| GET | `/api/student/{id}/context` |
| **GET** | **`/api/student/{id}/optimal-plan`** (RL) |
| **GET** | **`/api/student/{id}/optimal-plan/explain`** (RL + XRL) |

### Researcher
| Method | Endpoint |
|--------|----------|
| GET | `/api/metrics` |
| GET | `/api/importance` |
| GET | `/api/roc` |
| **GET** | **`/api/rl/evaluation`** (rigorous results) |

### Admin
| Method | Endpoint |
|--------|----------|
| GET | `/api/admin/system` |
| GET | `/api/admin/models` |
| GET | `/api/admin/dataset` |
| GET | `/api/admin/logs` |
| POST | `/api/admin/retrain` |
| POST | `/api/admin/dataset/reset` |

---

## Research Contributions

1. **Two-stage Digital Twin** — validated through +3.2pp AUC gain when Sem 2 data arrives
2. **Data-driven intervention effects** — learned from graduate/dropout trajectories, not hardcoded
3. **Model-based RL** — digital twin as world model (no need for live exploration)
4. **Explainable RL** — SHAP on Q-values + counterfactual + natural language (not just prediction explainability)
5. **Causal-grounded evaluation** — IPW + Doubly Robust ATE with E-value sensitivity
6. **Fairness audit** — per-protected-group gap analysis flagging future work

---

## Pipelines — Run Order

```bash
# 1. Classification + SHAP + intervention effects (~2 min)
python pipelines/ml_pipeline.py

# 2. RL training (~30 sec)
python pipelines/train_rl.py

# 3. Rigorous evaluation (~5 min)
python pipelines/evaluate_rl_rigorous.py
```

All pipelines save to `models/`. Restart `main.py` after retraining to reload.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn, Starlette middleware |
| Classification | XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer), XRL via SHAP on Q-values |
| RL | Fitted Q-Iteration with XGBoost regressors (offline RL, no deep learning) |
| Causal Inference | Inverse Propensity Weighting, Doubly Robust estimator |
| Statistics | SciPy (paired t-test, Wilcoxon, bootstrap) |
| Data | pandas, NumPy |
| Visualization | matplotlib (SHAP plots), Chart.js (dashboard) |
| Frontend | Vanilla HTML5, CSS3, JavaScript |
| Auth | Token-based sessions, role middleware |

---

## Dataset

Source: **UCI ML Repository — "Predict Students' Dropout and Academic Success"** (Realinho et al., 2022)
https://doi.org/10.24432/C5MC89

- 4,424 student records × 35 features + 7 engineered
- Target: Dropout (1,421) vs Non-Dropout (3,003)
- Class ratio: 1:2.11 (imbalanced, handled via `scale_pos_weight`)
