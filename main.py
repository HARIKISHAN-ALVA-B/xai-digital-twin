"""
=============================================================================
 FILE: main.py
=============================================================================
 FastAPI backend with:
   - Dashboard summary APIs
   - Student CRUD (Create, Read, Update, Delete) with JSON persistence
   - XGBoost prediction + SHAP explanation per student
   - Digital Twin progression (Stage 1 -> Stage 2)
   - Intervention simulation / comparison
   - CSV export, ROC curve data, contextual stats
=============================================================================
"""

import os, sys, json, pickle, uuid
import datetime
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.digital_twin import DigitalTwinManager
from services.recommendations import generate_recommendations
from services.auth import (
    authenticate, create_session, get_session, destroy_session,
    check_permission, get_all_users, create_user, delete_user,
    ROLE_PERMISSIONS,
)
from services.rl_environment import StudentEnvironment, ACTIONS
from services.rl_policies import FQIAgent
from services.rl_explainer import RLExplainer


# --- Custom JSON response that handles numpy types ---
class NumpyJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, cls=_NumpyEncoder).encode("utf-8")

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    load_resources()
    yield

app = FastAPI(title="XAI Digital Twin Framework", default_response_class=NumpyJSONResponse, lifespan=lifespan)

# Serve static files (plots, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Globals ---
XGB_S1 = XGB_S2 = LR_S1 = None
SCALER_S1 = SCALER_S2 = None
EXPLAINER_S1 = EXPLAINER_S2 = None
FEAT_S1 = FEAT_S2 = None
METRICS_S1 = METRICS_S2 = None
IMP_S1 = IMP_S2 = None
MANAGER = DigitalTwinManager()
DF = None

# Data-driven intervention models (Block 7)
SEM2_PREDICTOR = None
LEARNED_EFFECTS = None

# RL policy + XRL (Phase 1-3 + novelty layers)
RL_AGENT = None
RL_EXPLAINER = None

MODELS_DIR = "models"
CHANGES_FILE = "data/crud_changes.json"


def load_resources():
    global XGB_S1, XGB_S2, LR_S1, SCALER_S1, SCALER_S2
    global EXPLAINER_S1, EXPLAINER_S2, FEAT_S1, FEAT_S2
    global METRICS_S1, METRICS_S2, IMP_S1, IMP_S2, MANAGER, DF
    global SEM2_PREDICTOR, LEARNED_EFFECTS, RL_AGENT, RL_EXPLAINER

    pkl_files = [
        ("xgb_stage1", "XGB_S1"), ("xgb_stage2", "XGB_S2"), ("lr_stage1", "LR_S1"),
        ("scaler_stage1", "SCALER_S1"), ("scaler_stage2", "SCALER_S2"),
        ("shap_explainer_s1", "EXPLAINER_S1"), ("shap_explainer_s2", "EXPLAINER_S2"),
    ]
    g = globals()
    for fname, var in pkl_files:
        with open(f"{MODELS_DIR}/{fname}.pkl", "rb") as f:
            g[var] = pickle.load(f)

    json_files = [
        ("features_stage1", "FEAT_S1"), ("features_stage2", "FEAT_S2"),
        ("metrics_stage1", "METRICS_S1"), ("metrics_stage2", "METRICS_S2"),
        ("importance_stage1", "IMP_S1"), ("importance_stage2", "IMP_S2"),
    ]
    for fname, var in json_files:
        with open(f"{MODELS_DIR}/{fname}.json", "r") as f:
            g[var] = json.load(f)

    XGB_S1 = g["XGB_S1"]; XGB_S2 = g["XGB_S2"]; LR_S1 = g["LR_S1"]
    SCALER_S1 = g["SCALER_S1"]; SCALER_S2 = g["SCALER_S2"]
    EXPLAINER_S1 = g["EXPLAINER_S1"]; EXPLAINER_S2 = g["EXPLAINER_S2"]
    FEAT_S1 = g["FEAT_S1"]; FEAT_S2 = g["FEAT_S2"]
    METRICS_S1 = g["METRICS_S1"]; METRICS_S2 = g["METRICS_S2"]
    IMP_S1 = g["IMP_S1"]; IMP_S2 = g["IMP_S2"]

    # Load data-driven intervention models (optional — backward compatible)
    sem2_path = f"{MODELS_DIR}/sem2_predictor.pkl"
    effects_path = f"{MODELS_DIR}/intervention_effects.json"
    if os.path.exists(sem2_path):
        with open(sem2_path, "rb") as f:
            SEM2_PREDICTOR = pickle.load(f)
        print("[APP] Loaded Sem 2 outcome predictor")
    if os.path.exists(effects_path):
        with open(effects_path, "r") as f:
            LEARNED_EFFECTS = json.load(f).get("effects")
        print(f"[APP] Loaded data-driven intervention effects: {list(LEARNED_EFFECTS.keys())}")

    # Load RL policy if available
    rl_path = f"{MODELS_DIR}/rl_policy.pkl"
    if os.path.exists(rl_path):
        try:
            RL_AGENT = FQIAgent.load(rl_path)
            RL_EXPLAINER = RLExplainer(RL_AGENT)
            print(f"[APP] Loaded RL policy ({RL_AGENT.n_actions} actions, gamma={RL_AGENT.gamma}) + XRL explainer")
        except Exception as e:
            print(f"[APP] Failed to load RL policy: {e}")

    DF = pd.read_csv("data/processed_data.csv")
    n = MANAGER.build_from_dataframe(DF, XGB_S1, XGB_S2, FEAT_S1, FEAT_S2)
    print(f"[APP] Loaded {n} digital twins from {len(DF)} records")

    _apply_saved_changes()


# ===================================================================
#  Persistence
# ===================================================================

def _apply_saved_changes():
    if not os.path.exists(CHANGES_FILE):
        return
    with open(CHANGES_FILE, "r") as f:
        changes = json.load(f)

    for sid in changes.get("deleted", []):
        MANAGER.delete_twin(sid)

    for entry in changes.get("added", []):
        sid = entry["student_id"]
        if not MANAGER.get_twin(sid):
            MANAGER.add_student(sid, entry["data"], XGB_S1, XGB_S2, FEAT_S1, FEAT_S2)

    for entry in changes.get("updated", []):
        twin = MANAGER.get_twin(entry["student_id"])
        if twin:
            for k, v in entry["data"].items():
                twin.raw_data[k] = v
            twin.raw_data = compute_engineered_features(twin.raw_data)
            twin.update_to_stage2(XGB_S2, FEAT_S2)

    a = len(changes.get("added", []))
    d = len(changes.get("deleted", []))
    u = len(changes.get("updated", []))
    print(f"[APP] Applied saved changes: {a} added, {d} deleted, {u} updated")


def _save_change(change_type, student_id, data=None):
    if os.path.exists(CHANGES_FILE):
        with open(CHANGES_FILE, "r") as f:
            changes = json.load(f)
    else:
        changes = {"added": [], "deleted": [], "updated": []}

    if change_type == "add":
        changes["deleted"] = [s for s in changes["deleted"] if s != student_id]
        changes["added"].append({"student_id": student_id, "data": data})
    elif change_type == "delete":
        was_added = any(e["student_id"] == student_id for e in changes["added"])
        changes["added"] = [e for e in changes["added"] if e["student_id"] != student_id]
        changes["updated"] = [e for e in changes["updated"] if e["student_id"] != student_id]
        if not was_added:
            changes["deleted"].append(student_id)
    elif change_type == "update":
        changes["updated"] = [e for e in changes["updated"] if e["student_id"] != student_id]
        changes["updated"].append({"student_id": student_id, "data": data})

    with open(CHANGES_FILE, "w") as f:
        json.dump(changes, f, indent=2, default=str)


# ===================================================================
#  Helper
# ===================================================================

def compute_engineered_features(d):
    """Given a raw student dict, compute all derived features in-place."""
    enr1 = max(d.get("Curricular units 1st sem (enrolled)", 1), 1)
    d["Approval_rate_sem1"] = d.get("Curricular units 1st sem (approved)", 0) / enr1
    d["Failure_rate_sem1"]  = 1 - d["Approval_rate_sem1"]

    enr2 = max(d.get("Curricular units 2nd sem (enrolled)", 1), 1)
    d["Approval_rate_sem2"] = d.get("Curricular units 2nd sem (approved)", 0) / enr2
    d["Failure_rate_sem2"]  = 1 - d["Approval_rate_sem2"]

    d["Grade_change"]    = d.get("Curricular units 2nd sem (grade)", 0) - d.get("Curricular units 1st sem (grade)", 0)
    d["Approved_change"] = d.get("Curricular units 2nd sem (approved)", 0) - d.get("Curricular units 1st sem (approved)", 0)

    financial = d.get("Debtor", 0) * 0.5 + (1 - d.get("Tuition fees up to date", 1)) * 0.5
    evals1 = max(d.get("Curricular units 1st sem (evaluations)", 1), 1)
    fail_ratio = max(0, min(1, (evals1 - d.get("Curricular units 1st sem (approved)", 0)) / evals1))
    disengage = max(0, min(1, d.get("Curricular units 1st sem (without evaluations)", 0) / enr1))
    grade_def = max(0, min(1, (10 - d.get("Curricular units 1st sem (grade)", 0)) / 10))
    d["Stress_indicator"] = round((0.25 * financial + 0.35 * fail_ratio + 0.20 * disengage + 0.20 * grade_def) * 10, 2)

    return d


# ===================================================================
#  Startup event — load models when server starts
# ===================================================================

# Resources loaded via lifespan handler above


# ===================================================================
#  AUTH MIDDLEWARE — Role-Based Access Control
# ===================================================================

from starlette.middleware.base import BaseHTTPMiddleware

class RoleMiddleware(BaseHTTPMiddleware):
    """Check permissions on /api/* routes. Skips auth for login, static, and root."""

    OPEN_PATHS = {"/", "/api/auth/login", "/api/auth/roles", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        # Skip auth for open paths, static files, and favicon
        if path in self.OPEN_PATHS or path.startswith("/static") or path == "/favicon.ico":
            return await call_next(request)

        # All /api/* routes require auth
        if path.startswith("/api/"):
            session = get_session(request)
            if not session:
                return JSONResponse({"error": "Authentication required", "login_url": "/api/auth/login"}, status_code=401)

            # Extract student ID from path if present
            sid = None
            parts = path.split("/")
            if len(parts) >= 4 and parts[1] == "api":
                if parts[2] == "student" and parts[3] not in ("",):
                    sid = parts[3]
                elif parts[2] == "connect" and parts[3] not in ("", "message", "meeting"):
                    sid = parts[3]

            allowed, reason = check_permission(session, method, path, sid=sid)
            if not allowed:
                return JSONResponse({"error": reason, "role": session["role"]}, status_code=403)

        return await call_next(request)

app.add_middleware(RoleMiddleware)


# ===================================================================
#  AUTH ROUTES — Login, Logout, Session, User Management
# ===================================================================

@app.post("/api/auth/login")
async def api_login(request: Request):
    data = await request.json()
    username = data.get("username", "")
    password = data.get("password", "")
    user = authenticate(username, password)
    if not user:
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    token = create_session(user)
    response = NumpyJSONResponse({"message": "Login successful", "user": user, "token": token})
    response.set_cookie("session_token", token, httponly=True, samesite="lax")
    return response


@app.post("/api/auth/logout")
async def api_logout(request: Request):
    destroy_session(request)
    response = NumpyJSONResponse({"message": "Logged out"})
    response.delete_cookie("session_token")
    return response


@app.get("/api/auth/me")
async def api_auth_me(request: Request):
    session = get_session(request)
    if not session:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    return session


@app.get("/api/auth/roles")
async def api_auth_roles():
    return {role: {"description": info["description"]} for role, info in ROLE_PERMISSIONS.items()}


@app.get("/api/auth/users")
async def api_auth_users(request: Request):
    """Admin only: list all users."""
    return get_all_users()


@app.post("/api/auth/users")
async def api_create_user(request: Request):
    """Admin only: create a user."""
    data = await request.json()
    ok, msg = create_user(
        data.get("username", ""), data.get("password", ""),
        data.get("role", "student"), data.get("name", ""),
        data.get("student_id"),
    )
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    return {"message": msg}


@app.delete("/api/auth/users/{username}")
async def api_delete_user(username: str):
    """Admin only: delete a user."""
    ok, msg = delete_user(username)
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    return {"message": msg}


# ===================================================================
#  ROUTES — Pages
# ===================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return FileResponse("templates/dashboard.html", media_type="text/html")


# ===================================================================
#  API — Dashboard
# ===================================================================

@app.get("/api/dashboard")
async def api_dashboard():
    stats = MANAGER.get_dashboard_stats()
    stats["metrics_s1"] = METRICS_S1
    stats["metrics_s2"] = METRICS_S2
    stats["importance_s1"] = IMP_S1
    stats["importance_s2"] = IMP_S2
    summaries = MANAGER.get_all_summaries()
    stats["outcome_counts"] = {
        "Graduate": sum(1 for s in summaries if s.get("actual_target") == "Graduate"),
        "Dropout":  sum(1 for s in summaries if s.get("actual_target") == "Dropout"),
        "Enrolled": sum(1 for s in summaries if s.get("actual_target") == "Enrolled"),
    }
    return stats


# ===================================================================
#  API — Student CRUD (with persistence)
# ===================================================================

@app.get("/api/students")
async def api_students(search: str = "", risk: str = "all", sort: str = "id"):
    search = search.strip().lower()
    summaries = MANAGER.get_all_summaries()
    if search:
        summaries = [s for s in summaries if search in s["student_id"].lower()
                     or search in s.get("actual_target", "").lower()]
    if risk == "at_risk":
        summaries = [s for s in summaries if s["risk_label"] == 1]
    elif risk == "safe":
        summaries = [s for s in summaries if s["risk_label"] == 0]
    if sort == "risk":
        summaries.sort(key=lambda x: x["risk_prob"], reverse=True)
    else:
        summaries.sort(key=lambda x: x["student_id"])
    return {"students": summaries, "total": len(summaries)}


@app.get("/api/student/{sid}")
async def api_student_detail(sid: str):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    return twin.to_dict()


@app.post("/api/student", status_code=201)
async def api_create_student(request: Request):
    data = await request.json()
    if not data or "student_id" not in data:
        return NumpyJSONResponse({"error": "student_id required"}, status_code=400)

    sid = data.pop("student_id")
    if MANAGER.get_twin(sid):
        return NumpyJSONResponse({"error": f"{sid} already exists"}, status_code=409)

    for f in FEAT_S2:
        if f not in data:
            data[f] = 0
    data = compute_engineered_features(data)

    twin = MANAGER.add_student(sid, data, XGB_S1, XGB_S2, FEAT_S1, FEAT_S2)
    _save_change("add", sid, data)
    return {"message": f"{sid} created", "student": twin.to_dict()}


@app.put("/api/student/{sid}")
async def api_update_student(sid: str, request: Request):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)

    data = await request.json()
    for k, v in data.items():
        twin.raw_data[k] = v
    twin.raw_data = compute_engineered_features(twin.raw_data)
    new_state = twin.update_to_stage2(XGB_S2, FEAT_S2)
    _save_change("update", sid, data)
    return {"message": f"{sid} updated to Stage 2", "new_state": new_state}


@app.delete("/api/student/{sid}")
async def api_delete_student(sid: str):
    if MANAGER.delete_twin(sid):
        _save_change("delete", sid)
        return {"message": f"{sid} deleted"}
    return NumpyJSONResponse({"error": "Student not found"}, status_code=404)


# ===================================================================
#  API — Prediction & SHAP
# ===================================================================

@app.get("/api/student/{sid}/predict")
async def api_predict(sid: str, stage: str = "2"):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)

    features  = FEAT_S2 if stage == "2" else FEAT_S1
    model     = XGB_S2  if stage == "2" else XGB_S1
    explainer = EXPLAINER_S2 if stage == "2" else EXPLAINER_S1

    feat_vals = pd.DataFrame([twin.raw_data])[features].astype(float)
    prob  = float(model.predict_proba(feat_vals)[0][1])
    label = int(prob > 0.5)

    sv = explainer.shap_values(feat_vals)[0]
    contributions = {}
    for i, f in enumerate(features):
        contributions[f] = {
            "value": round(float(feat_vals.iloc[0, i]), 4),
            "shap": round(float(sv[i]), 4),
            "direction": "increases risk" if sv[i] > 0 else "decreases risk"
        }
    sorted_c = dict(sorted(contributions.items(), key=lambda x: abs(x[1]["shap"]), reverse=True))

    top5 = []
    for f, info in list(sorted_c.items())[:5]:
        top5.append({"feature": f, "value": info["value"],
                     "shap": info["shap"], "direction": info["direction"]})

    return {
        "student_id": sid, "stage": int(stage),
        "risk_prob": round(prob, 4), "risk_label": label,
        "base_value": round(float(explainer.expected_value), 4),
        "contributions": sorted_c, "top_factors": top5
    }


# ===================================================================
#  API — What-If Prediction (stateless, non-mutating)
# ===================================================================

@app.post("/api/student/{sid}/what-if")
async def api_what_if(sid: str, request: Request):
    """
    Predict risk with feature overrides, WITHOUT modifying the twin.
    Body: JSON dict of {feature_name: new_value}
    """
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)

    overrides = await request.json()
    data = dict(twin.raw_data)
    data.update(overrides)

    # Recompute derived features if academic features were overridden
    for sem, lbl in [(1, '1st'), (2, '2nd')]:
        enrolled = max(data.get(f'Curricular units {lbl} sem (enrolled)', 1), 1)
        approved = data.get(f'Curricular units {lbl} sem (approved)', 0)
        data[f'Approval_rate_sem{sem}'] = approved / enrolled
        data[f'Failure_rate_sem{sem}'] = 1 - data[f'Approval_rate_sem{sem}']
    data['Grade_change'] = data.get('Curricular units 2nd sem (grade)', 0) - data.get('Curricular units 1st sem (grade)', 0)
    data['Approved_change'] = data.get('Curricular units 2nd sem (approved)', 0) - data.get('Curricular units 1st sem (approved)', 0)

    X = pd.DataFrame([data])[FEAT_S2].astype(float)
    risk = float(XGB_S2.predict_proba(X)[0][1])
    orig_X = pd.DataFrame([twin.raw_data])[FEAT_S2].astype(float)
    orig_risk = float(XGB_S2.predict_proba(orig_X)[0][1])

    return {"risk_prob": round(risk, 4), "original_risk": round(orig_risk, 4),
            "delta": round(orig_risk - risk, 4)}


# ===================================================================
#  API — Counterfactual Targets ("What needs to change?")
# ===================================================================

@app.get("/api/student/{sid}/counterfactual")
async def api_counterfactual(sid: str, target_risk: float = 0.30):
    """
    Find minimum feature changes to move student below target risk.
    Returns actionable targets sorted by risk-reduction impact.
    """
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)

    data = dict(twin.raw_data)
    features = FEAT_S2
    model = XGB_S2

    X = pd.DataFrame([data])[features].astype(float)
    current_risk = float(model.predict_proba(X)[0][1])

    if current_risk < target_risk:
        return {"student_id": sid, "current_risk": round(current_risk, 4),
                "target_risk": target_risk, "status": "already_safe", "targets": []}

    # Get SHAP values to identify risk-increasing features
    sv = EXPLAINER_S2.shap_values(X)[0]

    # Feature metadata for search bounds
    _binary = {'Tuition fees up to date', 'Scholarship holder', 'Debtor',
               'Displaced', 'International', 'Daytime/evening attendance', 'Gender'}
    _rates  = {'Approval_rate_sem1', 'Approval_rate_sem2',
               'Failure_rate_sem1', 'Failure_rate_sem2'}
    _grades = {f'Curricular units {s} sem (grade)' for s in ['1st', '2nd']}

    targets = []
    for i, feat in enumerate(features):
        shap_val = float(sv[i])
        if shap_val <= 0.005:
            continue                   # skip features not increasing risk

        orig = float(data.get(feat, 0))

        if feat in _binary:
            # Try flipping
            test = dict(data)
            test[feat] = 1.0 - orig
            Xt = pd.DataFrame([test])[features].astype(float)
            new_risk = float(model.predict_proba(Xt)[0][1])
            if new_risk < current_risk:
                targets.append({
                    "feature": feat, "current": orig,
                    "target": 1.0 - orig, "change": 1.0 - 2 * orig,
                    "new_risk": round(new_risk, 4),
                    "reduction": round(current_risk - new_risk, 4),
                    "reaches_safe": new_risk < target_risk,
                    "type": "binary"
                })
            continue

        # Continuous: determine direction and step
        if feat in _rates:
            direction, step, lo, hi = (1 if 'Approval' in feat else -1), 0.05, 0, 1
        elif feat in _grades:
            direction, step, lo, hi = 1, 0.5, 0, 20
        elif 'approved' in feat.lower():
            direction, step, lo, hi = 1, 1, 0, 15
        elif feat == 'Stress_indicator':
            direction, step, lo, hi = -1, 0.5, 0, 10
        elif feat == 'Age at enrollment':
            continue     # non-actionable
        else:
            direction = -1 if shap_val > 0 else 1
            step, lo, hi = 1, 0, 100

        best = None
        for mult in range(1, 30):
            test = dict(data)
            nv = orig + direction * step * mult
            nv = max(lo, min(hi, nv))
            if nv == orig:
                break
            test[feat] = nv
            # recompute dependent derived features
            enrolled = lambda s: max(test.get(f'Curricular units {s} sem (enrolled)', 1), 1)
            if 'sem (approved)' in feat or 'sem (grade)' in feat:
                for s, lbl in [(1, '1st'), (2, '2nd')]:
                    app = test.get(f'Curricular units {lbl} sem (approved)', 0)
                    test[f'Approval_rate_sem{s}'] = app / enrolled(lbl)
                    test[f'Failure_rate_sem{s}'] = 1 - test[f'Approval_rate_sem{s}']
                test['Grade_change'] = test.get('Curricular units 2nd sem (grade)', 0) - test.get('Curricular units 1st sem (grade)', 0)
                test['Approved_change'] = test.get('Curricular units 2nd sem (approved)', 0) - test.get('Curricular units 1st sem (approved)', 0)
            Xt = pd.DataFrame([test])[features].astype(float)
            new_risk = float(model.predict_proba(Xt)[0][1])
            if new_risk < current_risk:
                best = {"feature": feat, "current": round(orig, 2),
                        "target": round(nv, 2), "change": round(nv - orig, 2),
                        "new_risk": round(new_risk, 4),
                        "reduction": round(current_risk - new_risk, 4),
                        "reaches_safe": new_risk < target_risk, "type": "continuous"}
                if new_risk < target_risk:
                    break
        if best:
            targets.append(best)

    targets.sort(key=lambda x: x["reduction"], reverse=True)
    return {"student_id": sid, "current_risk": round(current_risk, 4),
            "target_risk": target_risk, "targets": targets[:6]}


# ===================================================================
#  API — SHAP-Based Counseling Recommendations
# ===================================================================

@app.get("/api/student/{sid}/recommendations")
async def api_recommendations(sid: str, stage: str = "2"):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)

    features  = FEAT_S2 if stage == "2" else FEAT_S1
    model     = XGB_S2  if stage == "2" else XGB_S1
    explainer = EXPLAINER_S2 if stage == "2" else EXPLAINER_S1

    feat_vals = pd.DataFrame([twin.raw_data])[features].astype(float)
    prob  = float(model.predict_proba(feat_vals)[0][1])
    sv = explainer.shap_values(feat_vals)[0]

    contributions = {}
    for i, f in enumerate(features):
        contributions[f] = {
            "value": round(float(feat_vals.iloc[0, i]), 4),
            "shap": round(float(sv[i]), 4),
            "direction": "increases risk" if sv[i] > 0 else "decreases risk",
        }

    result = generate_recommendations(contributions)
    result["student_id"] = sid
    result["risk_prob"] = round(prob, 4)
    result["risk_label"] = int(prob > 0.5)
    return result


# ===================================================================
#  API — Twin Progression & Simulation
# ===================================================================

@app.get("/api/student/{sid}/progression")
async def api_progression(sid: str):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    return twin.get_progression()


@app.post("/api/student/{sid}/simulate")
async def api_simulate(sid: str, request: Request):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    data = await request.json()
    intervention = data.get("intervention", "none")
    result = twin.simulate_intervention(intervention, XGB_S2, FEAT_S2, learned_effects=LEARNED_EFFECTS)
    return result


@app.get("/api/student/{sid}/compare")
async def api_compare(sid: str):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    return twin.compare_all_interventions(XGB_S2, FEAT_S2, learned_effects=LEARNED_EFFECTS)


# ===================================================================
#  API — Time-Based Simulation (Task 1 & 2)
# ===================================================================

@app.get("/api/student/{sid}/timeline")
async def api_timeline(sid: str, weeks: int = 6, intervention: str = "none"):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    weeks = max(1, min(12, weeks))
    timeline = twin.simulate_weeks(
        weeks, intervention, XGB_S2, FEAT_S2,
        learned_effects=LEARNED_EFFECTS,
        sem2_predictor=SEM2_PREDICTOR,
        sem1_features=FEAT_S1,
    )
    return NumpyJSONResponse({
        "student_id": sid,
        "intervention": intervention,
        "weeks": weeks,
        "current_risk": twin.to_dict()["risk_prob"],
        "data_driven": LEARNED_EFFECTS is not None,
        "timeline": timeline,
    })


@app.get("/api/student/{sid}/best-intervention")
async def api_best_intervention(sid: str, weeks: int = 6):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    weeks = max(1, min(12, weeks))
    result = twin.get_best_intervention(
        XGB_S2, FEAT_S2, weeks,
        learned_effects=LEARNED_EFFECTS,
        sem2_predictor=SEM2_PREDICTOR,
        sem1_features=FEAT_S1,
    )
    result["student_id"] = sid
    return NumpyJSONResponse(result)


# ===================================================================
#  API — Contextual Stats
# ===================================================================

@app.get("/api/student/{sid}/context")
async def api_student_context(sid: str):
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)

    all_s = MANAGER.get_all_summaries()
    total = len(all_s)
    td = twin.to_dict()
    risk_prob = td["risk_prob"]

    lower = sum(1 for s in all_s if s["risk_prob"] < risk_prob)
    percentile = round(lower / total * 100, 1) if total > 0 else 0

    cat = td["risk_category"]
    similar = [s for s in all_s if s["risk_category"] == cat]
    outcomes = {}
    for s in similar:
        t = s["actual_target"]
        outcomes[t] = outcomes.get(t, 0) + 1

    return {
        "risk_percentile": percentile,
        "total_students": total,
        "similar_count": len(similar),
        "similar_outcomes": outcomes,
        "risk_category": cat,
    }


# ===================================================================
#  API — Model Metrics, Importance, ROC
# ===================================================================

@app.get("/api/metrics")
async def api_metrics():
    return {"stage1": METRICS_S1, "stage2": METRICS_S2}

@app.get("/api/importance")
async def api_importance():
    return {"stage1": IMP_S1, "stage2": IMP_S2}

@app.get("/api/at-risk")
async def api_at_risk():
    return {"students": MANAGER.get_at_risk()}

@app.get("/api/roc")
async def api_roc():
    roc = {}
    for stage, metrics in [("stage1", METRICS_S1), ("stage2", METRICS_S2)]:
        roc[stage] = {}
        for mn in ["Logistic Regression", "XGBoost", "Ensemble"]:
            m = metrics.get(mn, {})
            if "roc_fpr" in m:
                roc[stage][mn] = {"fpr": m["roc_fpr"], "tpr": m["roc_tpr"], "auc": m["auc"]}
    return roc


# ===================================================================
#  API — Reinforcement Learning: Optimal Intervention Plan
# ===================================================================

@app.get("/api/student/{sid}/optimal-plan")
async def api_optimal_plan(sid: str, horizon: int = 12):
    """
    Return RL-optimized intervention plan for a student.
    Uses trained FQI policy to roll out action sequence over `horizon` weeks.
    """
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    if RL_AGENT is None or not RL_AGENT.trained:
        return NumpyJSONResponse(
            {"error": "RL policy not available. Train with `python pipelines/train_rl.py`."},
            status_code=503,
        )

    horizon = max(1, min(24, horizon))
    env = StudentEnvironment(
        initial_raw=twin.raw_data,
        model_s2=XGB_S2,
        features_s2=FEAT_S2,
        learned_effects=LEARNED_EFFECTS,
        horizon=horizon,
    )

    initial_risk = env.current_risk
    state = env.reset()
    plan = []
    total_reward = 0.0
    done = False
    while not done:
        action_idx = RL_AGENT.policy(state)
        q_vals = RL_AGENT.q_values(state).tolist()
        state, reward, done, info = env.step(action_idx)
        total_reward += reward
        plan.append({
            "week": info["week"],
            "action": info["action"],
            "projected_risk": round(info["risk"], 4),
            "reward": round(reward, 3),
            "q_values": {ACTIONS[i]: round(q, 3) for i, q in enumerate(q_vals)},
        })

    final_risk = env.current_risk

    # Cost summary
    action_counts = {a: 0 for a in ACTIONS}
    for step in plan:
        action_counts[step["action"]] += 1
    cost_map = {"none": 0.0, "mentoring": 0.3, "financial_aid": 0.5}
    total_cost = sum(action_counts[a] * cost_map[a] for a in ACTIONS)

    return NumpyJSONResponse({
        "student_id": sid,
        "horizon": horizon,
        "initial_risk": round(initial_risk, 4),
        "final_risk": round(final_risk, 4),
        "risk_reduction": round(initial_risk - final_risk, 4),
        "total_reward": round(total_reward, 3),
        "total_cost": round(total_cost, 2),
        "action_summary": action_counts,
        "plan": plan,
    })


@app.get("/api/student/{sid}/optimal-plan/explain")
async def api_plan_explain(sid: str, horizon: int = 12):
    """
    XRL endpoint — return the RL-optimized plan with per-step SHAP attributions,
    counterfactuals, and natural-language explanations.
    """
    twin = MANAGER.get_twin(sid)
    if not twin:
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    if RL_AGENT is None or RL_EXPLAINER is None:
        return NumpyJSONResponse({"error": "RL policy not available."}, status_code=503)

    horizon = max(1, min(24, horizon))

    def env_factory():
        return StudentEnvironment(
            initial_raw=twin.raw_data,
            model_s2=XGB_S2, features_s2=FEAT_S2,
            learned_effects=LEARNED_EFFECTS, horizon=horizon,
        )

    env = env_factory()
    initial_risk = env.current_risk
    state = env.reset()
    plan_with_xrl = []
    done = False
    while not done:
        explanation = RL_EXPLAINER.explain_action(state, env_factory=env_factory)
        action_idx = ACTIONS.index(explanation['chosen_action'])
        state, reward, done, info = env.step(action_idx)
        plan_with_xrl.append({
            "week": info["week"],
            "action": info["action"],
            "projected_risk": round(info["risk"], 4),
            "reward": round(reward, 3),
            "q_values": explanation['q_values'],
            "decision_margin": round(explanation['decision_margin'], 3),
            "top_factors": explanation['top_factors'][:3],
            "counterfactual": explanation['counterfactual'],
            "explanation": explanation['explanation'],
        })

    sequence_explanation = RL_EXPLAINER.explain_sequence(plan_with_xrl)

    return NumpyJSONResponse({
        "student_id": sid,
        "horizon": horizon,
        "initial_risk": round(initial_risk, 4),
        "final_risk": round(env.current_risk, 4),
        "risk_reduction": round(initial_risk - env.current_risk, 4),
        "plan": plan_with_xrl,
        "sequence_explanation": sequence_explanation,
    })


# ===================================================================
#  API — Counselor ↔ Student Connect (messages + meetings)
# ===================================================================

CONNECT_FILE = "data/connect.json"

def _load_connect():
    if os.path.exists(CONNECT_FILE):
        data = json.load(open(CONNECT_FILE))
        data.setdefault("notes", [])
        return data
    return {"messages": [], "meetings": [], "notes": []}

def _save_connect(data):
    with open(CONNECT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


@app.get("/api/connect/{sid}")
async def api_connect_get(sid: str, request: Request):
    session = get_session(request)
    if session and session["role"] == "student" and session.get("student_id") != sid:
        return JSONResponse({"error": "Access denied"}, status_code=403)
    connect = _load_connect()
    msgs = [m for m in connect["messages"] if m["student_id"] == sid]
    meets = [m for m in connect["meetings"] if m["student_id"] == sid]
    # Auto-mark all messages as read when a student fetches their inbox
    if session and session["role"] == "student":
        changed = False
        for m in connect["messages"]:
            if m["student_id"] == sid and not m["read"]:
                m["read"] = True
                changed = True
        if changed:
            _save_connect(connect)
    return {"student_id": sid, "messages": msgs, "meetings": meets}


@app.post("/api/connect/{sid}/message")
async def api_connect_message(sid: str, request: Request):
    if not MANAGER.get_twin(sid):
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    session = get_session(request)
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "Message text required"}, status_code=400)
    connect = _load_connect()
    msg = {
        "id": str(uuid.uuid4())[:8],
        "student_id": sid,
        "from": session["name"] if session else "Counselor",
        "text": text,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "read": False,
    }
    connect["messages"].append(msg)
    _save_connect(connect)
    return {"message": "Sent", "data": msg}


@app.post("/api/connect/{sid}/meeting")
async def api_connect_meeting(sid: str, request: Request):
    if not MANAGER.get_twin(sid):
        return NumpyJSONResponse({"error": "Student not found"}, status_code=404)
    session = get_session(request)
    data = await request.json()
    date = (data.get("date") or "").strip()
    time = (data.get("time") or "").strip()
    reason = (data.get("reason") or "").strip()
    if not date or not time:
        return JSONResponse({"error": "Date and time required"}, status_code=400)
    connect = _load_connect()
    meet = {
        "id": str(uuid.uuid4())[:8],
        "student_id": sid,
        "counselor": session["name"] if session else "Counselor",
        "date": date,
        "time": time,
        "reason": reason or "Academic counseling session",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    connect["meetings"].append(meet)
    _save_connect(connect)
    return {"message": "Meeting scheduled", "data": meet}


@app.put("/api/connect/message/{mid}/read")
async def api_connect_message_read(mid: str):
    connect = _load_connect()
    for m in connect["messages"]:
        if m["id"] == mid:
            m["read"] = True
            _save_connect(connect)
            return {"message": "Marked as read"}
    return JSONResponse({"error": "Message not found"}, status_code=404)


@app.delete("/api/connect/message/{mid}")
async def api_connect_message_delete(mid: str):
    connect = _load_connect()
    before = len(connect["messages"])
    connect["messages"] = [m for m in connect["messages"] if m["id"] != mid]
    if len(connect["messages"]) == before:
        return JSONResponse({"error": "Message not found"}, status_code=404)
    _save_connect(connect)
    return {"message": "Deleted"}


@app.delete("/api/connect/meeting/{mid}")
async def api_connect_meeting_delete(mid: str):
    connect = _load_connect()
    before = len(connect["meetings"])
    connect["meetings"] = [m for m in connect["meetings"] if m["id"] != mid]
    if len(connect["meetings"]) == before:
        return JSONResponse({"error": "Meeting not found"}, status_code=404)
    _save_connect(connect)
    return {"message": "Deleted"}


# ===================================================================
#  API — Private counselor notes (#4)
# ===================================================================

@app.get("/api/connect/{sid}/notes")
async def api_notes_get(sid: str):
    connect = _load_connect()
    return {"notes": [n for n in connect["notes"] if n["student_id"] == sid]}

@app.post("/api/connect/{sid}/note")
async def api_notes_post(sid: str, request: Request):
    if not MANAGER.get_twin(sid):
        return JSONResponse({"error": "Student not found"}, status_code=404)
    session = get_session(request)
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "Note text required"}, status_code=400)
    connect = _load_connect()
    note = {
        "id": str(uuid.uuid4())[:8],
        "student_id": sid,
        "author": session["name"] if session else "Counselor",
        "text": text,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    connect["notes"].append(note)
    _save_connect(connect)
    return {"message": "Note saved", "data": note}

@app.delete("/api/connect/note/{nid}")
async def api_notes_delete(nid: str):
    connect = _load_connect()
    before = len(connect["notes"])
    connect["notes"] = [n for n in connect["notes"] if n["id"] != nid]
    if len(connect["notes"]) == before:
        return JSONResponse({"error": "Note not found"}, status_code=404)
    _save_connect(connect)
    return {"message": "Deleted"}


# ===================================================================
#  API — Intervention history (#2)
# ===================================================================

INTERVENTIONS_FILE = "data/interventions.json"

def _load_interventions():
    if os.path.exists(INTERVENTIONS_FILE):
        with open(INTERVENTIONS_FILE) as f:
            return json.load(f)
    return []

def _save_interventions(data):
    with open(INTERVENTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

@app.get("/api/student/{sid}/interventions")
async def api_interventions_get(sid: str):
    if not MANAGER.get_twin(sid):
        return JSONResponse({"error": "Student not found"}, status_code=404)
    logs = _load_interventions()
    return {"interventions": [i for i in logs if i["student_id"] == sid]}

@app.post("/api/student/{sid}/interventions")
async def api_interventions_post(sid: str, request: Request):
    if not MANAGER.get_twin(sid):
        return JSONResponse({"error": "Student not found"}, status_code=404)
    session = get_session(request)
    data = await request.json()
    itype = data.get("intervention_type", "none")
    if itype not in ("none", "mentoring", "financial_aid"):
        return JSONResponse({"error": "Invalid intervention type"}, status_code=400)
    entry = {
        "id": str(uuid.uuid4())[:8],
        "student_id": sid,
        "counselor": session["name"] if session else "Counselor",
        "intervention_type": itype,
        "notes": (data.get("notes") or "").strip(),
        "date": data.get("date") or datetime.date.today().isoformat(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    logs = _load_interventions()
    logs.append(entry)
    _save_interventions(logs)
    return {"message": "Logged", "data": entry}


# ===================================================================
#  API — Student acknowledgements (#3)
# ===================================================================

ACKNOWLEDGEMENTS_FILE = "data/acknowledgements.json"

def _load_acknowledgements():
    if os.path.exists(ACKNOWLEDGEMENTS_FILE):
        with open(ACKNOWLEDGEMENTS_FILE) as f:
            return json.load(f)
    return []

def _save_acknowledgements(data):
    with open(ACKNOWLEDGEMENTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

@app.get("/api/student/{sid}/acknowledge")
async def api_ack_get(sid: str):
    acks = _load_acknowledgements()
    return {"acknowledgements": [a for a in acks if a["student_id"] == sid]}

@app.post("/api/student/{sid}/acknowledge")
async def api_ack_post(sid: str, request: Request):
    if not MANAGER.get_twin(sid):
        return JSONResponse({"error": "Student not found"}, status_code=404)
    data = await request.json()
    category = (data.get("category") or "").strip()
    status = data.get("status", "trying")
    if not category:
        return JSONResponse({"error": "category required"}, status_code=400)
    acks = _load_acknowledgements()
    # Upsert by (student_id, category)
    existing = next((a for a in acks if a["student_id"] == sid and a["category"] == category), None)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    if existing:
        existing["status"] = status
        existing["timestamp"] = now
    else:
        acks.append({
            "id": str(uuid.uuid4())[:8],
            "student_id": sid,
            "category": category,
            "status": status,
            "timestamp": now,
        })
    _save_acknowledgements(acks)
    return {"message": "Saved"}


# ===================================================================
#  API — Admin analytics (#7)
# ===================================================================

@app.get("/api/admin/analytics")
async def api_admin_analytics():
    summaries = MANAGER.get_all_summaries()
    probs = [s["risk_prob"] for s in summaries]
    at_risk = [s for s in summaries if s["risk_label"] == 1]

    interventions = _load_interventions()
    itype_counts = {"none": 0, "mentoring": 0, "financial_aid": 0}
    for i in interventions:
        itype_counts[i.get("intervention_type", "none")] = itype_counts.get(i.get("intervention_type","none"), 0) + 1

    connect = _load_connect()
    outcome_counts = {}
    for s in summaries:
        t = s.get("actual_target", "Unknown")
        outcome_counts[t] = outcome_counts.get(t, 0) + 1

    top5 = sorted(summaries, key=lambda x: x["risk_prob"], reverse=True)[:5]

    return {
        "total_students": len(summaries),
        "at_risk_count": len(at_risk),
        "safe_count": len(summaries) - len(at_risk),
        "outcome_counts": outcome_counts,
        "avg_risk": round(sum(probs) / len(probs), 4) if probs else 0,
        "high_risk_count": sum(1 for p in probs if p > 0.6),
        "medium_risk_count": sum(1 for p in probs if 0.3 <= p <= 0.6),
        "low_risk_count": sum(1 for p in probs if p < 0.3),
        "total_messages": len(connect.get("messages", [])),
        "total_meetings": len(connect.get("meetings", [])),
        "total_notes": len(connect.get("notes", [])),
        "total_interventions": len(interventions),
        "intervention_breakdown": itype_counts,
        "top_at_risk": [{"student_id": s["student_id"], "risk_prob": s["risk_prob"]} for s in top5],
    }


# ===================================================================
#  API — Fairness thresholds (#5)
# ===================================================================

@app.get("/api/fairness/thresholds")
async def api_fairness_thresholds():
    path = f"{MODELS_DIR}/fairness_thresholds.json"
    if not os.path.exists(path):
        return {"available": False, "message": "Run pipelines/ml_pipeline.py to generate fairness thresholds."}
    with open(path) as f:
        data = json.load(f)
    data["available"] = True
    return data


@app.get("/api/rl/evaluation")
async def api_rl_evaluation():
    """Return the rigorous RL evaluation results (for researcher UI)."""
    path = f"{MODELS_DIR}/rl_evaluation_rigorous.json"
    if not os.path.exists(path):
        return NumpyJSONResponse(
            {"error": "Rigorous evaluation not run yet. Run `python pipelines/evaluate_rl_rigorous.py`."},
            status_code=404,
        )
    with open(path, "r") as f:
        return json.load(f)


# ===================================================================
#  API — Admin Panel (system management only)
# ===================================================================

@app.get("/api/admin/dataset")
async def api_admin_dataset():
    """View dataset info and first rows."""
    if DF is None:
        return {"error": "No dataset loaded"}
    return NumpyJSONResponse({
        "rows": len(DF),
        "columns": len(DF.columns),
        "column_names": list(DF.columns),
        "target_distribution": DF["Target"].value_counts().to_dict() if "Target" in DF.columns else {},
        "sample": DF.head(10).to_dict(orient="records"),
    })


@app.post("/api/admin/dataset/reset")
async def api_admin_dataset_reset():
    """Reset CRUD changes (remove all manual edits)."""
    if os.path.exists(CHANGES_FILE):
        os.remove(CHANGES_FILE)
    return {"message": "CRUD changes reset. Restart server to reload original data."}


@app.get("/api/admin/logs")
async def api_admin_logs():
    """View CRUD change log."""
    if not os.path.exists(CHANGES_FILE):
        return {"added": [], "deleted": [], "updated": [], "total_changes": 0}
    with open(CHANGES_FILE, "r") as f:
        changes = json.load(f)
    changes["total_changes"] = (
        len(changes.get("added", [])) +
        len(changes.get("deleted", [])) +
        len(changes.get("updated", []))
    )
    return changes


@app.get("/api/admin/models")
async def api_admin_models():
    """List all model artifacts with file sizes."""
    models_info = []
    for fname in os.listdir(MODELS_DIR):
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.isfile(fpath):
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
            models_info.append({"file": fname, "size_kb": size_kb})
    models_info.sort(key=lambda x: x["file"])
    return {"models_dir": MODELS_DIR, "files": models_info, "total_files": len(models_info)}


@app.post("/api/admin/retrain")
async def api_admin_retrain():
    """Trigger model retraining (runs the pipeline)."""
    import subprocess
    try:
        result = subprocess.run(
            ["python", "pipelines/ml_pipeline.py"],
            capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "message": "Retraining complete. Restart server to load new models." if result.returncode == 0 else "Retraining failed.",
        }
    except subprocess.TimeoutExpired:
        return NumpyJSONResponse({"error": "Retraining timed out (5 min limit)"}, status_code=500)
    except Exception as e:
        return NumpyJSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/admin/system")
async def api_admin_system():
    """System overview for admin."""
    total_twins = len(MANAGER.twins)
    users = get_all_users()
    return {
        "total_students": total_twins,
        "total_users": len(users),
        "users_by_role": {},
        "models_loaded": {
            "xgb_stage1": XGB_S1 is not None,
            "xgb_stage2": XGB_S2 is not None,
            "shap_s1": EXPLAINER_S1 is not None,
            "shap_s2": EXPLAINER_S2 is not None,
            "sem2_predictor": SEM2_PREDICTOR is not None,
            "learned_effects": LEARNED_EFFECTS is not None,
        },
        "data_driven": LEARNED_EFFECTS is not None,
        "users": users,
    }


# ===================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
