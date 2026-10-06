"""
=============================================================================
 FILE: services/auth.py
=============================================================================
 Simple role-based access control for the Digital Twin system.

 Roles:
   - student:    View own risk, recommendations, progress
   - counselor:  View all students, SHAP, simulation, interventions
   - admin:      Full access (CRUD, retrain, manage)
   - researcher: Model metrics, SHAP global, feature importance, ROC
=============================================================================
"""

import json, os, hashlib, secrets
from functools import wraps
from fastapi import Request
from fastapi.responses import JSONResponse


# ── Role definitions: which endpoints each role can access ──
ROLE_PERMISSIONS = {
    "student": {
        "allowed_prefixes": [
            "/api/student/{own_id}",  # own data only (checked at runtime)
            "/api/connect",           # read own messages/meetings (checked at runtime)
        ],
        "allowed_exact": [
            "/api/dashboard",
        ],
        "description": "View own risk score, recommendations, and progress",
    },
    "counselor": {
        "allowed_prefixes": [
            "/api/student",
            "/api/dashboard",
            "/api/at-risk",
            "/api/importance",
            "/api/rl",
            "/api/connect",
            "/api/fairness",
        ],
        "description": "View all students, SHAP, simulations, interventions, CRUD, RL recommendations",
    },
    "admin": {
        "allowed_prefixes": [
            "/api/auth",
            "/api/admin",
        ],
        "description": "User management, data management, model retraining, system logs",
    },
    "researcher": {
        "allowed_prefixes": [
            "/api/metrics",
            "/api/importance",
            "/api/roc",
            "/api/dashboard",
            "/api/rl",
            "/api/fairness",
        ],
        "description": "Model performance, SHAP global analysis, RL evaluation, validation outputs",
    },
}

# ── User store (JSON file) ──
USERS_FILE = "data/users.json"

DEFAULT_USERS = {
    "admin": {"password_hash": None, "role": "admin", "name": "System Admin", "student_id": None},
    "counselor": {"password_hash": None, "role": "counselor", "name": "Dr. Smith", "student_id": None},
    "researcher": {"password_hash": None, "role": "researcher", "name": "Researcher", "student_id": None},
    "student1": {"password_hash": None, "role": "student", "name": "Student Demo", "student_id": "STU0001"},
}


def _hash(password):
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    # Initialize with defaults (password = username for demo)
    users = {}
    for username, info in DEFAULT_USERS.items():
        users[username] = {**info, "password_hash": _hash(username)}
    _save_users(users)
    return users


def _save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ── Session store (in-memory, token → user) ──
SESSIONS = {}


def authenticate(username, password):
    """Verify credentials, return user info or None."""
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if user["password_hash"] != _hash(password):
        return None
    return {"username": username, "role": user["role"], "name": user["name"], "student_id": user.get("student_id")}


def create_session(user_info):
    """Create a session token for an authenticated user."""
    token = secrets.token_hex(16)
    SESSIONS[token] = user_info
    return token


def get_session(request: Request):
    """Extract session from request (cookie or header)."""
    # Check Authorization header first
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return SESSIONS.get(token)
    # Check cookie
    token = request.cookies.get("session_token")
    if token:
        return SESSIONS.get(token)
    return None


def destroy_session(request: Request):
    """Remove session."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        SESSIONS.pop(token, None)
    token = request.cookies.get("session_token")
    if token:
        SESSIONS.pop(token, None)


def check_permission(session, method, path, sid=None):
    """
    Check if a session's role has permission for the given endpoint.
    Returns (allowed: bool, reason: str).
    """
    if not session:
        return False, "Not authenticated"

    role = session["role"]
    perms = ROLE_PERMISSIONS.get(role)
    if not perms:
        return False, "Unknown role"

    # Student: can only access own data + dashboard
    if role == "student":
        own_id = session.get("student_id")
        if path == "/api/dashboard":
            return True, "OK"
        if sid and own_id and sid == own_id:
            return True, "Own data"
        if sid and own_id and sid != own_id:
            return False, "Students can only access their own data"
        if "/api/student" in path and not sid:
            return False, "Students cannot list all students"
        return False, "Access denied for student role"

    # Counselor: full student analytics + CRUD
    if role == "counselor":
        for prefix in perms.get("allowed_prefixes", []):
            if path.startswith(prefix):
                return True, "OK"
        return False, "Access denied for counselor role"

    # Admin: system management only (auth + admin endpoints), NO analytics
    if role == "admin":
        for prefix in perms.get("allowed_prefixes", []):
            if path.startswith(prefix):
                return True, "OK"
        return False, "Admin role is restricted to system management (users, data, models, logs)"

    # Researcher: model metrics and analysis only
    if role == "researcher":
        for prefix in perms.get("allowed_prefixes", []):
            if path.startswith(prefix):
                return True, "OK"
        return False, "Researchers can only access model metrics and analysis"

    return False, "Access denied"


def get_all_users():
    """Return user list (without password hashes) for admin."""
    users = _load_users()
    return {u: {"role": v["role"], "name": v["name"], "student_id": v.get("student_id")}
            for u, v in users.items()}


def create_user(username, password, role, name, student_id=None):
    """Create a new user (admin only)."""
    users = _load_users()
    if username in users:
        return False, "User already exists"
    if role not in ROLE_PERMISSIONS:
        return False, f"Invalid role: {role}"
    users[username] = {
        "password_hash": _hash(password),
        "role": role,
        "name": name,
        "student_id": student_id,
    }
    _save_users(users)
    return True, "User created"


def delete_user(username):
    """Delete a user (admin only)."""
    users = _load_users()
    if username not in users:
        return False, "User not found"
    if username == "admin":
        return False, "Cannot delete admin"
    del users[username]
    _save_users(users)
    return True, "User deleted"
