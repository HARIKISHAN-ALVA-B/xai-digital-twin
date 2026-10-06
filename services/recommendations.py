"""
=============================================================================
 FILE: services/recommendations.py
=============================================================================
 Category-based counseling recommendation engine.

 Uses SHAP contributions to:
   1. Map features into counseling-relevant categories
   2. Aggregate per-category SHAP scores (sum of positive SHAP = risk drivers)
   3. Rank categories by impact
   4. Generate structured, explainable recommendations
=============================================================================
"""


# ── Feature-to-Category mapping ──
# Each category contains feature substrings. A feature is matched to the
# FIRST category whose any pattern appears in the feature name (case-insensitive).
# This avoids hardcoding every feature individually.

CATEGORY_PATTERNS = {
    "academic": [
        "grade", "approved", "credited", "approval_rate", "failure_rate",
        "grade_change", "approved_change",
    ],
    "financial": [
        "debtor", "tuition", "scholarship",
    ],
    "engagement": [
        "without evaluations", "enrolled", "evaluations", "daytime",
        "attendance",
    ],
    "stress": [
        "stress",
    ],
    "background": [
        "age", "gender", "marital", "displaced", "international",
        "nacionality", "mother", "father", "previous qualification",
        "application", "course",
    ],
    "macro": [
        "unemployment", "inflation", "gdp",
    ],
}

# ── Per-category recommendation templates ──
# Each has a primary recommendation and a detail line.
# Selected only when the category is a top risk driver.

CATEGORY_RECOMMENDATIONS = {
    "academic": {
        "title": "Academic Support",
        "recommendation": "Provide academic mentoring and tutoring to improve course performance.",
        "actions": [
            "Assign a faculty mentor for regular academic check-ins",
            "Enroll in supplementary tutoring for struggling subjects",
            "Create a structured study plan targeting failed or low-grade units",
        ],
    },
    "financial": {
        "title": "Financial Assistance",
        "recommendation": "Address financial barriers that may lead to dropout.",
        "actions": [
            "Evaluate eligibility for scholarships or fee waivers",
            "Connect with financial aid office for payment plan options",
            "Refer to emergency funding programs if tuition is overdue",
        ],
    },
    "engagement": {
        "title": "Engagement & Attendance",
        "recommendation": "Improve student engagement and participation in coursework.",
        "actions": [
            "Investigate reasons for missed evaluations or low enrollment",
            "Schedule regular check-ins to monitor attendance patterns",
            "Consider peer study groups to increase academic involvement",
        ],
    },
    "stress": {
        "title": "Stress & Wellbeing",
        "recommendation": "Offer stress counseling and time management support.",
        "actions": [
            "Refer to campus counseling services for stress management",
            "Provide workshops on time management and study techniques",
            "Monitor workload balance across enrolled units",
        ],
    },
    "background": {
        "title": "Personal & Demographic Support",
        "recommendation": "Consider personal circumstances that may affect academic success.",
        "actions": [
            "Assess if personal factors require accommodation or support",
            "Connect with student services for displaced or international students",
            "Review if course placement aligns with student background",
        ],
    },
    "macro": {
        "title": "External Factors",
        "recommendation": "External economic conditions are contributing to risk.",
        "actions": [
            "Acknowledge that this factor is outside institutional control",
            "Focus mitigation efforts on academic and financial support",
            "Monitor economic indicators that affect student retention",
        ],
    },
}


def classify_feature(feature_name):
    """Map a feature name to its category using pattern matching."""
    name_lower = feature_name.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower:
                return category
    return "background"  # fallback


def generate_recommendations(shap_contributions):
    """
    Generate structured counseling recommendations from SHAP contributions.

    Args:
        shap_contributions: dict of {feature_name: {"shap": float, "value": float, "direction": str}}
                            (as returned by the /predict endpoint)

    Returns:
        dict with category_scores, top_categories, and recommendations
    """
    # Step 1: Aggregate SHAP values by category
    # Only consider features that INCREASE risk (positive SHAP)
    category_scores = {}
    category_features = {}

    for feature, info in shap_contributions.items():
        cat = classify_feature(feature)
        shap_val = info["shap"]

        if cat not in category_scores:
            category_scores[cat] = 0.0
            category_features[cat] = []

        # Sum positive SHAP values (risk-increasing factors)
        if shap_val > 0:
            category_scores[cat] += shap_val
            category_features[cat].append({
                "feature": feature,
                "shap": shap_val,
                "value": info["value"],
            })

    # Step 2: Rank categories by total positive SHAP impact
    ranked = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Select top categories (those with meaningful positive SHAP)
    top_categories = []
    for cat, score in ranked:
        if score > 0.01:  # only include categories with real impact
            top_categories.append({
                "category": cat,
                "score": round(score, 4),
                "contributing_features": sorted(
                    category_features.get(cat, []),
                    key=lambda x: x["shap"], reverse=True
                )[:3],  # top 3 features per category
            })
        if len(top_categories) >= 3:
            break

    # Step 4: Generate recommendations for top categories
    recommendations = []
    for tc in top_categories:
        cat = tc["category"]
        rec_template = CATEGORY_RECOMMENDATIONS.get(cat, CATEGORY_RECOMMENDATIONS["background"])
        recommendations.append({
            "category": cat,
            "title": rec_template["title"],
            "recommendation": rec_template["recommendation"],
            "actions": rec_template["actions"],
            "impact_score": tc["score"],
            "key_drivers": [f["feature"] for f in tc["contributing_features"]],
        })

    # Build summary sentence
    if recommendations:
        top_names = [r["title"] for r in recommendations]
        if len(top_names) == 1:
            summary = f"Primary concern: {top_names[0]}."
        else:
            summary = f"Key risk areas: {', '.join(top_names[:-1])} and {top_names[-1]}."
    else:
        summary = "No significant risk factors identified."

    return {
        "summary": summary,
        "category_scores": {cat: round(score, 4) for cat, score in ranked if score > 0},
        "top_categories": top_categories,
        "recommendations": recommendations,
    }
