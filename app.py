"""Customer Segmentation — Streamlit inference app.

Loads the trained K-Means model + scaler, takes a customer profile via UI,
returns the predicted segment with persona description and marketing action.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------- Page config ----------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide",
)


# ---------- Constants ----------
# Order MUST match the order used in training (Section 8 of the notebook)
FEATURES = [
    "Age", "Income", "Total_spends", "Total_children", "Customer_Since",
    "Recency", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth",
    "Education_enc", "HasPartner", "AcceptedAny",
]

EDUCATION_LABELS = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]


# ---------- Model loading ----------
@st.cache_resource
def load_artifacts():
    """Find scaler and model in either ./models/ or ./ ."""
    base = Path(__file__).parent
    for d in [base / "models", base]:
        scaler_p, model_p = d / "scaler.pkl", d / "model_kmeans.pkl"
        if scaler_p.exists() and model_p.exists():
            return joblib.load(scaler_p), joblib.load(model_p)
    raise FileNotFoundError(
        "scaler.pkl / model_kmeans.pkl not found. Run the notebook first."
    )


@st.cache_resource
def derive_personas(_scaler, _model):
    """Inverse-transform centroids and label clusters by spend level."""
    centroids = _scaler.inverse_transform(_model.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=FEATURES)

    spend_idx = FEATURES.index("Total_spends")
    high_cluster = int(np.argmax(centroids[:, spend_idx]))

    personas = {
        high_cluster: {
            "name": "High-Engagement Premium",
            "emoji": "🎯",
            "color": "#1B5E20",
            "bg": "#E8F5E9",
            "description": (
                "Higher income, substantially higher spend across all categories, "
                "fewer dependents, prefers in-store and catalog purchases, more "
                "responsive to past campaigns."
            ),
            "action": (
                "Premium product targeting, loyalty tier upgrades, exclusive in-store "
                "events, personalised catalogue mailers. Avoid heavy-discount messaging "
                "— this segment is not price-sensitive."
            ),
        },
        1 - high_cluster: {
            "name": "Budget-Conscious Family",
            "emoji": "💡",
            "color": "#E65100",
            "bg": "#FFF3E0",
            "description": (
                "Lower income, lower total spend, more children at home, high web "
                "visits relative to web purchases (browsing without converting), "
                "lower campaign acceptance."
            ),
            "action": (
                "Family bundles, multi-buy discounts, abandoned-cart re-engagement, "
                "first-purchase incentives. Channel mix should favour deal-driven "
                "email and web push."
            ),
        },
    }
    return personas, centroids_df


# ---------- Load model ----------
try:
    scaler, model = load_artifacts()
    personas, centroids_df = derive_personas(scaler, model)
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error(str(e))
    st.stop()


# ---------- Header ----------
st.title("🛒 Customer Segmentation")
st.caption(
    "K-Means (k=2) trained on the Customer Personality Analysis dataset "
    "(2,201 customers, 12 engineered features). Inputs below are scored "
    "live against the model."
)

# Sidebar — methodology summary
with st.sidebar:
    st.header("About this model")
    st.markdown(
        """
        **Algorithm:** K-Means, k=2  
        **Selected by:** highest Silhouette across K-Means, Hierarchical, DBSCAN  
        **Validation metrics:** Silhouette ↑, Davies–Bouldin ↓, Dunn ↑  
        **Coverage:** 100% (vs DBSCAN's 18.5%)
        """
    )
    st.markdown("---")
    st.markdown(
        "**Why two clusters?** Every internal-validation metric "
        "monotonically degrades as k increases. The data has one dominant "
        "axis (Income → Total_spends), giving a clean engagement binary."
    )
    st.markdown("---")
    st.markdown(
        "[📓 Notebook](customer_segmentation.ipynb) · "
        "[🐙 GitHub](https://github.com/MainakJ99/customer-segmentation)"
    )


# ---------- Preset profiles ----------
st.subheader("Pick a preset, or build a profile below")

PRESETS = {
    "👔 Affluent professional": {
        "Age": 50, "Income": 85000, "Total_spends": 1500, "Total_children": 0,
        "Customer_Since": 4700, "Recency": 20,
        "NumWebPurchases": 8, "NumStorePurchases": 10, "NumWebVisitsMonth": 4,
        "Education_enc": 4, "HasPartner": 1, "AcceptedAny": 1,
    },
    "👨‍👩‍👧 Young family": {
        "Age": 35, "Income": 35000, "Total_spends": 80, "Total_children": 2,
        "Customer_Since": 4500, "Recency": 50,
        "NumWebPurchases": 2, "NumStorePurchases": 3, "NumWebVisitsMonth": 8,
        "Education_enc": 2, "HasPartner": 1, "AcceptedAny": 0,
    },
    "🛍️ Window-shopper": {
        "Age": 28, "Income": 28000, "Total_spends": 50, "Total_children": 0,
        "Customer_Since": 4400, "Recency": 80,
        "NumWebPurchases": 1, "NumStorePurchases": 1, "NumWebVisitsMonth": 12,
        "Education_enc": 2, "HasPartner": 0, "AcceptedAny": 0,
    },
}

if "active" not in st.session_state:
    st.session_state.active = PRESETS["👔 Affluent professional"].copy()

cols = st.columns(len(PRESETS))
for col, (label, preset) in zip(cols, PRESETS.items()):
    if col.button(label, use_container_width=True):
        st.session_state.active = preset.copy()


# ---------- Inputs ----------
st.subheader("Customer profile")
v = st.session_state.active

left, right = st.columns(2)

with left:
    age = st.slider("Age", 25, 90, v["Age"])
    income = st.slider("Annual income (USD)", 0, 150_000, v["Income"], step=1_000)
    total_spends = st.slider(
        "Total spend, last 2 years (USD)", 0, 3_000, v["Total_spends"], step=10
    )
    children = st.slider("Total children at home", 0, 3, v["Total_children"])
    customer_since = st.slider(
        "Customer tenure (days enrolled)", 4_000, 5_500, v["Customer_Since"]
    )
    recency = st.slider("Days since last purchase", 0, 100, v["Recency"])

with right:
    web_purch = st.slider("Number of web purchases", 0, 30, v["NumWebPurchases"])
    store_purch = st.slider("Number of store purchases", 0, 15, v["NumStorePurchases"])
    web_visits = st.slider(
        "Web visits per month", 0, 20, v["NumWebVisitsMonth"]
    )
    education = st.selectbox(
        "Education",
        options=list(range(5)),
        format_func=lambda x: EDUCATION_LABELS[x],
        index=v["Education_enc"],
    )
    has_partner = st.radio(
        "Has partner", ["No", "Yes"], index=v["HasPartner"], horizontal=True
    )
    accepted = st.checkbox(
        "Has accepted at least one past campaign", value=bool(v["AcceptedAny"])
    )


# ---------- Predict ----------
profile = np.array([[
    age, income, total_spends, children, customer_since, recency,
    web_purch, store_purch, web_visits, education,
    1 if has_partner == "Yes" else 0, int(accepted),
]])

cluster = int(model.predict(scaler.transform(profile))[0])
persona = personas[cluster]


# ---------- Output ----------
st.markdown("---")

result_col, action_col = st.columns([1, 1])

with result_col:
    st.markdown(
        f"""
        <div style="background-color:{persona['bg']};
                    padding:24px; border-radius:8px;
                    border-left:6px solid {persona['color']};">
            <div style="font-size:14px; color:#666; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:8px;">
                Predicted segment
            </div>
            <div style="font-size:28px; font-weight:600; color:{persona['color']};">
                {persona['emoji']} {persona['name']}
            </div>
            <div style="margin-top:12px; color:#333; line-height:1.6;">
                {persona['description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with action_col:
    st.markdown(
        f"""
        <div style="background-color:#F5F5F5;
                    padding:24px; border-radius:8px;
                    border-left:6px solid #1565C0;">
            <div style="font-size:14px; color:#666; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:8px;">
                Recommended action
            </div>
            <div style="margin-top:12px; color:#222; line-height:1.7; font-size:15px;">
                {persona['action']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Centroid comparison ----------
with st.expander("📊 How the two segments differ (cluster centroids on original scale)"):
    display = centroids_df.copy()
    display.index = [
        f"{personas[i]['emoji']} {personas[i]['name']}"
        for i in display.index
    ]
    st.dataframe(
        display.T.style.format("{:,.1f}").background_gradient(axis=1, cmap="RdYlGn"),
        use_container_width=True,
    )
    st.caption(
        "Each row is a feature; columns are the two cluster centroids. Green = "
        "higher value, red = lower. The contrast across `Income`, `Total_spends`, "
        "`AcceptedAny`, and `NumStorePurchases` is what defines the split."
    )
