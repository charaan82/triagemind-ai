import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TriageMind AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# TAB CREATION
# ============================================================

dashboard_tab, triage_tab, network_tab, mitre_tab, soc_tab, diagnostics_tab = st.tabs(
    [
        "🏠 Dashboard",
        "🚨 Alert Triage",
        "📊 Network Detection",
        "🎯 MITRE ATT&CK",
        "🛡️ SOC Investigation",
        "⚙️ Diagnostics"
    ]
)

# ============================================================
# BASIC PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
MITRE_DIR = os.path.join(BASE_DIR, "mitre_data")

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "classifier.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL = None
FEATURE_COLUMNS = []

try:
    if os.path.exists(CLASSIFIER_PATH):
        MODEL = joblib.load(CLASSIFIER_PATH)

    if os.path.exists(FEATURE_PATH):
        FEATURE_COLUMNS = joblib.load(FEATURE_PATH)

except Exception as e:
    st.warning(f"Model loading issue: {e}")

# ============================================================
# LOAD MITRE DATA
# ============================================================

MITRE_TECHNIQUES = []

MITRE_FILES = [
    "enterprise-attack-19.2.json",
    "enterprise-attack.json",
    "mitre_techniques.json"
]

for filename in MITRE_FILES:

    path = os.path.join(
        MITRE_DIR,
        filename
    )

    if os.path.exists(path):

        try:

            if filename.endswith(".json"):

                import json

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    mitre_data = json.load(f)

                if isinstance(mitre_data, dict):

                    if "objects" in mitre_data:

                        MITRE_TECHNIQUES = mitre_data["objects"]

                    elif "techniques" in mitre_data:

                        MITRE_TECHNIQUES = mitre_data["techniques"]

                    else:

                        MITRE_TECHNIQUES = []

                elif isinstance(mitre_data, list):

                    MITRE_TECHNIQUES = mitre_data

                break

        except Exception:
            continue

# ============================================================
# OPTIONAL FAISS PATH
# ============================================================

FAISS_PATH = None

possible_faiss = [
    os.path.join(
        MITRE_DIR,
        "mitre_index.faiss"
    ),
    os.path.join(
        BASE_DIR,
        "faiss_index",
        "mitre_index.faiss"
    ),
    os.path.join(
        BASE_DIR,
        "faiss_index",
        "index.faiss"
    )
]

for path in possible_faiss:

    if os.path.exists(path):

        FAISS_PATH = path
        break

# ============================================================
# FEATURE IMPORTANCE HELPER
# ============================================================

def get_feature_importance():

    try:

        if MODEL is None:
            return None

        if not hasattr(
            MODEL,
            "feature_importances_"
        ):
            return None

        importance = MODEL.feature_importances_

        if FEATURE_COLUMNS:

            names = FEATURE_COLUMNS

        else:

            names = [
                f"Feature {i + 1}"
                for i in range(len(importance))
            ]

        n = min(
            len(names),
            len(importance)
        )

        df = pd.DataFrame(
            {
                "Feature": names[:n],
                "Importance": importance[:n]
            }
        )

        return df.sort_values(
            "Importance",
            ascending=False
        )

    except Exception:
        return None


# ============================================================
# DASHBOARD
# ============================================================

with dashboard_tab:

    st.markdown(
        """
        <div style="
            padding:30px;
            border-radius:20px;
            background:linear-gradient(
                135deg,
                #0f172a,
                #172554,
                #111827
            );
            border:1px solid #334155;
            margin-bottom:25px;
        ">

        <div style="
            color:#60a5fa;
            font-weight:600;
            letter-spacing:1px;
        ">
        SECURITY OPERATIONS CENTER
        </div>

        <h1 style="
            color:white;
            font-size:42px;
            margin:8px 0;
        ">
        🛡️ TriageMind AI
        </h1>

        <p style="
            color:#cbd5e1;
            font-size:18px;
        ">
        Intelligent Cybersecurity Alert Triage Platform
        </p>

        <p style="
            color:#94a3b8;
        ">
        Machine Learning • MITRE ATT&CK •
        Explainable AI • SOC Investigation
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🟢 System Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        if MODEL is not None:

            st.success("ML ENGINE ONLINE")

            st.metric(
                "Classifier",
                type(MODEL).__name__
            )

        else:

            st.error("ML ENGINE OFFLINE")

            st.metric(
                "Classifier",
                "Unavailable"
            )

    with c2:

        if MITRE_TECHNIQUES:

            st.success("MITRE ONLINE")

            st.metric(
                "Techniques",
                len(MITRE_TECHNIQUES)
            )

        else:

            st.warning("MITRE DATA NOT FOUND")

            st.metric(
                "Techniques",
                0
            )

    with c3:

        st.success("FEATURES READY")

        st.metric(
            "Features",
            len(FEATURE_COLUMNS)
        )

    with c4:

        st.info("CASE HISTORY")

        st.metric(
            "Investigations",
            len(st.session_state.history)
        )

    st.divider()

    st.subheader("📊 Security Operations Summary")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        if "Risk Score" in history_df.columns:

            risk = pd.to_numeric(
                history_df["Risk Score"],
                errors="coerce"
            ).fillna(0)

            avg_risk = risk.mean()

        else:

            avg_risk = 0

        if "Severity" in history_df.columns:

            high_count = history_df[
                history_df["Severity"].isin(
                    ["HIGH", "CRITICAL"]
                )
            ].shape[0]

            latest = history_df.iloc[-1][
                "Severity"
            ]

        else:

            high_count = 0
            latest = "N/A"

    else:

        avg_risk = 0
        high_count = 0
        latest = "No alerts"

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Investigations",
            len(st.session_state.history)
        )

    with b:
        st.metric(
            "Average Risk",
            f"{avg_risk:.1f}/100"
        )

    with c:
        st.metric(
            "High / Critical",
            high_count
        )

    with d:
        st.metric(
            "Latest Severity",
            latest
        )

    st.divider()

    st.subheader("🚀 Security Workflow")

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.markdown(
            """
            ### 01 🚨 Detect

            Upload network traffic or
            provide a security alert.
            """
        )

    with w2:
        st.markdown(
            """
            ### 02 🔎 Triage

            Analyze suspicious activity
            using the ML classifier.
            """
        )

    with w3:
        st.markdown(
            """
            ### 03 🎯 Map

            Connect observed behavior
            with MITRE ATT&CK.
            """
        )

    with w4:
        st.markdown(
            """
            ### 04 🛡️ Investigate

            Review evidence and generate
            a SOC investigation report.
            """
        )

    st.divider()

    st.subheader("🏗️ TriageMind Architecture")

    st.code(
        """
SECURITY ALERT
      │
      ▼
┌─────────────────┐
│  TRIAGEMIND AI  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 ML ENGINE  MITRE ATT&CK
    │         │
    ▼         ▼
Attack      Technique
Detection   Mapping
    │         │
    └────┬────┘
         ▼
    RISK ENGINE
         │
         ▼
 SOC INVESTIGATION
         │
    ┌────┴────┐
    ▼         ▼
 REPORT    ANALYTICS
        """,
        language="text"
    )

    st.divider()

    st.subheader("🧠 Platform Capabilities")

    p1, p2, p3 = st.columns(3)

    with p1:

        st.markdown(
            """
            ### 🤖 Machine Learning

            • Network intrusion detection  
            • Attack classification  
            • Confidence analysis  
            • Feature importance  
            • Model evaluation
            """
        )

    with p2:

        st.markdown(
            """
            ### 🎯 Threat Intelligence

            • MITRE ATT&CK  
            • Technique identification  
            • Tactic mapping  
            • Alert enrichment  
            • Threat context
            """
        )

    with p3:

        st.markdown(
            """
            ### 🛡️ SOC Operations

            • Alert triage  
            • Risk scoring  
            • Investigation workflow  
            • Security reports  
            • Investigation history
            """
        )

    st.divider()

    st.subheader("🔄 Detection Pipeline")

    pipeline = pd.DataFrame(
        {
            "Stage": [
                "1. Input",
                "2. ML Analysis",
                "3. MITRE Mapping",
                "4. Risk Analysis",
                "5. Investigation",
                "6. Reporting"
            ],
            "Purpose": [
                "Security alert or network CSV",
                "Classify suspicious activity",
                "Identify ATT&CK techniques",
                "Calculate security risk",
                "Guide analyst investigation",
                "Generate security report"
            ]
        }
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📋 Recent Investigations")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No investigations yet. "
            "Start from the Alert Triage or "
            "Network Detection tab."
        )

    st.divider()

    st.subheader("🤖 Model Information")

    m1, m2 = st.columns(2)

    with m1:

        if MODEL is not None:

            st.success(
                "Machine-learning model loaded successfully."
            )

            st.write(
                "**Model:**",
                type(MODEL).__name__
            )

            st.write(
                "**Input Features:**",
                len(FEATURE_COLUMNS)
            )

        else:

            st.error(
                "Machine-learning model could not be loaded."
            )

    with m2:

        importance_df = get_feature_importance()

        if importance_df is not None:

            st.write("**Top Model Features**")

            chart_df = (
                importance_df
                .head(10)
                .set_index("Feature")
            )

            st.bar_chart(
                chart_df
            )

        else:

            st.info(
                "Feature importance is unavailable."
            )

    st.divider()

    st.subheader("🎯 MITRE ATT&CK Status")

    if MITRE_TECHNIQUES:

        st.success(
            "MITRE ATT&CK knowledge base loaded."
        )

        st.metric(
            "Available Techniques",
            len(MITRE_TECHNIQUES)
        )

    else:

        st.warning(
            "MITRE ATT&CK data was not detected."
        )

    st.divider()

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:20px;
            color:#94a3b8;
        ">

        <strong style="color:#60a5fa;">
        TriageMind AI
        </strong>

        <br>

        Intelligent Cybersecurity Alert Triage

        <br><br>

        Machine Learning • MITRE ATT&CK •
        Explainable AI • SOC Automation

        </div>
        """,
        unsafe_allow_html=True
    )
