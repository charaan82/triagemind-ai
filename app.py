# ============================================================
# TRIAGEMIND AI
# Intelligent Cybersecurity Alert Triage Platform
# ============================================================

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TriageMind AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

APP_NAME = "TriageMind AI"
APP_VERSION = "2.0"

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "network_results" not in st.session_state:
    st.session_state.network_results = None

if "network_original" not in st.session_state:
    st.session_state.network_original = None

if "alert_results" not in st.session_state:
    st.session_state.alert_results = []


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b,
            #111827
        );
        border: 1px solid #334155;
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 17px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE SEARCH
# ============================================================

def find_file(file_names):

    names = {
        str(name).lower()
        for name in file_names
    }

    for root, dirs, files in os.walk(BASE_DIR):

        dirs[:] = [
            d for d in dirs
            if d not in {
                "__pycache__",
                ".git",
                "venv",
                ".venv",
            }
        ]

        for file_name in files:

            if file_name.lower() in names:

                return Path(root) / file_name

    return None


# ============================================================
# LOCATE FILES
# ============================================================

def locate_model():

    path = find_file(
        [
            "classifier.pkl",
            "model.pkl",
            "triagemind_model.pkl",
            "random_forest.pkl",
            "rf_model.pkl",
        ]
    )

    if path:
        return path

    for root, dirs, files in os.walk(BASE_DIR):

        dirs[:] = [
            d for d in dirs
            if d not in {
                "__pycache__",
                ".git",
                "venv",
                ".venv",
            }
        ]

        for file_name in files:

            if file_name.lower().endswith(".pkl"):

                if any(
                    word in file_name.lower()
                    for word in [
                        "feature",
                        "metric",
                        "confusion",
                    ]
                ):
                    continue

                return Path(root) / file_name

    return None


MODEL_PATH = locate_model()

FEATURE_PATH = find_file(
    [
        "feature_columns.pkl",
        "features.pkl",
        "model_features.pkl",
    ]
)

MITRE_PATH = find_file(
    [
        "mitre_techniques.json",
        "enterprise-attack.json",
        "enterprise-attack-19.2.json",
    ]
)

FAISS_PATH = find_file(
    [
        "mitre_index.faiss",
        "index.faiss",
        "mitre.faiss",
    ]
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(path):

    if path is None:
        return None, "Model file not found."

    try:

        try:

            model = joblib.load(
                path
            )

        except Exception:

            with open(
                path,
                "rb"
            ) as file:

                model = pickle.load(
                    file
                )

        return model, None

    except Exception as error:

        return None, str(error)


MODEL, MODEL_ERROR = load_model(
    MODEL_PATH
)


# ============================================================
# FEATURE COLUMN LOADING
# ============================================================

@st.cache_resource
def load_features(path):

    if path is None:
        return None, None

    try:

        with open(
            path,
            "rb"
        ) as file:

            features = pickle.load(
                file
            )

        if isinstance(
            features,
            np.ndarray
        ):
            features = features.tolist()

        if isinstance(
            features,
            tuple
        ):
            features = list(features)

        return features, None

    except Exception as error:

        return None, str(error)


FEATURE_COLUMNS, FEATURE_ERROR = load_features(
    FEATURE_PATH
)


# ============================================================
# MITRE DATA
# ============================================================

def normalize_mitre(data):

    objects = []

    if isinstance(data, list):

        objects = data

    elif isinstance(data, dict):

        if isinstance(
            data.get("techniques"),
            list
        ):

            objects = data["techniques"]

        elif isinstance(
            data.get("objects"),
            list
        ):

            for obj in data["objects"]:

                if obj.get(
                    "type"
                ) != "attack-pattern":

                    continue

                technique_id = ""

                for reference in obj.get(
                    "external_references",
                    []
                ):

                    if (
                        reference.get(
                            "source_name"
                        )
                        == "mitre-attack"
                    ):

                        technique_id = (
                            reference.get(
                                "external_id",
                                ""
                            )
                        )

                        break

                if technique_id:

                    objects.append(
                        {
                            "id": technique_id,
                            "name": obj.get(
                                "name",
                                ""
                            ),
                            "description": obj.get(
                                "description",
                                ""
                            ),
                            "tactics": [],
                        }
                    )

    techniques = []

    for item in objects:

        if not isinstance(
            item,
            dict
        ):
            continue

        technique_id = (
            item.get("id")
            or item.get("technique_id")
            or item.get("external_id")
            or ""
        )

        name = (
            item.get("name")
            or item.get("technique")
            or "Unknown Technique"
        )

        description = (
            item.get("description")
            or ""
        )

        tactics = (
            item.get("tactics")
            or item.get("tactic")
            or []
        )

        if isinstance(
            tactics,
            str
        ):

            tactics = [tactics]

        techniques.append(
            {
                "id": str(
                    technique_id
                ),
                "name": str(
                    name
                ),
                "description": str(
                    description
                ),
                "tactics": [
                    str(x)
                    for x in tactics
                ],
            }
        )

    return techniques


@st.cache_data
def load_mitre(path):

    if path is None:
        return [], "MITRE JSON file not found."

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        techniques = normalize_mitre(
            data
        )

        return techniques, None

    except Exception as error:

        return [], str(error)


MITRE_TECHNIQUES, MITRE_ERROR = load_mitre(
    MITRE_PATH
)


# ============================================================
# MITRE SEARCH ENGINE
# ============================================================

@st.cache_resource
def build_search_engine(techniques):

    if not techniques:
        return None, None

    documents = []

    for item in techniques:

        text = " ".join(
            [
                item["id"],
                item["name"],
                item["description"],
                " ".join(
                    item["tactics"]
                ),
            ]
        )

        documents.append(
            text
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    return vectorizer, matrix


def search_mitre(
    query,
    top_k=5
):

    if not query.strip():
        return []

    if not MITRE_TECHNIQUES:
        return []

    vectorizer, matrix = build_search_engine(
        MITRE_TECHNIQUES
    )

    query_vector = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        matrix
    )[0]

    indexes = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in indexes:

        result = MITRE_TECHNIQUES[
            int(index)
        ].copy()

        result["score"] = float(
            scores[index]
        )

        results.append(
            result
        )

    return results


# ============================================================
# TARGET COLUMNS
# ============================================================

TARGET_COLUMNS = [
    "label",
    "Label",
    "target",
    "Target",
    "class",
    "Class",
    "attack",
    "Attack",
    "attack_cat",
    "Attack_cat",
    "category",
    "Category",
]


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    dataframe
):

    df = dataframe.copy()

    for column in TARGET_COLUMNS:

        if column in df.columns:

            df = df.drop(
                columns=[column]
            )

    # --------------------------------------------------------
    # Use saved feature columns.
    # --------------------------------------------------------

    if FEATURE_COLUMNS:

        output = pd.DataFrame(
            index=df.index
        )

        for feature in FEATURE_COLUMNS:

            if feature in df.columns:

                series = df[feature]

                if pd.api.types.is_numeric_dtype(
                    series
                ):

                    output[feature] = pd.to_numeric(
                        series,
                        errors="coerce"
                    )

                else:

                    output[feature] = pd.factorize(
                        series.astype(str)
                    )[0]

            else:

                output[feature] = 0

        output = output.replace(
            [np.inf, -np.inf],
            np.nan
        )

        output = output.fillna(
            0
        )

        return output

    # --------------------------------------------------------
    # Automatic numeric fallback.
    # --------------------------------------------------------

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric = numeric.fillna(
        0
    )

    return numeric


# ============================================================
# PREDICTION
# ============================================================

def predict_network(
    dataframe
):

    if MODEL is None:

        raise RuntimeError(
            "ML model is not loaded."
        )

    features = prepare_features(
        dataframe
    )

    predictions = MODEL.predict(
        features
    )

    probabilities = None

    if hasattr(
        MODEL,
        "predict_proba"
    ):

        try:

            probabilities = MODEL.predict_proba(
                features
            )

        except Exception:

            probabilities = None

    return (
        predictions,
        probabilities,
        features
    )


# ============================================================
# ATTACK DETECTION
# ============================================================

def is_attack(value):

    text = str(
        value
    ).strip().lower()

    attack_values = {
        "1",
        "true",
        "yes",
        "attack",
        "anomaly",
        "malicious",
        "malware",
        "intrusion",
    }

    return text in attack_values


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(
    attack_rate,
    mitre_confidence,
    prediction_confidence
):

    score = (
        attack_rate * 0.55
        + mitre_confidence * 0.25
        + prediction_confidence * 0.20
    )

    return round(
        max(
            0,
            min(
                100,
                score
            )
        ),
        2
    )


def get_severity(
    score
):

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 35:
        return "MEDIUM"

    return "LOW"


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    if MODEL is None:
        return None

    if not hasattr(
        MODEL,
        "feature_importances_"
    ):
        return None

    importance = np.asarray(
        MODEL.feature_importances_
    )

    if FEATURE_COLUMNS:

        if len(importance) == len(
            FEATURE_COLUMNS
        ):

            result = pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "Importance": importance,
                }
            )

            return result.sort_values(
                "Importance",
                ascending=False
            )

    return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🛡️ TriageMind AI"
    )

    st.caption(
        "Cybersecurity Alert Triage Platform"
    )

    st.divider()

    st.subheader(
        "SYSTEM STATUS"
    )

    if MODEL is not None:

        st.success(
            "🟢 ML Model Ready"
        )

    else:

        st.error(
            "🔴 ML Model Missing"
        )

    if MITRE_TECHNIQUES:

        st.success(
            f"🟢 MITRE Ready "
            f"({len(MITRE_TECHNIQUES)})"
        )

    else:

        st.error(
            "🔴 MITRE Data Missing"
        )

    if FAISS_PATH:

        st.success(
            "🟢 FAISS Available"
        )

    else:

        st.info(
            "🔵 FAISS Optional"
        )

    st.divider()

    st.subheader(
        "TECHNOLOGY"
    )

    st.write(
        "Python"
    )

    st.write(
        "Scikit-learn"
    )

    st.write(
        "Pandas / NumPy"
    )

    st.write(
        "MITRE ATT&CK"
    )

    st.write(
        "TF-IDF Retrieval"
    )

    st.write(
        "Streamlit"
    )

    st.divider()

    st.caption(
        f"TriageMind AI v{APP_VERSION}"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🛡️ TriageMind AI</h1>

        <p>
        Intelligent Cybersecurity Alert Triage Platform
        </p>

        <p>
        Machine Learning • MITRE ATT&CK •
        Explainable AI • SOC Investigation
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

(
    dashboard_tab,
    alert_tab,
    network_tab,
    mitre_tab,
    soc_tab,
    diagnostics_tab,
) = st.tabs(
    [
        "🏠 Dashboard",
        "🚨 Alert Triage",
        "📊 Network Detection",
        "🎯 MITRE ATT&CK",
        "🛡️ SOC Investigation",
        "⚙️ Diagnostics",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with dashboard_tab:

    st.header(
        "Security Operations Dashboard"
    )

    st.write(
        "AI-assisted network detection, "
        "threat-intelligence enrichment and "
        "security alert triage."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "ML Model",
            "READY"
            if MODEL is not None
            else "OFFLINE"
        )

    with c2:

        st.metric(
            "MITRE Techniques",
            len(
                MITRE_TECHNIQUES
            )
        )

    with c3:

        st.metric(
            "Model Features",
            len(
                FEATURE_COLUMNS
            )
            if FEATURE_COLUMNS
            else "AUTO"
        )

    with c4:

        st.metric(
            "Investigations",
            len(
                st.session_state.history
            )
        )

    st.divider()

    st.subheader(
        "Platform Capabilities"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            ### 🤖 ML Detection

            • Network intrusion detection

            • Attack classification

            • Prediction confidence

            • Feature importance

            • Model evaluation
            """
        )

    with c2:

        st.markdown(
            """
            ### 🎯 Threat Intelligence

            • MITRE ATT&CK mapping

            • Technique search

            • Tactic identification

            • Alert enrichment

            • Threat context
            """
        )

    with c3:

        st.markdown(
            """
            ### 🛡️ SOC Operations

            • Alert triage

            • Risk scoring

            • Investigation workflow

            • Security reports

            • CSV export
            """
        )

    st.divider()

    st.subheader(
        "TriageMind Architecture"
    )

    st.code(
        """
SECURITY ALERT / NETWORK TRAFFIC
              |
              v
       +--------------+
       | TriageMind AI|
       +------+-------+
              |
       +------+------+
       |             |
       v             v
 ML CLASSIFIER   MITRE SEARCH
       |             |
       v             v
 ATTACK STATUS   TECHNIQUE
       |          MAPPING
       +------+- - -+
              |
              v
        RISK ENGINE
              |
              v
       SOC INVESTIGATION
              |
       +------+------+
       |             |
       v             v
    REPORT       ANALYTICS
        """,
        language="text",
    )

    st.divider()

    st.subheader(
        "Recent Investigations"
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No investigations have been performed yet."
        )


# ============================================================
# ALERT TRIAGE
# ============================================================

with alert_tab:

    st.header(
        "🚨 Security Alert Triage"
    )

    st.write(
        "Enter a security alert and TriageMind "
        "will search the MITRE ATT&CK knowledge base."
    )

    alert_text = st.text_area(
        "Security Alert",
        height=180,
        placeholder=(
            "Example:\n"
            "PowerShell executed an encoded command "
            "and downloaded a suspicious payload."
        ),
    )

    top_k = st.slider(
        "MITRE Results",
        1,
        10,
        5,
    )

    if st.button(
        "🔎 ANALYZE ALERT",
        type="primary",
        use_container_width=True,
    ):

        if not alert_text.strip():

            st.warning(
                "Please enter a security alert."
            )

        else:

            with st.spinner(
                "Analyzing security alert..."
            ):

                results = search_mitre(
                    alert_text,
                    top_k,
                )

            st.session_state.alert_results = (
                results
            )

            if results:

                best = results[0]

                confidence = (
                    best["score"]
                    * 100
                )

                st.success(
                    "Alert analysis completed."
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Primary Technique",
                        best["id"],
                    )

                with c2:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%",
                    )

                with c3:

                    st.metric(
                        "Matches",
                        len(results),
                    )

                st.divider()

                st.subheader(
                    "🎯 Primary MITRE Mapping"
                )

                st.write(
                    f"### {best['name']}"
                )

                st.write(
                    f"**Technique ID:** {best['id']}"
                )

                st.write(
                    "**Tactics:** "
                    + (
                        ", ".join(
                            best["tactics"]
                        )
                        if best["tactics"]
                        else "Not specified"
                    )
                )

                st.info(
                    best["description"]
                )

                st.divider()

                st.subheader(
                    "Related Techniques"
                )

                for number, result in enumerate(
                    results,
                    1,
                ):

                    with st.expander(
                        f"{number}. "
                        f"{result['id']} — "
                        f"{result['name']}"
                    ):

                        st.write(
                            "Similarity: "
                            f"{result['score'] * 100:.2f}%"
                        )

                        st.write(
                            "Tactics: "
                            + (
                                ", ".join(
                                    result["tactics"]
                                )
                                if result["tactics"]
                                else "Not specified"
                            )
                        )

                        st.write(
                            result["description"]
                        )

            else:

                st.warning(
                    "No relevant MITRE techniques found."
                )


# ============================================================
# NETWORK DETECTION
# ============================================================

with network_tab:

    st.header(
        "📊 Network Intrusion Detection"
    )

    st.write(
        "Upload a CSV network dataset and run "
        "the trained machine-learning classifier."
    )

    uploaded_file = st.file_uploader(
        "Upload Network CSV",
        type=["csv"],
        key="network_upload",
    )

    if uploaded_file:

        try:

            network_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                "Network dataset loaded successfully."
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Records",
                    len(network_df),
                )

            with c2:

                st.metric(
                    "Columns",
                    len(network_df.columns),
                )

            with c3:

                st.metric(
                    "File Size",
                    f"{uploaded_file.size / 1024:.1f} KB",
                )

            with st.expander(
                "👁️ Dataset Preview",
                expanded=True,
            ):

                st.dataframe(
                    network_df.head(20),
                    use_container_width=True,
                )

            if MODEL is None:

                st.error(
                    "ML model is missing."
                )

            else:

                if st.button(
                    "🚨 RUN INTRUSION DETECTION",
                    type="primary",
                    use_container_width=True,
                ):

                    try:

                        with st.spinner(
                            "Running ML analysis..."
                        ):

                            (
                                predictions,
                                probabilities,
                                prepared_features,
                            ) = predict_network(
                                network_df
                            )

                        attack_flags = [
                            is_attack(
                                value
                            )
                            for value in predictions
                        ]

                        total = len(
                            attack_flags
                        )

                        attack_count = sum(
                            attack_flags
                        )

                        normal_count = (
                            total
                            - attack_count
                        )

                        attack_rate = (
                            attack_count
                            / total
                            * 100
                            if total
                            else 0
                        )

                        prediction_confidence = 0

                        if probabilities is not None:

                            try:

                                prediction_confidence = (
                                    np.max(
                                        probabilities,
                                        axis=1
                                    ).mean()
                                    * 100
                                )

                            except Exception:

                                prediction_confidence = 0

                        result_df = network_df.copy()

                        result_df[
                            "TriageMind Prediction"
                        ] = [
                            "ATTACK"
                            if flag
                            else "NORMAL"
                            for flag in attack_flags
                        ]

                        if probabilities is not None:

                            try:

                                result_df[
                                    "Prediction Confidence"
                                ] = (
                                    np.max(
                                        probabilities,
                                        axis=1
                                    )
                                    * 100
                                ).round(2)

                            except Exception:

                                pass

                        st.session_state.network_results = (
                            result_df
                        )

                        st.session_state.network_original = (
                            network_df
                        )

                        st.success(
                            "Detection completed."
                        )

                        c1, c2, c3, c4 = st.columns(4)

                        with c1:

                            st.metric(
                                "Total Records",
                                total,
                            )

                        with c2:

                            st.metric(
                                "Potential Attacks",
                                attack_count,
                            )

                        with c3:

                            st.metric(
                                "Normal",
                                normal_count,
                            )

                        with c4:

                            st.metric(
                                "Attack Rate",
                                f"{attack_rate:.2f}%",
                            )

                        st.divider()

                        st.subheader(
                            "📈 Detection Distribution"
                        )

                        chart_df = pd.DataFrame(
                            {
                                "Status": [
                                    "Normal",
                                    "Attack",
                                ],
                                "Count": [
                                    normal_count,
                                    attack_count,
                                ],
                            }
                        )

                        st.bar_chart(
                            chart_df.set_index(
                                "Status"
                            )
                        )

                        st.divider()

                        st.subheader(
                            "🔍 Detection Results"
                        )

                        st.dataframe(
                            result_df.head(200),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "📥 DOWNLOAD RESULTS",
                            data=result_df.to_csv(
                                index=False
                            ),
                            file_name=(
                                "triagemind_detection_results.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True,
                        )

                        st.divider()

                        st.subheader(
                            "🧠 Explainable AI"
                        )

                        importance_df = (
                            get_feature_importance()
                        )

                        if importance_df is not None:

                            st.bar_chart(
                                importance_df.head(
                                    15
                                ).set_index(
                                    "Feature"
                                )
                            )

                            st.dataframe(
                                importance_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                        else:

                            st.info(
                                "Feature importance is not "
                                "available for this model."
                            )

                        # ----------------------------------------
                        # EVALUATION
                        # ----------------------------------------

                        label_column = None

                        for column in TARGET_COLUMNS:

                            if column in network_df.columns:

                                label_column = column
                                break

                        if label_column:

                            st.divider()

                            st.subheader(
                                "📐 Model Evaluation"
                            )

                            try:

                                y_true = pd.to_numeric(
                                    network_df[
                                        label_column
                                    ],
                                    errors="coerce",
                                ).fillna(
                                    0
                                ).astype(
                                    int
                                )

                                y_pred = np.array(
                                    [
                                        1
                                        if flag
                                        else 0
                                        for flag in attack_flags
                                    ]
                                )

                                accuracy = accuracy_score(
                                    y_true,
                                    y_pred,
                                )

                                precision = precision_score(
                                    y_true,
                                    y_pred,
                                    zero_division=0,
                                )

                                recall = recall_score(
                                    y_true,
                                    y_pred,
                                    zero_division=0,
                                )

                                f1 = f1_score(
                                    y_true,
                                    y_pred,
                                    zero_division=0,
                                )

                                c1, c2, c3, c4 = (
                                    st.columns(4)
                                )

                                with c1:

                                    st.metric(
                                        "Accuracy",
                                        f"{accuracy * 100:.2f}%",
                                    )

                                with c2:

                                    st.metric(
                                        "Precision",
                                        f"{precision * 100:.2f}%",
                                    )

                                with c3:

                                    st.metric(
                                        "Recall",
                                        f"{recall * 100:.2f}%",
                                    )

                                with c4:

                                    st.metric(
                                        "F1 Score",
                                        f"{f1 * 100:.2f}%",
                                    )

                                matrix = confusion_matrix(
                                    y_true,
                                    y_pred,
                                    labels=[
                                        0,
                                        1,
                                    ],
                                )

                                matrix_df = pd.DataFrame(
                                    matrix,
                                    index=[
                                        "Actual Normal",
                                        "Actual Attack",
                                    ],
                                    columns=[
                                        "Predicted Normal",
                                        "Predicted Attack",
                                    ],
                                )

                                st.subheader(
                                    "Confusion Matrix"
                                )

                                st.dataframe(
                                    matrix_df,
                                    use_container_width=True,
                                )

                            except Exception as error:

                                st.warning(
                                    "Could not calculate "
                                    "evaluation metrics."
                                )

                                st.code(
                                    str(error)
                                )

                    except Exception as error:

                        st.error(
                            "Network analysis failed."
                        )

                        st.code(
                            str(error)
                        )

        except Exception as error:

            st.error(
                "Could not read the CSV file."
            )

            st.code(
                str(error)
            )


# ============================================================
# MITRE ATT&CK
# ============================================================

with mitre_tab:

    st.header(
        "🎯 MITRE ATT&CK Explorer"
    )

    st.write(
        "Search the MITRE ATT&CK knowledge base "
        "using natural-language threat descriptions."
    )

    query = st.text_input(
        "Threat / Behavior Search",
        placeholder=(
            "PowerShell credential dumping "
            "phishing lateral movement"
        ),
    )

    result_count = st.slider(
        "Number of Results",
        1,
        10,
        5,
    )

    if st.button(
        "🔎 SEARCH MITRE ATT&CK",
        type="primary",
        use_container_width=True,
    ):

        if not query.strip():

            st.warning(
                "Enter a search query."
            )

        else:

            with st.spinner(
                "Searching MITRE ATT&CK..."
            ):

                results = search_mitre(
                    query,
                    result_count,
                )

            if results:

                for number, result in enumerate(
                    results,
                    1,
                ):

                    with st.expander(
                        f"{number}. "
                        f"{result['id']} — "
                        f"{result['name']}"
                    ):

                        c1, c2, c3 = st.columns(3)

                        with c1:

                            st.write(
                                "**Technique ID**"
                            )

                            st.code(
                                result["id"]
                            )

                        with c2:

                            st.write(
                                "**Relevance**"
                            )

                            st.write(
                                f"{result['score'] * 100:.2f}%"
                            )

                        with c3:

                            st.write(
                                "**Tactics**"
                            )

                            st.write(
                                ", ".join(
                                    result["tactics"]
                                )
                                if result["tactics"]
                                else "Not specified"
                            )

                        st.write(
                            "**Description**"
                        )

                        st.info(
                            result["description"]
                        )

            else:

                st.warning(
                    "No matching MITRE techniques found."
                )

    st.divider()

    st.subheader(
        "MITRE Knowledge Base"
    )

    if MITRE_TECHNIQUES:

        st.success(
            f"{len(MITRE_TECHNIQUES)} "
            "techniques loaded."
        )

        mitre_table = pd.DataFrame(
            [
                {
                    "Technique ID": item["id"],
                    "Technique": item["name"],
                    "Tactics": ", ".join(
                        item["tactics"]
                    ),
                }
                for item in MITRE_TECHNIQUES
            ]
        )

        st.dataframe(
            mitre_table,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.error(
            "MITRE ATT&CK data is unavailable."
        )


# ============================================================
# SOC INVESTIGATION
# ============================================================

with soc_tab:

    st.header(
        "🛡️ Full SOC Investigation"
    )

    st.write(
        "Combine network detection, MITRE ATT&CK "
        "mapping and risk scoring."
    )

    st.divider()

    soc_alert = st.text_area(
        "1️⃣ Security Alert",
        height=150,
        placeholder=(
            "PowerShell executed an encoded command "
            "and attempted to download a remote payload."
        ),
    )

    soc_file = st.file_uploader(
        "2️⃣ Network Traffic CSV",
        type=["csv"],
        key="soc_upload",
    )

    if st.button(
        "🚨 RUN COMPLETE SOC TRIAGE",
        type="primary",
        use_container_width=True,
    ):

        if not soc_alert.strip():

            st.warning(
                "Enter a security alert."
            )

        elif soc_file is None:

            st.warning(
                "Upload a network CSV."
            )

        elif MODEL is None:

            st.error(
                "ML model is not available."
            )

        else:

            try:

                with st.spinner(
                    "Running SOC investigation..."
                ):

                    soc_df = pd.read_csv(
                        soc_file
                    )

                    (
                        predictions,
                        probabilities,
                        prepared_features,
                    ) = predict_network(
                        soc_df
                    )

                    mitre_results = search_mitre(
                        soc_alert,
                        5,
                    )

                attack_flags = [
                    is_attack(
                        value
                    )
                    for value in predictions
                ]

                total = len(
                    attack_flags
                )

                attacks = sum(
                    attack_flags
                )

                normal = (
                    total
                    - attacks
                )

                attack_rate = (
                    attacks
                    / total
                    * 100
                    if total
                    else 0
                )

                mitre_confidence = (
                    mitre_results[0]["score"]
                    * 100
                    if mitre_results
                    else 0
                )

                prediction_confidence = 0

                if probabilities is not None:

                    try:

                        prediction_confidence = (
                            np.max(
                                probabilities,
                                axis=1
                            ).mean()
                            * 100
                        )

                    except Exception:

                        prediction_confidence = 0

                risk = calculate_risk(
                    attack_rate,
                    mitre_confidence,
                    prediction_confidence,
                )

                severity = get_severity(
                    risk
                )

                primary_technique = (
                    mitre_results[0]["name"]
                    if mitre_results
                    else "Not identified"
                )

                st.session_state.history.append(
                    {
                        "Time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Technique": primary_technique,
                        "Risk Score": risk,
                        "Severity": severity,
                        "Attack Rate": round(
                            attack_rate,
                            2,
                        ),
                    }
                )

                st.success(
                    "SOC investigation completed."
                )

                st.divider()

                st.subheader(
                    "🚦 Risk Assessment"
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "Risk Score",
                        f"{risk}/100",
                    )

                with c2:

                    st.metric(
                        "Severity",
                        severity,
                    )

                with c3:

                    st.metric(
                        "Attack Rate",
                        f"{attack_rate:.2f}%",
                    )

                with c4:

                    st.metric(
                        "MITRE Confidence",
                        f"{mitre_confidence:.2f}%",
                    )

                st.progress(
                    int(risk)
                )

                st.divider()

                st.subheader(
                    "📊 Network Detection"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Records",
                        total,
                    )

                with c2:

                    st.metric(
                        "Potential Attacks",
                        attacks,
                    )

                with c3:

                    st.metric(
                        "Normal",
                        normal,
                    )

                st.divider()

                st.subheader(
                    "🎯 MITRE ATT&CK Mapping"
                )

                if mitre_results:

                    best = mitre_results[0]

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.write(
                            "**Technique ID**"
                        )

                        st.code(
                            best["id"]
                        )

                    with c2:

                        st.write(
                            "**Technique**"
                        )

                        st.write(
                            best["name"]
                        )

                    with c3:

                        st.write(
                            "**Confidence**"
                        )

                        st.write(
                            f"{best['score'] * 100:.2f}%"
                        )

                    st.write(
                        "**Tactics:** "
                        + (
                            ", ".join(
                                best["tactics"]
                            )
                            if best["tactics"]
                            else "Not specified"
                        )
                    )

                    st.info(
                        best["description"]
                    )

                else:

                    st.info(
                        "No MITRE technique identified."
                    )

                st.divider()

                st.subheader(
                    "👨‍💻 SOC Investigation Workflow"
                )

                workflow = [
                    (
                        "1. Validate Alert",
                        "Confirm the alert and investigate "
                        "whether the activity is suspicious."
                    ),
                    (
                        "2. Identify Endpoint",
                        "Determine the affected system, "
                        "user and network segment."
                    ),
                    (
                        "3. Investigate Process Activity",
                        "Review suspicious processes, "
                        "commands and parent processes."
                    ),
                    (
                        "4. Correlate Network Events",
                        "Review DNS, connections and "
                        "unusual outbound traffic."
                    ),
                    (
                        "5. Map ATT&CK Technique",
                        "Use ATT&CK context to guide "
                        "additional investigation."
                    ),
                    (
                        "6. Assess Scope",
                        "Search for related hosts, "
                        "accounts and alerts."
                    ),
                    (
                        "7. Contain",
                        "Follow the organization's "
                        "incident-response procedures."
                    ),
                    (
                        "8. Document",
                        "Preserve evidence and document "
                        "the investigation."
                    ),
                ]

                for title, description in workflow:

                    with st.expander(
                        title
                    ):

                        st.write(
                            description
                        )

                st.divider()

                st.subheader(
                    "📄 Security Report"
                )

                report = f"""
============================================================
                    TRIAGEMIND AI
              SECURITY TRIAGE REPORT
============================================================

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

------------------------------------------------------------
SECURITY ALERT
------------------------------------------------------------

{soc_alert}

------------------------------------------------------------
RISK ASSESSMENT
------------------------------------------------------------

Risk Score:
{risk}/100

Severity:
{severity}

Attack Rate:
{attack_rate:.2f}%

MITRE Confidence:
{mitre_confidence:.2f}%

------------------------------------------------------------
NETWORK DETECTION
------------------------------------------------------------

Total Records:
{total}

Potential Attacks:
{attacks}

Normal Records:
{normal}

------------------------------------------------------------
MITRE ATT&CK
------------------------------------------------------------
"""

                if mitre_results:

                    best = mitre_results[0]

                    report += f"""
Technique ID:
{best["id"]}

Technique:
{best["name"]}

Tactics:
{", ".join(best["tactics"])}

Confidence:
{best["score"] * 100:.2f}%

Description:
{best["description"]}
"""

                else:

                    report += (
                        "\nNo MITRE technique identified.\n"
                    )

                report += """
------------------------------------------------------------
SOC INVESTIGATION WORKFLOW
------------------------------------------------------------
"""

                for title, description in workflow:

                    report += (
                        f"\n{title}\n"
                        f"{description}\n"
                    )

                report += """
------------------------------------------------------------
END OF REPORT
------------------------------------------------------------
TriageMind AI
"""

                st.download_button(
                    "📥 DOWNLOAD SECURITY REPORT",
                    data=report,
                    file_name=(
                        "triagemind_security_report.txt"
                    ),
                    mime="text/plain",
                    use_container_width=True,
                )

            except Exception as error:

                st.error(
                    "SOC investigation failed."
                )

                st.code(
                    str(error)
                )


# ============================================================
# DIAGNOSTICS
# ============================================================

with diagnostics_tab:

    st.header(
        "⚙️ TriageMind Diagnostics"
    )

    st.write(
        "Check whether TriageMind can find "
        "its required project files."
    )

    st.divider()

    diagnostics = pd.DataFrame(
        {
            "Component": [
                "Application",
                "ML Model",
                "Feature Columns",
                "MITRE ATT&CK",
                "FAISS",
            ],
            "Status": [
                "READY",
                "READY"
                if MODEL is not None
                else "MISSING",
                "READY"
                if FEATURE_COLUMNS is not None
                else "OPTIONAL",
                "READY"
                if MITRE_TECHNIQUES
                else "MISSING",
                "AVAILABLE"
                if FAISS_PATH
                else "OPTIONAL",
            ],
            "Path": [
                str(BASE_DIR),
                str(MODEL_PATH)
                if MODEL_PATH
                else "Not found",
                str(FEATURE_PATH)
                if FEATURE_PATH
                else "Not found",
                str(MITRE_PATH)
                if MITRE_PATH
                else "Not found",
                str(FAISS_PATH)
                if FAISS_PATH
                else "Not found",
            ],
        }
    )

    st.dataframe(
        diagnostics,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "ML Model"
    )

    if MODEL is not None:

        st.success(
            "ML classifier loaded successfully."
        )

        st.write(
            "Model type:",
            type(MODEL).__name__
        )

        if FEATURE_COLUMNS:

            st.write(
                "Expected features:",
                len(FEATURE_COLUMNS)
            )

    else:

        st.error(
            "ML model not found."
        )

        if MODEL_ERROR:

            st.code(
                MODEL_ERROR
            )

    st.divider()

    st.subheader(
        "MITRE ATT&CK"
    )

    if MITRE_TECHNIQUES:

        st.success(
            f"{len(MITRE_TECHNIQUES)} "
            "MITRE techniques loaded."
        )

    else:

        st.error(
            "MITRE ATT&CK data not found."
        )

        if MITRE_ERROR:

            st.code(
                MITRE_ERROR
            )

    st.divider()

    st.subheader(
        "Expected Project Structure"
    )

    st.code(
        """
TriageMind/
│
├── app.py
├── requirements.txt
│
├── model/
│   ├── classifier.pkl
│   └── feature_columns.pkl
│
├── mitre_data/
│   ├── enterprise-attack.json
│   └── mitre_techniques.json
│
└── dataset/
    └── UNSW-NB15.csv
        """,
        language="text",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ TriageMind AI v2.0 | "
    "Machine Learning + MITRE ATT&CK + "
    "Explainable AI + SOC Automation"
)