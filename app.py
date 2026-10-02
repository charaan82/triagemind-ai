import os
import pandas as pd
import streamlit as st
import joblib

from triagemind_engine import (
    check_files,
    load_classifier,
    load_mitre_techniques,
    load_faiss,
    load_embedding_model,
    search_mitre,
    classify_csv
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TriageMind AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .risk-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dddddd;
        margin-bottom: 20px;
    }

    .small-note {
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ TriageMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Cybersecurity Alert Triage Platform'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "TriageMind combines machine-learning based network "
    "detection with MITRE ATT&CK knowledge retrieval "
    "to assist security analysts during alert triage."
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "triage_report" not in st.session_state:
    st.session_state.triage_report = None


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not check_files():

    st.error(
        "❌ Required TriageMind files are missing."
    )

    st.info(
        "Please check your model, MITRE data and "
        "FAISS files."
    )

    st.stop()


# ============================================================
# LOAD ML EVALUATION RESULTS
# ============================================================

try:

    model_metrics = joblib.load(
        "model_metrics.pkl"
    )

    confusion_matrix_data = joblib.load(
        "confusion_matrix.pkl"
    )

    evaluation_available = True

except Exception:

    model_metrics = {}
    confusion_matrix_data = None
    evaluation_available = False


# ============================================================
# LOAD TRIAGEMIND MODELS
# ============================================================

@st.cache_resource
def load_triagemind():

    classifier, feature_columns = (
        load_classifier()
    )

    techniques = (
        load_mitre_techniques()
    )

    mitre_index = (
        load_faiss()
    )

    embedding_model = (
        load_embedding_model()
    )

    return (
        classifier,
        feature_columns,
        techniques,
        mitre_index,
        embedding_model
    )


with st.spinner(
    "Loading TriageMind AI components..."
):

    (
        classifier,
        feature_columns,
        techniques,
        mitre_index,
        embedding_model
    ) = load_triagemind()


st.success(
    "✅ TriageMind AI is ready"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ TriageMind")

    st.write(
        "AI-assisted cybersecurity alert triage"
    )

    st.divider()

    st.subheader(
        "System Components"
    )

    st.write("✅ Random Forest")
    st.write("✅ UNSW-NB15")
    st.write("✅ MITRE ATT&CK")
    st.write("✅ Sentence-BERT")
    st.write("✅ FAISS")

    st.divider()

    st.subheader(
        "Workflow"
    )

    st.write(
        """
        1. Security alert
        2. Network data
        3. ML detection
        4. MITRE retrieval
        5. Risk assessment
        6. Analyst guidance
        7. Triage report
        """
    )

    st.divider()

    if st.button(
        "🔄 Start New Analysis"
    ):

        st.session_state.analysis_complete = False
        st.session_state.triage_report = None

        st.rerun()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Dashboard",
        "🔍 Alert Analysis",
        "📊 Network Analysis",
        "🛡️ Full Triage",
        "📖 About"
    ]
)


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:

    st.header(
        "🏠 TriageMind Dashboard"
    )

    st.write(
        "Overview of the TriageMind cybersecurity "
        "detection and threat-intelligence system."
    )

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.subheader(
        "🟢 System Status"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            "Random Forest\n\nREADY"
        )

    with col2:

        st.success(
            "MITRE ATT&CK\n\nREADY"
        )

    with col3:

        st.success(
            "Sentence-BERT + FAISS\n\nREADY"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "🧠 Machine Learning Performance"
    )

    if evaluation_available:

        accuracy = model_metrics.get(
            "accuracy",
            0
        )

        precision = model_metrics.get(
            "precision",
            0
        )

        recall = model_metrics.get(
            "recall",
            0
        )

        f1 = model_metrics.get(
            "f1_score",
            0
        )

        roc_auc = model_metrics.get(
            "roc_auc",
            0
        )

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Precision",
                f"{precision * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Recall",
                f"{recall * 100:.2f}%"
            )

        with col4:

            st.metric(
                "F1 Score",
                f"{f1 * 100:.2f}%"
            )

        with col5:

            st.metric(
                "ROC-AUC",
                f"{roc_auc:.3f}"
            )

        st.caption(
            "These values come from evaluate_model.py."
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.subheader(
            "📊 Confusion Matrix"
        )

        if confusion_matrix_data is not None:

            cm_df = pd.DataFrame(
                confusion_matrix_data,
                index=[
                    "Actual Normal",
                    "Actual Attack"
                ],
                columns=[
                    "Predicted Normal",
                    "Predicted Attack"
                ]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

            if os.path.exists(
                "confusion_matrix.png"
            ):

                st.image(
                    "confusion_matrix.png",
                    caption=(
                        "TriageMind Random Forest "
                        "Confusion Matrix"
                    )
                )

    else:

        st.warning(
            "ML evaluation results are not available."
        )

        st.code(
            "python evaluate_model.py"
        )

    st.divider()

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.subheader(
        "🏗️ TriageMind Architecture"
    )

    st.code(
        """
                    TRIAGEMIND AI
                         |
             +-----------+-----------+
             |                       |
             v                       v
      Security Alert           Network Data
             |                       |
             v                       v
       Sentence-BERT           Random Forest
             |                       |
             v                       v
           FAISS              Attack / Normal
             |                       |
             v                       |
       MITRE ATT&CK                  |
             |                       |
             +-----------+-----------+
                         |
                         v
                  Risk Assessment
                         |
                         v
                 Analyst Guidance
                         |
                         v
                  Triage Report
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Technology Stack"
    )

    technology_df = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Streamlit",
                "Scikit-learn",
                "Random Forest",
                "Sentence-BERT",
                "FAISS",
                "MITRE ATT&CK",
                "UNSW-NB15"
            ],

            "Purpose": [
                "Application development",
                "Web dashboard",
                "Machine learning",
                "Network intrusion detection",
                "Security text embeddings",
                "Vector similarity search",
                "Threat intelligence",
                "Network security dataset"
            ]
        }
    )

    st.dataframe(
        technology_df,
        use_container_width=True
    )


# ============================================================
# TAB 2 — SECURITY ALERT ANALYSIS
# ============================================================

with tab2:

    st.header(
        "🔍 Security Alert Analysis"
    )

    st.write(
        "Enter suspicious activity or a security "
        "alert to search the local MITRE ATT&CK "
        "knowledge base."
    )

    alert = st.text_area(
        "Security Alert",
        placeholder=(
            "Example:\n"
            "PowerShell was used to execute a "
            "malicious encoded command."
        ),
        height=180,
        key="alert_input"
    )

    if st.button(
        "🚨 Analyze Security Alert",
        type="primary",
        key="alert_button"
    ):

        if not alert.strip():

            st.warning(
                "Please enter a security alert."
            )

        else:

            with st.spinner(
                "Searching MITRE ATT&CK..."
            ):

                try:

                    matches = search_mitre(
                        alert,
                        embedding_model,
                        mitre_index,
                        techniques
                    )

                except Exception as error:

                    st.error(
                        f"MITRE search failed: {error}"
                    )

                    matches = []

            st.divider()

            if matches:

                best = matches[0]

                st.subheader(
                    "🎯 Best MITRE ATT&CK Match"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Technique ID",
                        best["id"]
                    )

                with col2:

                    st.metric(
                        "Technique",
                        best["name"]
                    )

                with col3:

                    st.metric(
                        "Distance",
                        round(
                            best["distance"],
                            4
                        )
                    )

                st.write(
                    "**Tactic:**",
                    ", ".join(
                        best["tactics"]
                    )
                )

                st.write(
                    "**Technique Description:**"
                )

                st.info(
                    best["description"]
                )

                if len(matches) > 1:

                    st.subheader(
                        "🔎 Other Relevant Techniques"
                    )

                    for index, match in enumerate(
                        matches[1:],
                        start=2
                    ):

                        with st.expander(
                            f"{index}. "
                            f"{match['id']} — "
                            f"{match['name']}"
                        ):

                            st.write(
                                "**Tactics:**",
                                ", ".join(
                                    match["tactics"]
                                )
                            )

                            st.write(
                                "**Distance:**",
                                round(
                                    match["distance"],
                                    4
                                )
                            )

                            st.write(
                                "**Description:**"
                            )

                            st.write(
                                match["description"]
                            )

            else:

                st.warning(
                    "No relevant MITRE ATT&CK "
                    "technique was found."
                )


# ============================================================
# TAB 3 — NETWORK ANALYSIS
# ============================================================

with tab3:

    st.header(
        "📊 Network Intrusion Analysis"
    )

    st.write(
        "Upload UNSW-NB15 network data and use "
        "the Random Forest classifier to identify "
        "potentially suspicious records."
    )

    uploaded_file = st.file_uploader(
        "Upload Network CSV",
        type=["csv"],
        key="network_upload"
    )

    if uploaded_file is not None:

        try:

            preview_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"File selected: "
                f"{uploaded_file.name}"
            )

            st.subheader(
                "📋 Dataset Preview"
            )

            st.dataframe(
                preview_df.head(10),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Rows",
                    len(preview_df)
                )

            with col2:

                st.metric(
                    "Columns",
                    len(preview_df.columns)
                )

        except Exception as error:

            st.error(
                f"Could not read CSV: {error}"
            )

        if st.button(
            "🔎 Analyze Network Data",
            type="primary",
            key="network_button"
        ):

            temp_file = (
                "uploaded_network_data.csv"
            )

            uploaded_file.seek(0)

            with open(
                temp_file,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            with st.spinner(
                "Running Random Forest..."
            ):

                try:

                    predictions = classify_csv(
                        temp_file,
                        classifier,
                        feature_columns
                    )

                except Exception as error:

                    st.error(
                        f"Network analysis failed: "
                        f"{error}"
                    )

                    predictions = None

            if predictions is not None:

                predictions = list(
                    predictions
                )

                st.success(
                    "✅ Network analysis completed."
                )

                prediction_series = pd.Series(
                    predictions
                )

                total_records = len(
                    predictions
                )

                attack_records = int(
                    prediction_series.eq(1).sum()
                )

                normal_records = int(
                    prediction_series.eq(0).sum()
                )

                if total_records > 0:

                    attack_percentage = (
                        attack_records /
                        total_records *
                        100
                    )

                else:

                    attack_percentage = 0

                st.subheader(
                    "📊 Detection Summary"
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                with col1:

                    st.metric(
                        "Total Records",
                        total_records
                    )

                with col2:

                    st.metric(
                        "Attack Records",
                        attack_records
                    )

                with col3:

                    st.metric(
                        "Normal Records",
                        normal_records
                    )

                with col4:

                    st.metric(
                        "Attack %",
                        f"{attack_percentage:.2f}%"
                    )

                st.subheader(
                    "📈 Attack vs Normal"
                )

                chart_df = pd.DataFrame(
                    {
                        "Prediction": [
                            "Normal",
                            "Attack"
                        ],
                        "Records": [
                            normal_records,
                            attack_records
                        ]
                    }
                )

                st.bar_chart(
                    chart_df.set_index(
                        "Prediction"
                    )
                )

                st.subheader(
                    "🔎 Prediction Samples"
                )

                sample_predictions = pd.DataFrame(
                    {
                        "Row": range(
                            1,
                            min(
                                total_records,
                                25
                            ) + 1
                        ),

                        "Prediction": [
                            (
                                "Attack"
                                if int(value) == 1
                                else "Normal"
                            )
                            for value in
                            predictions[:25]
                        ]
                    }
                )

                st.dataframe(
                    sample_predictions,
                    use_container_width=True
                )

                result_df = pd.DataFrame(
                    {
                        "Prediction": [
                            (
                                "Attack"
                                if int(value) == 1
                                else "Normal"
                            )
                            for value in predictions
                        ]
                    }
                )

                csv_data = result_df.to_csv(
                    index=False
                )

                st.download_button(
                    "📥 Download Prediction CSV",
                    data=csv_data,
                    file_name=(
                        "triagemind_predictions.csv"
                    ),
                    mime="text/csv"
                )


# ============================================================
# TAB 4 — FULL TRIAGE
# ============================================================

with tab4:

    st.header(
        "🛡️ Complete TriageMind Investigation"
    )

    st.write(
        "Combine security-alert analysis and "
        "network intrusion detection into one "
        "investigation report."
    )

    st.subheader(
        "1️⃣ Security Alert"
    )

    full_alert = st.text_area(
        "Enter the Security Alert",
        placeholder=(
            "PowerShell was used to execute "
            "a malicious encoded command."
        ),
        height=150,
        key="full_alert_input"
    )

    st.subheader(
        "2️⃣ Network Dataset"
    )

    full_file = st.file_uploader(
        "Upload UNSW-NB15 CSV",
        type=["csv"],
        key="full_network_upload"
    )

    if st.button(
        "🚨 Run Complete Triage",
        type="primary",
        key="full_triage_button"
    ):

        if not full_alert.strip():

            st.warning(
                "Please enter a security alert."
            )

            st.stop()

        if full_file is None:

            st.warning(
                "Please upload network data."
            )

            st.stop()

        # ----------------------------------------------------
        # MITRE ANALYSIS
        # ----------------------------------------------------

        with st.spinner(
            "Step 1/3 — Analyzing MITRE ATT&CK..."
        ):

            try:

                mitre_matches = search_mitre(
                    full_alert,
                    embedding_model,
                    mitre_index,
                    techniques
                )

            except Exception as error:

                st.error(
                    f"MITRE analysis failed: "
                    f"{error}"
                )

                mitre_matches = []

        # ----------------------------------------------------
        # NETWORK ANALYSIS
        # ----------------------------------------------------

        with st.spinner(
            "Step 2/3 — Running network detection..."
        ):

            temp_file = (
                "full_triage_network.csv"
            )

            full_file.seek(0)

            with open(
                temp_file,
                "wb"
            ) as file:

                file.write(
                    full_file.getbuffer()
                )

            try:

                predictions = classify_csv(
                    temp_file,
                    classifier,
                    feature_columns
                )

            except Exception as error:

                st.error(
                    f"Network analysis failed: "
                    f"{error}"
                )

                predictions = []

        predictions = list(
            predictions
        )

        # ----------------------------------------------------
        # NETWORK STATISTICS
        # ----------------------------------------------------

        total_records = len(
            predictions
        )

        attack_records = sum(
            1
            for value in predictions
            if int(value) == 1
        )

        normal_records = (
            total_records -
            attack_records
        )

        if total_records > 0:

            attack_percentage = (
                attack_records /
                total_records *
                100
            )

        else:

            attack_percentage = 0

        # ----------------------------------------------------
        # HEURISTIC RISK SCORE
        # ----------------------------------------------------

        if total_records == 0:

            risk_score = 0

        else:

            risk_score = (
                attack_percentage
            )

            if mitre_matches:

                risk_score += 15

            risk_score = min(
                risk_score,
                100
            )

        if risk_score >= 75:

            severity = "CRITICAL"

        elif risk_score >= 50:

            severity = "HIGH"

        elif risk_score >= 25:

            severity = "MEDIUM"

        else:

            severity = "LOW"

        # ----------------------------------------------------
        # BEST MITRE MATCH
        # ----------------------------------------------------

        best_mitre = (
            mitre_matches[0]
            if mitre_matches
            else None
        )

        # ----------------------------------------------------
        # SAVE REPORT
        # ----------------------------------------------------

        st.session_state.triage_report = {

            "alert": full_alert,

            "total_records":
                total_records,

            "attack_records":
                attack_records,

            "normal_records":
                normal_records,

            "attack_percentage":
                attack_percentage,

            "risk_score":
                risk_score,

            "severity":
                severity,

            "mitre_matches":
                mitre_matches,

            "best_mitre":
                best_mitre
        }

        st.session_state.analysis_complete = True

        st.success(
            "✅ Step 3/3 — Triage report generated."
        )

    # ========================================================
    # DISPLAY REPORT
    # ========================================================

    if st.session_state.analysis_complete:

        report = (
            st.session_state.triage_report
        )

        st.divider()

        st.header(
            "🛡️ TriageMind Investigation Report"
        )

        # ----------------------------------------------------
        # RISK ASSESSMENT
        # ----------------------------------------------------

        st.subheader(
            "🚦 Risk Assessment"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Risk Score",
                f"{report['risk_score']:.1f}/100"
            )

        with col2:

            st.metric(
                "Severity",
                report["severity"]
            )

        with col3:

            st.metric(
                "Attack %",
                f"{report['attack_percentage']:.2f}%"
            )

        st.progress(
            int(
                report["risk_score"]
            ) / 100
        )

        st.caption(
            "The risk score is a heuristic project "
            "indicator based on the predicted attack "
            "ratio and MITRE evidence. It is not a "
            "calibrated security-risk probability."
        )

        # ----------------------------------------------------
        # NETWORK SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📊 Network Detection"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Records",
                report["total_records"]
            )

        with col2:

            st.metric(
                "Attack Records",
                report["attack_records"]
            )

        with col3:

            st.metric(
                "Normal Records",
                report["normal_records"]
            )

        network_chart = pd.DataFrame(
            {
                "Type": [
                    "Normal",
                    "Attack"
                ],

                "Records": [
                    report["normal_records"],
                    report["attack_records"]
                ]
            }
        )

        st.bar_chart(
            network_chart.set_index(
                "Type"
            )
        )

        # ----------------------------------------------------
        # MITRE ATT&CK
        # ----------------------------------------------------

        st.subheader(
            "🎯 MITRE ATT&CK Analysis"
        )

        if report["best_mitre"]:

            best = report["best_mitre"]

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Technique ID",
                    best["id"]
                )

            with col2:

                st.metric(
                    "Technique",
                    best["name"]
                )

            st.write(
                "**Tactic:**",
                ", ".join(
                    best["tactics"]
                )
            )

            st.write(
                "**Similarity Distance:**",
                round(
                    best["distance"],
                    4
                )
            )

            st.write(
                "**Description:**"
            )

            st.info(
                best["description"]
            )

        else:

            st.warning(
                "No MITRE ATT&CK match found."
            )

        # ----------------------------------------------------
        # RELATED MITRE TECHNIQUES
        # ----------------------------------------------------

        if len(
            report["mitre_matches"]
        ) > 1:

            st.subheader(
                "🔎 Related MITRE Techniques"
            )

            mitre_table = []

            for match in (
                report["mitre_matches"]
            ):

                mitre_table.append(
                    {
                        "Technique ID":
                            match["id"],

                        "Technique":
                            match["name"],

                        "Tactic":
                            ", ".join(
                                match["tactics"]
                            ),

                        "Distance":
                            round(
                                match["distance"],
                                4
                            )
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    mitre_table
                ),
                use_container_width=True
            )

        # ----------------------------------------------------
        # SECURITY EVIDENCE
        # ----------------------------------------------------

        st.subheader(
            "🔎 Security Evidence"
        )

        st.code(
            report["alert"]
        )

        # ----------------------------------------------------
        # ANALYST GUIDANCE
        # ----------------------------------------------------

        st.subheader(
            "👨‍💻 Analyst Guidance"
        )

        if report["attack_records"] > 0:

            st.warning(
                "Network activity contains records "
                "classified as potential attacks. "
                "Investigate the associated hosts, "
                "connections and processes."
            )

        else:

            st.success(
                "No network records were classified "
                "as attacks by the current model."
            )

        if report["best_mitre"]:

            st.write(
                "Investigate activity associated "
                "with "
                f"**{report['best_mitre']['name']}**."
            )

            st.write(
                "Review endpoint process activity, "
                "authentication events, network "
                "connections and relevant security logs."
            )

        st.write(
            "Correlate the alert with additional "
            "security telemetry before taking "
            "containment or remediation actions."
        )

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------

        st.subheader(
            "📥 Export Investigation"
        )

        report_text = f"""
TRIAGEMIND AI — SECURITY TRIAGE REPORT
======================================

SECURITY ALERT
--------------
{report['alert']}

RISK ASSESSMENT
---------------
Risk Score: {report['risk_score']:.1f}/100
Severity: {report['severity']}
Attack Percentage: {report['attack_percentage']:.2f}%

NETWORK DETECTION
-----------------
Total Records: {report['total_records']}
Attack Records: {report['attack_records']}
Normal Records: {report['normal_records']}

MITRE ATT&CK
-------------

"""

        if report["best_mitre"]:

            report_text += f"""
Technique ID:
{report['best_mitre']['id']}

Technique:
{report['best_mitre']['name']}

Tactic:
{", ".join(report['best_mitre']['tactics'])}

Similarity Distance:
{report['best_mitre']['distance']}

Description:
{report['best_mitre']['description']}
"""

        else:

            report_text += """
No MITRE ATT&CK technique was identified.
"""

        report_text += """

ANALYST GUIDANCE
----------------
Investigate suspicious network activity,
review affected hosts and processes,
correlate endpoint and network logs,
and validate MITRE ATT&CK evidence.

IMPORTANT
---------
The risk score is a heuristic project indicator.
It is not a calibrated probability or a substitute
for professional security investigation.
"""

        st.download_button(
            "📥 Download Triage Report",
            data=report_text,
            file_name=(
                "triagemind_triage_report.txt"
            ),
            mime="text/plain"
        )


# ============================================================
# TAB 5 — ABOUT
# ============================================================

with tab5:

    st.header(
        "📖 About TriageMind"
    )

    st.write(
        """
        TriageMind is an AI-assisted cybersecurity
        alert triage prototype that combines machine
        learning, semantic search and MITRE ATT&CK
        threat intelligence.
        """
    )

    st.subheader(
        "🏗️ Project Architecture"
    )

    st.code(
        """
                    TRIAGEMIND
                        |
          +-------------+-------------+
          |                           |
          v                           v
     Security Alert             Network Dataset
          |                           |
          v                           v
    Sentence-BERT              Random Forest
          |                           |
          v                           v
        FAISS                  Attack / Normal
          |                           |
          v                           |
     MITRE ATT&CK                     |
          |                           |
          +-------------+-------------+
                        |
                        v
                 Risk Assessment
                        |
                        v
                Analyst Guidance
                        |
                        v
                 Final Report
        """
    )

    st.subheader(
        "⚙️ Technology Stack"
    )

    technology_df = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Streamlit",
                "Pandas",
                "Scikit-learn",
                "Random Forest",
                "Sentence-BERT",
                "FAISS",
                "MITRE ATT&CK",
                "UNSW-NB15"
            ],

            "Purpose": [
                "Core programming language",
                "Web application",
                "Data processing",
                "Machine learning",
                "Network classification",
                "Text embeddings",
                "Vector search",
                "Threat intelligence",
                "Intrusion-detection dataset"
            ]
        }
    )

    st.dataframe(
        technology_df,
        use_container_width=True
    )

    st.subheader(
        "📌 Project Features"
    )

    st.write(
        """
        • Network intrusion detection

        • MITRE ATT&CK technique retrieval

        • Sentence-BERT semantic embeddings

        • FAISS similarity search

        • Machine-learning evaluation

        • Risk assessment

        • Analyst guidance

        • Downloadable triage reports

        • Prediction CSV export
        """
    )

    st.info(
        "TriageMind is an academic/project prototype. "
        "Its predictions and heuristic risk score should "
        "be validated with additional security telemetry "
        "before real-world response actions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TriageMind AI • Cybersecurity Alert Triage • "
    "Random Forest + MITRE ATT&CK + "
    "Sentence-BERT + FAISS"
)