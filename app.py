import os
import json
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

import triagemind_engine as engine


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
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MITRE_DIR = os.path.join(
    BASE_DIR,
    "mitre_data"
)

MITRE_FILE = os.path.join(
    MITRE_DIR,
    "enterprise-attack.json"
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_alert" not in st.session_state:
    st.session_state.last_alert = None

if "last_network" not in st.session_state:
    st.session_state.last_network = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #172554
        );
        border: 1px solid #334155;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin: 0;
    }

    .hero p {
        color: #cbd5e1;
        margin: 6px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ENGINE FUNCTION ACCESS
# ============================================================

def get_engine_function(name):

    return getattr(
        engine,
        name,
        None
    )


load_classifier = get_engine_function(
    "load_classifier"
)

load_faiss = get_engine_function(
    "load_faiss"
)

load_embedding_model = get_engine_function(
    "load_embedding_model"
)

search_mitre_engine = get_engine_function(
    "search_mitre"
)

classify_csv = get_engine_function(
    "classify_csv"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model_safely():

    if load_classifier is None:

        return None, []

    try:

        result = load_classifier()

        classifier = None
        feature_columns = []

        # --------------------------------------------
        # Case 1: Function returns tuple/list
        # --------------------------------------------

        if isinstance(
            result,
            (tuple, list)
        ):

            if len(result) > 0:

                classifier = result[0]

            # Search all remaining returned values
            # for feature columns.
            for value in result[1:]:

                if isinstance(
                    value,
                    (list, tuple)
                ):

                    feature_columns = list(
                        value
                    )

                    break

                if hasattr(
                    value,
                    "tolist"
                ):

                    try:

                        converted = value.tolist()

                        if isinstance(
                            converted,
                            list
                        ):

                            feature_columns = converted

                            break

                    except Exception:
                        pass

        # --------------------------------------------
        # Case 2: Function returns only model
        # --------------------------------------------

        else:

            classifier = result

        return (
            classifier,
            feature_columns
        )

    except Exception as error:

        st.error(
            "Unable to load the ML model."
        )

        st.code(
            str(error)
        )

        return (
            None,
            []
        )


classifier, feature_columns = load_model_safely()


# ============================================================
# FAISS LOADING
# ============================================================

@st.cache_resource
def load_faiss_safely():

    if load_faiss is None:

        return None

    try:

        return load_faiss()

    except Exception:

        return None


faiss_index = load_faiss_safely()


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_safely():

    if load_embedding_model is None:

        return None

    try:

        return load_embedding_model()

    except Exception:

        return None


embedding_model = load_embedding_safely()


# ============================================================
# MITRE ATT&CK DATA
# ============================================================

@st.cache_data
def load_mitre_data():

    if not os.path.exists(
        MITRE_FILE
    ):

        return []

    try:

        with open(
            MITRE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        if isinstance(
            data,
            dict
        ):

            return data.get(
                "objects",
                []
            )

        if isinstance(
            data,
            list
        ):

            return data

    except Exception:

        return []

    return []


mitre_data = load_mitre_data()


# ============================================================
# LOCAL MITRE SEARCH
# ============================================================

def local_mitre_search(
    query
):

    if not mitre_data:

        return []

    query = query.lower().strip()

    results = []

    for item in mitre_data:

        if not isinstance(
            item,
            dict
        ):

            continue

        name = str(
            item.get(
                "name",
                ""
            )
        )

        description = str(
            item.get(
                "description",
                ""
            )
        )

        technique_id = ""

        references = item.get(
            "external_references",
            []
        )

        for reference in references:

            if not isinstance(
                reference,
                dict
            ):

                continue

            if reference.get(
                "source_name"
            ) == "mitre-attack":

                technique_id = str(
                    reference.get(
                        "external_id",
                        ""
                    )
                )

                break

        searchable_text = (
            name
            + " "
            + description
            + " "
            + technique_id
        ).lower()

        if query in searchable_text:

            results.append(
                {
                    "id": technique_id,
                    "name": name,
                    "description": description
                }
            )

        if len(results) >= 10:

            break

    return results


# ============================================================
# MITRE SEARCH
# ============================================================

def search_mitre(
    query
):

    if not query.strip():

        return []

    # Try semantic search first
    if (
        search_mitre_engine is not None
        and embedding_model is not None
        and faiss_index is not None
    ):

        try:

            results = search_mitre_engine(
                query,
                embedding_model,
                faiss_index,
                mitre_data
            )

            if results:

                return results

        except Exception:

            pass

    # Fallback to direct JSON search
    return local_mitre_search(
        query
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk(
    alert,
    mitre_results
):

    text = alert.lower()

    risk = 15

    keywords = [

        "powershell",
        "cmd.exe",
        "phishing",
        "credential",
        "brute force",
        "failed login",
        "malware",
        "ransomware",
        "exfiltration",
        "privilege escalation",
        "successful login",
        "remote access",
        "rdp",
        "ssh",
        "suspicious",
        "attack"
    ]

    for keyword in keywords:

        if keyword in text:

            risk += 5

    if mitre_results:

        risk += 10

    return min(
        risk,
        95
    )


# ============================================================
# SEVERITY
# ============================================================

def get_severity(
    risk
):

    if risk >= 75:

        return "CRITICAL"

    if risk >= 50:

        return "HIGH"

    if risk >= 25:

        return "MEDIUM"

    return "LOW"


# ============================================================
# ATTACK DESCRIPTION
# ============================================================

def get_description(
    alert
):

    text = alert.lower()

    if (
        "powershell" in text
        or "cmd.exe" in text
        or "command execution" in text
    ):

        return (
            "Possible command execution activity was detected. "
            "Review command-line arguments, parent processes, "
            "user accounts and endpoint telemetry."
        )

    if (
        "phishing" in text
        or "malicious email" in text
    ):

        return (
            "Possible phishing activity was detected. "
            "Review sender information, URLs, attachments "
            "and affected recipients."
        )

    if (
        "brute force" in text
        or "failed login" in text
        or "password guessing" in text
    ):

        return (
            "Repeated authentication attempts may indicate "
            "brute-force activity. Review authentication logs "
            "and successful login events."
        )

    if (
        "credential" in text
        or "password" in text
    ):

        return (
            "Possible credential-related activity was detected. "
            "Review authentication events, account activity "
            "and unusual access."
        )

    if (
        "malware" in text
        or "ransomware" in text
        or "trojan" in text
    ):

        return (
            "Potential malicious software activity was detected. "
            "Investigate affected endpoints, processes, files "
            "and network connections."
        )

    if (
        "exfiltration" in text
        or "data theft" in text
    ):

        return (
            "Possible unauthorized data transfer was detected. "
            "Review the source, destination, volume and timing."
        )

    if (
        "rdp" in text
        or "remote desktop" in text
        or "ssh" in text
    ):

        return (
            "Unexpected remote access activity was detected. "
            "Validate the source IP, account and authorization."
        )

    return (
        "Potentially suspicious activity was detected. "
        "Correlate this alert with endpoint, authentication "
        "and network telemetry."
    )


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

def get_actions(
    alert
):

    text = alert.lower()

    actions = [

        "Review related security logs.",

        "Identify affected hosts and users.",

        "Correlate endpoint and network telemetry.",

        "Check whether the activity is expected."
    ]

    if (
        "powershell" in text
        or "command" in text
    ):

        actions.append(
            "Review command lines and parent-child processes."
        )

    if (
        "phishing" in text
        or "email" in text
    ):

        actions.append(
            "Inspect URLs, attachments and affected recipients."
        )

    if (
        "login" in text
        or "credential" in text
    ):

        actions.append(
            "Validate authentication and account activity."
        )

    return actions


# ============================================================
# REPORT GENERATOR
# ============================================================

def create_report(
    result
):

    lines = [

        "TRIAGEMIND AI SECURITY REPORT",

        "=" * 45,

        "",

        "Incident ID: "
        + result["incident_id"],

        "Time: "
        + result["time"],

        "Risk Score: "
        + str(result["risk"])
        + "/100",

        "Severity: "
        + result["severity"],

        "",

        "SECURITY ALERT",

        "-" * 25,

        result["alert"],

        "",

        "ATTACK DESCRIPTION",

        "-" * 25,

        result["description"],

        "",

        "MITRE ATT&CK",

        "-" * 25
    ]

    if result["mitre"]:

        for item in result["mitre"]:

            lines.append(
                str(
                    item.get(
                        "id",
                        ""
                    )
                )
                + " - "
                + str(
                    item.get(
                        "name",
                        ""
                    )
                )
            )

    else:

        lines.append(
            "No automatic MITRE match found."
        )

    lines.extend(
        [
            "",
            "RECOMMENDED ACTIONS",
            "-" * 25
        ]
    )

    for action in result["actions"]:

        lines.append(
            "- " + action
        )

    return "\n".join(
        lines
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ TriageMind AI</h1>
        <p>Intelligent Cybersecurity Alert Triage Platform</p>
        <p>
        Machine Learning • Network Detection • MITRE ATT&CK • SOC Investigation
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🛡️ TriageMind"
    )

    st.caption(
        "SOC Analyst Workspace"
    )

    st.divider()

    st.subheader(
        "System Status"
    )

    if classifier is not None:

        st.success(
            "ML Model Ready"
        )

    else:

        st.error(
            "ML Model Unavailable"
        )

    if feature_columns:

        st.success(
            str(
                len(feature_columns)
            )
            + " Model Features"
        )

    else:

        st.warning(
            "Feature columns unavailable"
        )

    if mitre_data:

        st.success(
            str(
                len(mitre_data)
            )
            + " MITRE Objects"
        )

    else:

        st.warning(
            "MITRE data unavailable"
        )

    if faiss_index is not None:

        st.success(
            "FAISS Ready"
        )

    else:

        st.info(
            "FAISS unavailable"
        )


# ============================================================
# FIVE TABS
# ============================================================

dashboard_tab, triage_tab, network_tab, mitre_tab, investigation_tab = st.tabs(
    [
        "🏠 Overview",
        "🚨 Alert Triage",
        "📊 Network Detection",
        "🎯 MITRE ATT&CK",
        "🛡️ Investigation"
    ]
)


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with dashboard_tab:

    st.header(
        "Security Operations Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Investigations",
        len(
            st.session_state.history
        )
    )

    col2.metric(
        "MITRE Objects",
        len(
            mitre_data
        )
    )

    col3.metric(
        "Model Features",
        len(
            feature_columns
        )
        if feature_columns
        else 0
    )

    col4.metric(
        "ML Status",
        "READY"
        if classifier is not None
        else "OFFLINE"
    )

    st.divider()

    st.subheader(
        "Detection Pipeline"
    )

    pipeline = pd.DataFrame(
        {
            "Stage": [

                "Network Data",

                "ML Detection",

                "MITRE Enrichment",

                "Risk Assessment",

                "SOC Investigation"
            ],

            "Status": [

                "Ready"
                if classifier
                else "Unavailable",

                "Ready"
                if classifier
                else "Unavailable",

                "Ready"
                if mitre_data
                else "Unavailable",

                "Ready",

                "Ready"
            ]
        }
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Recent Investigations"
    )

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(
                st.session_state.history
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No investigations yet."
        )


# ============================================================
# TAB 2 - ALERT TRIAGE
# ============================================================

with triage_tab:

    st.header(
        "🚨 Alert Triage"
    )

    st.caption(
        "Analyze a security alert and enrich it with MITRE ATT&CK."
    )

    alert = st.text_area(
        "Security Alert",
        height=160,
        placeholder=(
            "Example: Multiple failed login attempts "
            "from an external IP followed by a successful login."
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        source_ip = st.text_input(
            "Source IP"
        )

    with col2:

        destination_ip = st.text_input(
            "Destination IP"
        )

    analyst_notes = st.text_area(
        "Analyst Notes"
    )

    analyze_button = st.button(
        "🔍 Analyze Alert",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not alert.strip():

            st.warning(
                "Please enter a security alert."
            )

        else:

            with st.spinner(
                "Analyzing alert..."
            ):

                mitre_matches = search_mitre(
                    alert
                )

            risk = calculate_risk(
                alert,
                mitre_matches
            )

            severity = get_severity(
                risk
            )

            incident_id = (
                "TM-"
                + datetime.now().strftime(
                    "%Y%m%d%H%M%S"
                )
            )

            result = {

                "incident_id":
                    incident_id,

                "time":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "alert":
                    alert,

                "source_ip":
                    source_ip,

                "destination_ip":
                    destination_ip,

                "notes":
                    analyst_notes,

                "risk":
                    risk,

                "severity":
                    severity,

                "description":
                    get_description(
                        alert
                    ),

                "mitre":
                    mitre_matches,

                "actions":
                    get_actions(
                        alert
                    )
            }

            st.session_state.last_alert = result

            st.session_state.history.append(
                {
                    "Incident ID":
                        incident_id,

                    "Time":
                        result["time"],

                    "Risk":
                        str(risk)
                        + "/100",

                    "Severity":
                        severity,

                    "MITRE Matches":
                        len(
                            mitre_matches
                        )
                }
            )

    result = st.session_state.last_alert

    if result:

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Risk Score",
            str(
                result["risk"]
            )
            + "/100"
        )

        col2.metric(
            "Severity",
            result["severity"]
        )

        col3.metric(
            "MITRE Matches",
            len(
                result["mitre"]
            )
        )

        st.progress(
            result["risk"] / 100
        )

        st.subheader(
            "📖 Attack Description"
        )

        st.info(
            result["description"]
        )

        st.subheader(
            "🎯 MITRE ATT&CK"
        )

        if result["mitre"]:

            for item in result["mitre"]:

                technique_id = str(
                    item.get(
                        "id",
                        ""
                    )
                )

                technique_name = str(
                    item.get(
                        "name",
                        "Unknown"
                    )
                )

                with st.expander(
                    technique_id
                    + " — "
                    + technique_name
                ):

                    st.write(
                        item.get(
                            "description",
                            "No description available."
                        )
                    )

        else:

            st.info(
                "No automatic MITRE match found."
            )

        st.subheader(
            "🛠️ Recommended Investigation"
        )

        for action in result["actions"]:

            st.write(
                "• "
                + action
            )

        report = create_report(
            result
        )

        st.download_button(
            "⬇️ Download Security Report",
            report,
            result["incident_id"]
            + ".txt",
            "text/plain",
            use_container_width=True
        )


# ============================================================
# TAB 3 - NETWORK DETECTION
# ============================================================

with network_tab:

    st.header(
        "📊 Network Detection"
    )

    st.caption(
        "Upload network traffic data and run the trained classifier."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin."
        )

    else:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Records",
                len(df)
            )

            col2.metric(
                "Columns",
                len(df.columns)
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            if classifier is None:

                st.error(
                    "ML classifier is unavailable."
                )

            elif classify_csv is None:

                st.error(
                    "classify_csv() was not found in triagemind_engine.py."
                )

            else:

                detection_button = st.button(
                    "🤖 Run ML Detection",
                    type="primary",
                    use_container_width=True
                )

                if detection_button:

                    temp_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".csv"
                        ) as temp_file:

                            uploaded_file.seek(
                                0
                            )

                            temp_file.write(
                                uploaded_file.getbuffer()
                            )

                            temp_path = (
                                temp_file.name
                            )

                        with st.spinner(
                            "Running ML detection..."
                        ):

                            predictions = classify_csv(
                                temp_path,
                                classifier,
                                feature_columns
                            )

                        predictions = list(
                            predictions
                        )

                        if len(
                            predictions
                        ) != len(df):

                            st.error(
                                "The number of predictions does not match the number of rows."
                            )

                        else:

                            result_df = df.copy()

                            result_df[
                                "TriageMind Prediction"
                            ] = [

                                "Attack"
                                if int(value) == 1
                                else "Normal"

                                for value
                                in predictions
                            ]

                            attack_count = sum(
                                int(value) == 1
                                for value
                                in predictions
                            )

                            normal_count = (
                                len(
                                    predictions
                                )
                                - attack_count
                            )

                            if len(
                                predictions
                            ) > 0:

                                attack_rate = (
                                    attack_count
                                    /
                                    len(
                                        predictions
                                    )
                                    * 100
                                )

                            else:

                                attack_rate = 0

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Attack",
                                attack_count
                            )

                            col2.metric(
                                "Normal",
                                normal_count
                            )

                            col3.metric(
                                "Attack Rate",
                                f"{attack_rate:.2f}%"
                            )

                            st.subheader(
                                "Detection Results"
                            )

                            st.dataframe(
                                result_df.head(
                                    100
                                ),
                                use_container_width=True
                            )

                            csv_data = (
                                result_df
                                .to_csv(
                                    index=False
                                )
                                .encode(
                                    "utf-8"
                                )
                            )

                            st.download_button(
                                "⬇️ Download Predictions",
                                csv_data,
                                "triagemind_predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )

                            st.session_state.last_network = (
                                result_df
                            )

                    except Exception as error:

                        st.error(
                            "Network detection failed."
                        )

                        st.code(
                            str(error)
                        )

                    finally:

                        if (
                            temp_path
                            and
                            os.path.exists(
                                temp_path
                            )
                        ):

                            try:

                                os.remove(
                                    temp_path
                                )

                            except Exception:

                                pass

        except Exception as error:

            st.error(
                "Could not read the CSV file."
            )

            st.code(
                str(error)
            )


# ============================================================
# TAB 4 - MITRE ATT&CK
# ============================================================

with mitre_tab:

    st.header(
        "🎯 MITRE ATT&CK Explorer"
    )

    st.caption(
        "Search the local MITRE ATT&CK knowledge base."
    )

    mitre_query = st.text_input(
        "Search Technique",
        placeholder=(
            "PowerShell, phishing, credential, brute force..."
        )
    )

    mitre_button = st.button(
        "🔎 Search ATT&CK",
        type="primary"
    )

    if mitre_button:

        if not mitre_query.strip():

            st.warning(
                "Enter a search term."
            )

        else:

            results = search_mitre(
                mitre_query
            )

            if not results:

                st.warning(
                    "No matching technique found."
                )

            else:

                for item in results:

                    technique_id = str(
                        item.get(
                            "id",
                            ""
                        )
                    )

                    technique_name = str(
                        item.get(
                            "name",
                            "Unknown"
                        )
                    )

                    with st.expander(
                        technique_id
                        + " — "
                        + technique_name
                    ):

                        st.write(
                            item.get(
                                "description",
                                "No description available."
                            )
                        )


# ============================================================
# TAB 5 - INVESTIGATION
# ============================================================

with investigation_tab:

    st.header(
        "🛡️ Investigation Workspace"
    )

    result = st.session_state.last_alert

    if result is None:

        st.info(
            "Analyze an alert in the Alert Triage tab first."
        )

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Risk",
            str(
                result["risk"]
            )
            + "/100"
        )

        col2.metric(
            "Severity",
            result["severity"]
        )

        col3.metric(
            "MITRE Matches",
            len(
                result["mitre"]
            )
        )

        st.subheader(
            "Latest Alert"
        )

        st.write(
            result["alert"]
        )

        st.subheader(
            "Recommended Actions"
        )

        for action in result["actions"]:

            st.write(
                "• "
                + action
            )

        report = create_report(
            result
        )

        st.download_button(
            "⬇️ Download Investigation Report",
            report,
            result["incident_id"]
            + "_investigation.txt",
            "text/plain"
        )

    st.divider()

    st.subheader(
        "Investigation History"
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        history_csv = (
            history_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "⬇️ Download History",
            history_csv,
            "triagemind_history.csv",
            "text/csv"
        )

    else:

        st.info(
            "No investigation history yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TriageMind AI • Machine Learning • "
    "Network Detection • MITRE ATT&CK • "
    "SOC Alert Triage"
)
