import os
import json
import pickle
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TriageMind AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
MITRE_DIR = os.path.join(BASE_DIR, "mitre_data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "classifier.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0b1120;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827,
            #172554
        );
        border: 1px solid #334155;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        font-size: 40px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #cbd5e1;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(CLASSIFIER_PATH):
        return None

    try:

        with open(
            CLASSIFIER_PATH,
            "rb"
        ) as f:

            return pickle.load(f)

    except Exception:

        return None


MODEL = load_model()


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

@st.cache_resource
def load_features():

    if not os.path.exists(FEATURE_PATH):
        return []

    try:

        with open(
            FEATURE_PATH,
            "rb"
        ) as f:

            data = pickle.load(f)

        if isinstance(data, pd.DataFrame):

            return list(data.columns)

        if isinstance(data, np.ndarray):

            return list(data)

        return list(data)

    except Exception:

        return []


FEATURE_COLUMNS = load_features()


# ============================================================
# LOAD MITRE ATT&CK
# ============================================================

@st.cache_data
def load_mitre():

    filenames = [
        "enterprise-attack.json",
        "enterprise-attack-19.2.json",
        "mitre_techniques.json"
    ]

    for filename in filenames:

        path = os.path.join(
            MITRE_DIR,
            filename
        )

        if not os.path.exists(path):
            continue

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if isinstance(data, dict):

                if "objects" in data:
                    return data["objects"]

                if "techniques" in data:
                    return data["techniques"]

            if isinstance(data, list):
                return data

        except Exception:

            return []

    return []


MITRE_DATA = load_mitre()


# ============================================================
# MITRE SEARCH
# ============================================================

def search_mitre(query):

    if not MITRE_DATA:
        return []

    query = query.lower()

    results = []

    for item in MITRE_DATA:

        if not isinstance(item, dict):
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

        combined = (
            name
            + " "
            + description
            + " "
            + technique_id
        ).lower()

        if query in combined:

            results.append(
                {
                    "id": technique_id,
                    "name": name,
                    "description": description
                }
            )

    return results[:10]


# ============================================================
# ALERT → MITRE
# ============================================================

def map_alert_to_mitre(alert):

    keywords = [
        "phishing",
        "powershell",
        "credential",
        "brute force",
        "failed login",
        "remote desktop",
        "rdp",
        "ssh",
        "scheduled task",
        "command",
        "script",
        "exfiltration",
        "ransomware",
        "malware"
    ]

    results = []

    text = alert.lower()

    for keyword in keywords:

        if keyword in text:

            matches = search_mitre(
                keyword
            )

            results.extend(
                matches
            )

    unique = {}

    for result in results:

        key = (
            result["id"]
            + result["name"]
        )

        unique[key] = result

    return list(
        unique.values()
    )[:10]


# ============================================================
# ATTACK DESCRIPTION
# ============================================================

ATTACK_DESCRIPTIONS = {

    "Brute Force":
        (
            "A brute-force attack attempts to gain access "
            "to an account or service by repeatedly trying "
            "different authentication values."
        ),

    "Credential Attack":
        (
            "A credential attack attempts to obtain, guess "
            "or abuse usernames, passwords or authentication "
            "tokens."
        ),

    "Phishing":
        (
            "Phishing uses deceptive emails, messages, "
            "websites or attachments to trick users into "
            "revealing information or executing malicious "
            "content."
        ),

    "PowerShell / Command Execution":
        (
            "PowerShell and command execution can be used "
            "for legitimate administration but may also "
            "appear during malicious activity."
        ),

    "Malware":
        (
            "Malware is software designed to perform "
            "unauthorized or harmful actions on a system."
        ),

    "Data Exfiltration":
        (
            "Data exfiltration involves unauthorized "
            "transfer of information from an environment "
            "to an external location."
        ),

    "Remote Access":
        (
            "Remote access involves connecting to systems "
            "using services such as RDP or SSH. Unexpected "
            "remote access should be investigated."
        ),

    "Suspicious Activity":
        (
            "The alert contains suspicious indicators but "
            "does not provide enough information to classify "
            "a specific attack type."
        )
}


# ============================================================
# DETECT ATTACK TYPE
# ============================================================

def detect_attack(alert):

    text = alert.lower()

    categories = {

        "Brute Force": [
            "brute force",
            "failed login",
            "password guessing",
            "multiple login"
        ],

        "Credential Attack": [
            "credential",
            "password",
            "credential dumping"
        ],

        "Phishing": [
            "phishing",
            "malicious email",
            "suspicious email"
        ],

        "PowerShell / Command Execution": [
            "powershell",
            "command execution",
            "script",
            "cmd.exe"
        ],

        "Malware": [
            "malware",
            "trojan",
            "ransomware",
            "malicious file"
        ],

        "Data Exfiltration": [
            "exfiltration",
            "data theft",
            "stolen data"
        ],

        "Remote Access": [
            "rdp",
            "remote desktop",
            "ssh",
            "remote access"
        ]
    }

    matched = []

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword in text:

                matched.append(
                    keyword
                )

                return category, matched

    return "Suspicious Activity", matched


# ============================================================
# RISK
# ============================================================

def calculate_risk(
    attack_type,
    alert
):

    risk = 20

    if attack_type != "Suspicious Activity":
        risk += 30

    high_risk_words = [
        "ransomware",
        "exfiltration",
        "credential dumping",
        "successful login",
        "privilege escalation",
        "malware"
    ]

    for word in high_risk_words:

        if word in alert.lower():

            risk += 10

    return min(
        risk,
        95
    )


# ============================================================
# SEVERITY
# ============================================================

def get_severity(risk):

    if risk >= 85:
        return "CRITICAL"

    if risk >= 65:
        return "HIGH"

    if risk >= 35:
        return "MEDIUM"

    return "LOW"


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

def recommended_actions(
    attack_type
):

    actions = {

        "Brute Force": [
            "Review authentication logs.",
            "Check whether the source IP is known.",
            "Investigate successful authentication events.",
            "Consider account protection measures."
        ],

        "Credential Attack": [
            "Review account authentication history.",
            "Check for credential misuse.",
            "Inspect endpoint activity.",
            "Investigate unusual login locations."
        ],

        "Phishing": [
            "Inspect the original email.",
            "Analyze URLs and attachments.",
            "Check other recipients.",
            "Inspect affected endpoints."
        ],

        "PowerShell / Command Execution": [
            "Review PowerShell command lines.",
            "Inspect parent-child processes.",
            "Review endpoint telemetry.",
            "Check for persistence."
        ],

        "Malware": [
            "Isolate the affected endpoint if appropriate.",
            "Review suspicious processes.",
            "Inspect files and persistence.",
            "Check related network connections."
        ],

        "Data Exfiltration": [
            "Identify transferred data.",
            "Review destination infrastructure.",
            "Check transfer volume and timing.",
            "Investigate affected accounts and hosts."
        ],

        "Remote Access": [
            "Review remote authentication logs.",
            "Verify the source IP.",
            "Check whether remote access was authorized.",
            "Inspect endpoint activity."
        ],

        "Suspicious Activity": [
            "Review the alert in additional telemetry.",
            "Identify source and destination systems.",
            "Check related events.",
            "Document analyst findings."
        ]
    }

    return actions.get(
        attack_type,
        actions["Suspicious Activity"]
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
        ML Detection • Risk Scoring • MITRE ATT&CK •
        SOC Investigation
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ TriageMind")

    st.divider()

    if MODEL:

        st.success(
            "ML Model: READY"
        )

    else:

        st.error(
            "ML Model: NOT FOUND"
        )

    if FEATURE_COLUMNS:

        st.success(
            f"Features: {len(FEATURE_COLUMNS)}"
        )

    else:

        st.error(
            "Feature file missing"
        )

    if MITRE_DATA:

        st.success(
            f"MITRE: {len(MITRE_DATA)} objects"
        )

    else:

        st.warning(
            "MITRE data unavailable"
        )


# ============================================================
# IMPORTANT:
# DEFINE ALL TABS BEFORE USING THEM
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
# DASHBOARD TAB
# ============================================================

with dashboard_tab:

    st.header(
        "🏠 Security Dashboard"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Investigations",
            len(
                st.session_state.history
            )
        )

    with c2:

        st.metric(
            "MITRE Objects",
            len(MITRE_DATA)
        )

    with c3:

        st.metric(
            "ML Features",
            len(FEATURE_COLUMNS)
        )

    with c4:

        st.metric(
            "ML Status",
            "ONLINE"
            if MODEL
            else "OFFLINE"
        )

    st.divider()

    st.subheader(
        "How TriageMind Works"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.info(
            "1\n\n"
            "Alert / Network Data"
        )

    with b:

        st.info(
            "2\n\n"
            "ML Detection"
        )

    with c:

        st.info(
            "3\n\n"
            "MITRE Enrichment"
        )

    with d:

        st.info(
            "4\n\n"
            "SOC Investigation"
        )


# ============================================================
# TRIAGE TAB
# ============================================================

with triage_tab:

    st.header(
        "🚨 Alert Triage"
    )

    st.write(
        "Enter security information below."
    )

    with st.form(
        "triage_form"
    ):

        alert = st.text_area(
            "Security Alert",
            height=160,
            placeholder=(
                "Example: Multiple failed login "
                "attempts followed by a successful "
                "login from an external IP."
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

        notes = st.text_area(
            "Analyst Notes"
        )

        submit = st.form_submit_button(
            "🔍 Analyze Alert",
            type="primary",
            use_container_width=True
        )

    if submit:

        if not alert.strip():

            st.warning(
                "Please enter a security alert."
            )

        else:

            incident_id = (
                "TM-"
                + datetime.now().strftime(
                    "%Y%m%d"
                )
                + "-"
                + uuid.uuid4().hex[:6].upper()
            )

            attack_type, keywords = detect_attack(
                alert
            )

            description = ATTACK_DESCRIPTIONS[
                attack_type
            ]

            risk = calculate_risk(
                attack_type,
                alert
            )

            severity = get_severity(
                risk
            )

            mitre_results = map_alert_to_mitre(
                alert
            )

            actions = recommended_actions(
                attack_type
            )

            case = {

                "Incident ID":
                    incident_id,

                "Time":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Attack Type":
                    attack_type,

                "Risk":
                    risk,

                "Severity":
                    severity,

                "MITRE Matches":
                    len(mitre_results)
            }

            st.session_state.history.append(
                case
            )

            st.session_state.last_result = {

                "incident_id":
                    incident_id,

                "alert":
                    alert,

                "source_ip":
                    source_ip,

                "destination_ip":
                    destination_ip,

                "notes":
                    notes,

                "attack_type":
                    attack_type,

                "description":
                    description,

                "risk":
                    risk,

                "severity":
                    severity,

                "mitre":
                    mitre_results,

                "actions":
                    actions
            }

            st.success(
                f"Incident {incident_id} created."
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Attack Type",
                    attack_type
                )

            with c2:

                st.metric(
                    "Risk",
                    f"{risk}/100"
                )

            with c3:

                st.metric(
                    "Severity",
                    severity
                )

            st.divider()

            # ------------------------------------------------
            # DESCRIPTION
            # ------------------------------------------------

            st.subheader(
                "📖 Attack Description"
            )

            st.info(
                description
            )

            if keywords:

                st.write(
                    "**Detected indicators:**"
                )

                st.write(
                    ", ".join(
                        keywords
                    )
                )

            # ------------------------------------------------
            # DOWNLOAD DESCRIPTION
            # ------------------------------------------------

            report = f"""
TRIAGEMIND AI
ATTACK DESCRIPTION REPORT
====================================

Incident ID:
{incident_id}

Attack Type:
{attack_type}

Risk Score:
{risk}/100

Severity:
{severity}

Source IP:
{source_ip if source_ip else "Not provided"}

Destination IP:
{destination_ip if destination_ip else "Not provided"}

------------------------------------
ATTACK DESCRIPTION
------------------------------------

{description}

------------------------------------
DETECTED INDICATORS
------------------------------------

{", ".join(keywords) if keywords else "None"}

------------------------------------
MITRE ATT&CK
------------------------------------

"""

            if mitre_results:

                for item in mitre_results:

                    report += (
                        f"{item['id']} - "
                        f"{item['name']}\n"
                    )

            else:

                report += (
                    "No automatic MITRE mapping found.\n"
                )

            report += """

------------------------------------
RECOMMENDED ACTIONS
------------------------------------

"""

            for action in actions:

                report += (
                    "• "
                    + action
                    + "\n"
                )

            report += """

------------------------------------
ANALYST NOTES
------------------------------------

"""

            report += (
                notes
                if notes
                else "No analyst notes."
            )

            st.download_button(
                "⬇️ Download Attack Description",
                report,
                f"{incident_id}_attack_report.txt",
                "text/plain",
                use_container_width=True
            )

            # ------------------------------------------------
            # MITRE
            # ------------------------------------------------

            st.subheader(
                "🎯 MITRE ATT&CK Mapping"
            )

            if mitre_results:

                for item in mitre_results:

                    with st.expander(
                        f"{item['id']} - "
                        f"{item['name']}"
                    ):

                        st.write(
                            item["description"]
                        )

            else:

                st.info(
                    "No automatic MITRE mapping found."
                )

            # ------------------------------------------------
            # ACTIONS
            # ------------------------------------------------

            st.subheader(
                "🛠️ Recommended Actions"
            )

            for action in actions:

                st.write(
                    "• " + action
                )


# ============================================================
# NETWORK DETECTION
# ============================================================

with network_tab:

    st.header(
        "📊 Network ML Detection"
    )

    st.write(
        "Upload network traffic data for prediction "
        "using your trained classifier."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded:

        try:

            df = pd.read_csv(
                uploaded
            )

            st.success(
                f"{len(df)} records loaded."
            )

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            if MODEL and FEATURE_COLUMNS:

                missing = [
                    col
                    for col in FEATURE_COLUMNS
                    if col not in df.columns
                ]

                if missing:

                    st.warning(
                        f"{len(missing)} model features "
                        "are missing from the uploaded CSV."
                    )

                    with st.expander(
                        "View missing features"
                    ):

                        st.write(
                            missing
                        )

                if st.button(
                    "🤖 Run ML Detection",
                    type="primary"
                ):

                    prediction_df = df.copy()

                    for col in FEATURE_COLUMNS:

                        if col not in prediction_df.columns:

                            prediction_df[col] = 0

                    prediction_df = prediction_df[
                        FEATURE_COLUMNS
                    ]

                    for col in prediction_df.columns:

                        if prediction_df[
                            col
                        ].dtype == "object":

                            prediction_df[
                                col
                            ] = (
                                prediction_df[
                                    col
                                ]
                                .astype("category")
                                .cat.codes
                            )

                    prediction_df = (
                        prediction_df
                        .replace(
                            [np.inf, -np.inf],
                            np.nan
                        )
                        .fillna(0)
                    )

                    try:

                        predictions = MODEL.predict(
                            prediction_df
                        )

                        results = df.copy()

                        results[
                            "TriageMind Prediction"
                        ] = predictions

                        st.success(
                            "Prediction completed."
                        )

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        csv = results.to_csv(
                            index=False
                        ).encode(
                            "utf-8"
                        )

                        st.download_button(
                            "⬇️ Download Results",
                            csv,
                            "triagemind_results.csv",
                            "text/csv"
                        )

                    except Exception as e:

                        st.error(
                            "Prediction failed."
                        )

                        st.code(
                            str(e)
                        )

        except Exception as e:

            st.error(
                f"Could not read CSV: {e}"
            )

    else:

        st.info(
            "Upload a CSV containing network features."
        )


# ============================================================
# MITRE TAB
# ============================================================

with mitre_tab:

    st.header(
        "🎯 MITRE ATT&CK Explorer"
    )

    query = st.text_input(
        "Search MITRE ATT&CK",
        placeholder="powershell, phishing, credential..."
    )

    if st.button(
        "🔎 Search"
    ):

        if query:

            results = search_mitre(
                query
            )

            if results:

                for result in results:

                    with st.expander(
                        f"{result['id']} - "
                        f"{result['name']}"
                    ):

                        st.write(
                            result["description"]
                        )

            else:

                st.warning(
                    "No matching technique found."
                )

        else:

            st.warning(
                "Enter a search term."
            )


# ============================================================
# SOC TAB
# ============================================================

with soc_tab:

    st.header(
        "🛡️ SOC Investigation"
    )

    result = st.session_state.last_result

    if result:

        st.subheader(
            result["incident_id"]
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Attack",
                result["attack_type"]
            )

        with c2:

            st.metric(
                "Risk",
                f"{result['risk']}/100"
            )

        with c3:

            st.metric(
                "Severity",
                result["severity"]
            )

        st.subheader(
            "Alert"
        )

        st.write(
            result["alert"]
        )

        st.subheader(
            "Attack Description"
        )

        st.info(
            result["description"]
        )

        st.subheader(
            "Recommended Actions"
        )

        for action in result["actions"]:

            st.write(
                "• " + action
            )

        st.subheader(
            "MITRE ATT&CK"
        )

        if result["mitre"]:

            for item in result["mitre"]:

                st.write(
                    f"**{item['id']}** - "
                    f"{item['name']}"
                )

        else:

            st.info(
                "No MITRE mapping."
            )

    else:

        st.info(
            "Run an alert investigation first."
        )

    st.divider()

    st.subheader(
        "Investigation History"
    )

    if st.session_state.history:

        history = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No investigations yet."
        )


# ============================================================
# DIAGNOSTICS
# ============================================================

with diagnostics_tab:

    st.header(
        "⚙️ Diagnostics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Model:**"
        )

        if MODEL:

            st.success(
                "classifier.pkl loaded"
            )

        else:

            st.error(
                "classifier.pkl missing"
            )

        st.write(
            "**Feature file:**"
        )

        if FEATURE_COLUMNS:

            st.success(
                f"{len(FEATURE_COLUMNS)} features loaded"
            )

        else:

            st.error(
                "feature_columns.pkl missing"
            )

    with col2:

        st.write(
            "**MITRE:**"
        )

        if MITRE_DATA:

            st.success(
                f"{len(MITRE_DATA)} objects loaded"
            )

        else:

            st.error(
                "MITRE data missing"
            )

        st.write(
            "**Project directory:**"
        )

        st.code(
            BASE_DIR
        )

    st.divider()

    st.subheader(
        "Required Files"
    )

    required_files = [

        CLASSIFIER_PATH,

        FEATURE_PATH,

        os.path.join(
            MITRE_DIR,
            "enterprise-attack.json"
        )
    ]

    for path in required_files:

        if os.path.exists(path):

            st.success(
                "✓ " + path
            )

        else:

            st.warning(
                "Missing: " + path
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TriageMind AI | ML Detection | "
    "MITRE ATT&CK | SOC Investigation"
)
