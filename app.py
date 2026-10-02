import os
import re
import json
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# TRIAGEMIND AI
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

MITRE_FILES = [
    os.path.join(
        MITRE_DIR,
        "enterprise-attack.json"
    ),
    os.path.join(
        MITRE_DIR,
        "enterprise-attack-19.2.json"
    ),
    os.path.join(
        MITRE_DIR,
        "mitre_techniques.json"
    )
]

DB_PATH = os.path.join(
    BASE_DIR,
    "triagemind_history.db"
)


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background:
        linear-gradient(
            135deg,
            #111827,
            #1f2937
        );
        border: 1px solid #374151;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin-bottom: 5px;
    }

    .attack-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #374151;
        background: #111827;
        margin-bottom: 12px;
    }

    .small-text {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATABASE
# ============================================================

def init_database():

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investigations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,

            prediction TEXT,

            probability REAL,

            severity TEXT,

            risk_score INTEGER,

            mitre_ids TEXT,

            analyst TEXT,

            notes TEXT,

            source TEXT

        )
        """
    )

    conn.commit()
    conn.close()


def save_investigation(
    result,
    analyst,
    notes,
    source
):

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO investigations
        (
            timestamp,
            prediction,
            probability,
            severity,
            risk_score,
            mitre_ids,
            analyst,
            notes,
            source
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(
                timespec="seconds"
            ),

            result.get(
                "prediction",
                ""
            ),

            float(
                result.get(
                    "probability",
                    0
                )
            ),

            result.get(
                "severity",
                ""
            ),

            int(
                result.get(
                    "risk_score",
                    0
                )
            ),

            json.dumps(
                result.get(
                    "mitre_ids",
                    []
                )
            ),

            analyst,

            notes,

            source
        )
    )

    conn.commit()
    conn.close()


def load_history():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT *
        FROM investigations
        ORDER BY id DESC
        LIMIT 100
        """,
        conn
    )

    conn.close()

    return df


init_database()


# ============================================================
# INPUT SANITIZATION
# ============================================================

def sanitize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        " ",
        text
    )

    dangerous_patterns = [

        r"ignore\s+(all\s+)?previous\s+instructions",

        r"disregard\s+(all\s+)?previous",

        r"system\s+prompt",

        r"developer\s+message",

        r"reveal\s+your\s+instructions",

        r"jailbreak"

    ]

    for pattern in dangerous_patterns:

        text = re.sub(
            pattern,
            "[SANITIZED]",
            text,
            flags=re.IGNORECASE
        )

    return text[:5000]


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(
        CLASSIFIER_PATH
    ):
        return None, None, (
            "classifier.pkl not found"
        )

    if not os.path.exists(
        FEATURE_PATH
    ):
        return None, None, (
            "feature_columns.pkl not found"
        )

    try:

        with open(
            CLASSIFIER_PATH,
            "rb"
        ) as file:

            classifier = pickle.load(
                file
            )

        with open(
            FEATURE_PATH,
            "rb"
        ) as file:

            feature_columns = pickle.load(
                file
            )

        if isinstance(
            feature_columns,
            np.ndarray
        ):

            feature_columns = (
                feature_columns.tolist()
            )

        if isinstance(
            feature_columns,
            dict
        ):

            for key in [
                "feature_columns",
                "features",
                "columns",
                "feature_names"
            ]:

                if key in feature_columns:

                    feature_columns = (
                        feature_columns[key]
                    )

                    break

        feature_columns = list(
            feature_columns
        )

        return (
            classifier,
            feature_columns,
            None
        )

    except Exception as error:

        return (
            None,
            None,
            str(error)
        )


classifier, feature_columns, model_error = (
    load_model()
)


# ============================================================
# MITRE ATT&CK LOADING
# ============================================================

def find_mitre_file():

    for file_path in MITRE_FILES:

        if os.path.exists(
            file_path
        ):

            return file_path

    return None


@st.cache_data
def load_mitre():

    file_path = find_mitre_file()

    if file_path is None:

        return []

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        objects = data.get(
            "objects",
            []
        )

        techniques = []

        for obj in objects:

            if obj.get(
                "type"
            ) != "attack-pattern":

                continue

            if obj.get(
                "revoked",
                False
            ):

                continue

            if obj.get(
                "x_mitre_deprecated",
                False
            ):

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

            if not technique_id:

                continue

            if not technique_id.startswith(
                "T"
            ):

                continue

            phases = []

            for phase in obj.get(
                "kill_chain_phases",
                []
            ):

                phases.append(
                    phase.get(
                        "phase_name",
                        ""
                    )
                )

            techniques.append(
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

                    "platforms": obj.get(
                        "x_mitre_platforms",
                        []
                    ),

                    "phases": phases
                }
            )

        return techniques

    except Exception:

        return []


MITRE_TECHNIQUES = load_mitre()


# ============================================================
# MITRE SEARCH
# ============================================================

def tokenize(text):

    return set(
        re.findall(
            r"[a-zA-Z0-9_-]+",
            str(text).lower()
        )
    )


def similarity_score(
    query,
    document
):

    query_tokens = tokenize(
        query
    )

    document_tokens = tokenize(
        document
    )

    if not query_tokens:
        return 0

    if not document_tokens:
        return 0

    common = query_tokens.intersection(
        document_tokens
    )

    return (
        len(common)
        /
        max(
            1,
            len(query_tokens)
        )
    )


def search_mitre(
    query,
    top_k=3
):

    query = sanitize_text(
        query
    )

    if not query:

        return []

    results = []

    query_lower = query.lower()

    for technique in MITRE_TECHNIQUES:

        searchable_text = " ".join(
            [
                technique["id"],

                technique["name"],

                technique["description"],

                " ".join(
                    technique["phases"]
                ),

                " ".join(
                    technique["platforms"]
                )
            ]
        )

        score = similarity_score(
            query,
            searchable_text
        )

        technique_id = (
            technique["id"].lower()
        )

        technique_name = (
            technique["name"].lower()
        )

        # Exact MITRE ID boost
        if technique_id in query_lower:

            score += 3.0

        # Exact technique-name boost
        if technique_name in query_lower:

            score += 2.0

        # Useful security keyword boosts
        keywords = {
            "powershell": [
                "powershell",
                "command",
                "scripting"
            ],

            "credential": [
                "credential",
                "password",
                "credential dumping"
            ],

            "scan": [
                "scan",
                "scanning",
                "reconnaissance",
                "port"
            ],

            "network": [
                "network",
                "connection",
                "traffic"
            ],

            "execution": [
                "execute",
                "execution",
                "command"
            ]
        }

        for keyword, words in keywords.items():

            if any(
                word in query_lower
                for word in words
            ):

                if keyword in searchable_text.lower():

                    score += 0.5

        results.append(
            (
                score,
                technique
            )
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    final_results = []

    for score, technique in results:

        if score <= 0:

            continue

        item = technique.copy()

        item["similarity"] = round(
            float(score),
            4
        )

        final_results.append(
            item
        )

        if len(
            final_results
        ) >= top_k:

            break

    return final_results


# ============================================================
# EXPLICIT TOP-3 RETRIEVAL
# ============================================================

def get_top3_mitre(
    query
):

    """
    Retrieve exactly the top three
    relevant MITRE ATT&CK techniques.
    """

    results = search_mitre(
        query,
        top_k=3
    )

    return results[:3]


# ============================================================
# MITRE ID VERIFICATION
# ============================================================

def verify_mitre_ids(
    ids
):

    valid_ids = {
        technique["id"]
        for technique in MITRE_TECHNIQUES
    }

    verified = []
    fabricated = []

    for technique_id in ids:

        clean_id = str(
            technique_id
        ).strip().upper()

        if clean_id in valid_ids:

            verified.append(
                clean_id
            )

        else:

            fabricated.append(
                clean_id
            )

    total = (
        len(verified)
        +
        len(fabricated)
    )

    fabricated_rate = (

        len(fabricated)
        /
        total

        if total > 0

        else 0
    )

    return {
        "verified": verified,

        "fabricated": fabricated,

        "fabricated_rate":
            fabricated_rate
    }


# ============================================================
# SEMANTIC TELEMETRY DESCRIPTION
# ============================================================

def semantic_parse(
    data
):

    descriptions = []

    field_labels = {

        "dur":
            "connection duration",

        "sbytes":
            "source bytes",

        "dbytes":
            "destination bytes",

        "spkts":
            "source packets",

        "dpkts":
            "destination packets",

        "sttl":
            "source TTL",

        "dttl":
            "destination TTL",

        "sload":
            "source load",

        "dload":
            "destination load",

        "tcprtt":
            "TCP round-trip time",

        "synack":
            "SYN-ACK delay",

        "ackdat":
            "ACK delay",

        "ct_state_ttl":
            "state/TTL connection count",

        "ct_flw_http_mthd":
            "HTTP method count",

        "is_ftp_login":
            "FTP login indicator",

        "ct_dst_src_ltm":
            "destination-source relationship count"
    }

    for key, label in field_labels.items():

        if key not in data:

            continue

        value = data[key]

        try:

            if pd.isna(value):

                continue

        except Exception:

            pass

        descriptions.append(
            f"{label}: {value}"
        )

    if not descriptions:

        return (
            "No numerical network telemetry "
            "was available for semantic parsing."
        )

    return " | ".join(
        descriptions
    )


# ============================================================
# MODEL INPUT
# ============================================================

def prepare_model_input(
    data
):

    if not feature_columns:

        raise ValueError(
            "Feature columns are unavailable."
        )

    input_df = pd.DataFrame(
        [data]
    )

    output = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    categorical_map = {

        "tcp": 1,
        "udp": 2,
        "icmp": 3,
        "http": 4,
        "https": 5,
        "ftp": 6,
        "ssh": 7,
        "dns": 8
    }

    for column in feature_columns:

        if column not in input_df.columns:

            continue

        value = input_df.iloc[
            0
        ][column]

        try:

            if pd.isna(value):

                continue

        except Exception:

            pass

        try:

            output.loc[
                0,
                column
            ] = float(value)

        except Exception:

            output.loc[
                0,
                column
            ] = categorical_map.get(
                str(value).lower(),
                0
            )

    return output


# ============================================================
# NETWORK PREDICTION
# ============================================================

def predict_network(
    data
):

    if classifier is None:

        return {

            "prediction":
                "Unavailable",

            "probability":
                0.0,

            "severity":
                "Unknown",

            "risk_score":
                0,

            "error":
                model_error
        }

    try:

        X = prepare_model_input(
            data
        )

        prediction = classifier.predict(
            X
        )[0]

        probability = 0.0

        if hasattr(
            classifier,
            "predict_proba"
        ):

            probabilities = (
                classifier.predict_proba(
                    X
                )[0]
            )

            classes = list(
                getattr(
                    classifier,
                    "classes_",
                    []
                )
            )

            attack_indexes = []

            for index, cls in enumerate(
                classes
            ):

                label = str(
                    cls
                ).lower()

                if (
                    label in [
                        "1",
                        "attack",
                        "anomaly",
                        "malicious",
                        "true"
                    ]
                    or
                    "attack" in label
                    or
                    "malicious" in label
                    or
                    "anomaly" in label
                ):

                    attack_indexes.append(
                        index
                    )

            if attack_indexes:

                probability = max(
                    probabilities[index]
                    for index
                    in attack_indexes
                )

            elif len(
                probabilities
            ) > 1:

                probability = max(
                    probabilities
                )

            else:

                probability = float(
                    probabilities[0]
                )

        else:

            label = str(
                prediction
            ).lower()

            if label in [
                "0",
                "normal",
                "benign"
            ]:

                probability = 0.0

            else:

                probability = 1.0

        if probability >= 0.90:

            severity = "Critical"

        elif probability >= 0.70:

            severity = "High"

        elif probability >= 0.45:

            severity = "Medium"

        else:

            severity = "Low"

        risk_score = int(
            min(
                100,
                probability * 100
            )
        )

        return {

            "prediction":
                str(prediction),

            "probability":
                float(probability),

            "severity":
                severity,

            "risk_score":
                risk_score,

            "error":
                None
        }

    except Exception as error:

        return {

            "prediction":
                "Prediction error",

            "probability":
                0.0,

            "severity":
                "Unknown",

            "risk_score":
                0,

            "error":
                str(error)
        }


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    result,
    mitre_results
):

    recommendations = []

    severity = result.get(
        "severity"
    )

    if severity in [
        "Critical",
        "High"
    ]:

        recommendations.extend(
            [
                "Review the affected host immediately.",

                "Check source and destination communication logs.",

                "Review authentication and process activity.",

                "Correlate this alert with related security events."
            ]
        )

    elif severity == "Medium":

        recommendations.extend(
            [
                "Review surrounding network activity.",

                "Validate whether the traffic is expected.",

                "Correlate the event with other alerts."
            ]
        )

    else:

        recommendations.extend(
            [
                "Continue monitoring the activity.",

                "Compare the event against the normal baseline."
            ]
        )

    if mitre_results:

        recommendations.append(
            "Validate the retrieved MITRE ATT&CK "
            "techniques against the original telemetry."
        )

    return recommendations


# ============================================================
# REPORT
# ============================================================

def build_report(
    result,
    telemetry,
    mitre_results,
    verification,
    recommendations,
    analyst,
    notes
):

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "TRIAGEMIND AI - SECURITY TRIAGE REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        "Generated: "
        +
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    lines.append(
        "Analyst: "
        +
        (
            analyst
            if analyst
            else "Not specified"
        )
    )

    lines.append("")

    lines.append(
        "MODEL ASSESSMENT"
    )

    lines.append(
        "-" * 40
    )

    lines.append(
        "Prediction: "
        +
        str(
            result.get(
                "prediction"
            )
        )
    )

    lines.append(
        "Probability: "
        +
        f"{result.get('probability', 0) * 100:.2f}%"
    )

    lines.append(
        "Severity: "
        +
        str(
            result.get(
                "severity"
            )
        )
    )

    lines.append(
        "Risk Score: "
        +
        str(
            result.get(
                "risk_score"
            )
        )
        +
        "/100"
    )

    lines.append("")

    lines.append(
        "SEMANTIC TELEMETRY"
    )

    lines.append(
        "-" * 40
    )

    lines.append(
        telemetry
    )

    lines.append("")

    lines.append(
        "TOP 3 MITRE ATT&CK TECHNIQUES"
    )

    lines.append(
        "-" * 40
    )

    for rank, item in enumerate(
        mitre_results[:3],
        start=1
    ):

        lines.append(
            f"Rank #{rank}: "
            f"{item['id']} - "
            f"{item['name']}"
        )

        lines.append(
            f"Similarity: "
            f"{item.get('similarity', 0)}"
        )

        lines.append(
            "Description: "
            +
            sanitize_text(
                item.get(
                    "description",
                    ""
                )
            )[:1000]
        )

        lines.append("")

    lines.append(
        "MITRE CITATION VERIFICATION"
    )

    lines.append(
        "-" * 40
    )

    lines.append(
        "Verified IDs: "
        +
        (
            ", ".join(
                verification[
                    "verified"
                ]
            )
            or "None"
        )
    )

    lines.append(
        "Invalid/Fabricated IDs: "
        +
        (
            ", ".join(
                verification[
                    "fabricated"
                ]
            )
            or "None"
        )
    )

    lines.append(
        "Fabricated-ID Rate: "
        +
        f"{verification['fabricated_rate'] * 100:.2f}%"
    )

    lines.append("")

    lines.append(
        "RECOMMENDED ACTIONS"
    )

    lines.append(
        "-" * 40
    )

    for index, action in enumerate(
        recommendations,
        start=1
    ):

        lines.append(
            f"{index}. {action}"
        )

    lines.append("")

    lines.append(
        "ANALYST NOTES"
    )

    lines.append(
        "-" * 40
    )

    lines.append(
        notes
        if notes
        else
        "No analyst notes."
    )

    lines.append("")

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

    <h3>
    Intelligent Cybersecurity Alert Triage Platform
    </h3>

    <p>
    Machine-learning network detection combined with
    MITRE ATT&CK Top-3 technique retrieval,
    semantic telemetry analysis and SOC investigation.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Status"
    )

    if classifier is not None:

        st.success(
            "ML classifier loaded"
        )

    else:

        st.error(
            "ML classifier unavailable"
        )

    if MITRE_TECHNIQUES:

        st.success(
            f"MITRE database loaded: "
            f"{len(MITRE_TECHNIQUES)} techniques"
        )

    else:

        st.error(
            "MITRE database unavailable"
        )

    analyst_name = st.text_input(
        "Analyst Name",
        "Security Analyst"
    )

    st.divider()

    st.caption(
        "TriageMind AI"
    )

    st.caption(
        "ML Detection + MITRE ATT&CK"
    )


# ============================================================
# TABS
# ============================================================

dashboard_tab, triage_tab, network_tab, mitre_tab, soc_tab, diagnostics_tab = st.tabs(
    [
        "🏠 Dashboard",
        "🚨 Alert Triage",
        "🌐 Network Detection",
        "🎯 MITRE ATT&CK",
        "🛡️ SOC Investigation",
        "⚙️ Diagnostics"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with dashboard_tab:

    history = load_history()

    total = len(
        history
    )

    high_count = 0

    average_risk = 0

    if not history.empty:

        high_count = len(
            history[
                history[
                    "severity"
                ].isin(
                    [
                        "High",
                        "Critical"
                    ]
                )
            ]
        )

        average_risk = (
            history[
                "risk_score"
            ].mean()
        )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Investigations",
        total
    )

    c2.metric(
        "High/Critical",
        high_count
    )

    c3.metric(
        "Average Risk",
        f"{average_risk:.1f}"
    )

    c4.metric(
        "MITRE Techniques",
        len(
            MITRE_TECHNIQUES
        )
    )

    st.divider()

    st.subheader(
        "TriageMind Workflow"
    )

    a, b, c, d = st.columns(4)

    a.info(
        "1️⃣\n\n"
        "**Telemetry**\n\n"
        "Network/security input"
    )

    b.info(
        "2️⃣\n\n"
        "**ML Detection**\n\n"
        "Detect suspicious activity"
    )

    c.info(
        "3️⃣\n\n"
        "**MITRE Top-3**\n\n"
        "Retrieve ATT&CK techniques"
    )

    d.info(
        "4️⃣\n\n"
        "**SOC Investigation**\n\n"
        "Risk and response"
    )

    if not history.empty:

        st.subheader(
            "Recent Investigations"
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
# ALERT TRIAGE
# ============================================================

with triage_tab:

    st.header(
        "🚨 Alert Triage"
    )

    st.write(
        "Enter a security alert and TriageMind "
        "will perform network assessment and "
        "retrieve the Top-3 MITRE ATT&CK techniques."
    )

    alert_text = st.text_area(
        "Security Alert",
        height=160,
        placeholder=(
            "Example: "
            "Multiple failed login attempts followed "
            "by suspicious PowerShell activity."
        )
    )

    c1, c2 = st.columns(2)

    with c1:

        source_ip = st.text_input(
            "Source IP",
            "192.168.1.50"
        )

        destination_ip = st.text_input(
            "Destination IP",
            "10.0.0.20"
        )

        protocol = st.selectbox(
            "Protocol",
            [
                "TCP",
                "UDP",
                "ICMP",
                "HTTP",
                "HTTPS",
                "DNS",
                "SSH",
                "FTP"
            ]
        )

    with c2:

        source_port = st.number_input(
            "Source Port",
            0,
            65535,
            49152
        )

        destination_port = st.number_input(
            "Destination Port",
            0,
            65535,
            443
        )

        packets = st.number_input(
            "Packet Count",
            0,
            1000000,
            100
        )

    notes = st.text_area(
        "Analyst Notes",
        height=100
    )

    if st.button(
        "🔍 Analyze Alert",
        type="primary",
        use_container_width=True
    ):

        clean_alert = sanitize_text(
            alert_text
        )

        if not clean_alert:

            st.warning(
                "Enter a security alert first."
            )

        else:

            network_data = {

                "dur":
                    10,

                "sbytes":
                    5000,

                "dbytes":
                    3000,

                "spkts":
                    packets,

                "dpkts":
                    packets,

                "sttl":
                    64,

                "dttl":
                    64,

                "sport":
                    source_port,

                "dsport":
                    destination_port,

                "proto":
                    protocol.lower()
            }

            result = predict_network(
                network_data
            )

            # ==================================================
            # TOP 3 MITRE RETRIEVAL
            # ==================================================

            mitre_results = get_top3_mitre(
                clean_alert
            )

            mitre_ids = [
                item["id"]
                for item
                in mitre_results[:3]
            ]

            verification = verify_mitre_ids(
                mitre_ids
            )

            telemetry = semantic_parse(
                network_data
            )

            recommendations = (
                generate_recommendations(
                    result,
                    mitre_results
                )
            )

            result[
                "mitre_ids"
            ] = verification[
                "verified"
            ]

            result[
                "alert"
            ] = clean_alert

            st.session_state[
                "latest_result"
            ] = result

            st.session_state[
                "latest_mitre"
            ] = mitre_results

            st.session_state[
                "latest_verification"
            ] = verification

            st.session_state[
                "latest_telemetry"
            ] = telemetry

            st.session_state[
                "latest_recommendations"
            ] = recommendations

            st.session_state[
                "latest_notes"
            ] = notes

            save_investigation(
                result,
                analyst_name,
                notes,
                "Manual Alert"
            )

            st.success(
                "Analysis completed and saved."
            )


# ============================================================
# LATEST TRIAGE RESULT
# ============================================================

if (
    "latest_result"
    in st.session_state
):

    with triage_tab:

        result = st.session_state[
            "latest_result"
        ]

        st.divider()

        st.subheader(
            "Security Assessment"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Prediction",
            result[
                "prediction"
            ]
        )

        b.metric(
            "Probability",
            f"{result['probability'] * 100:.2f}%"
        )

        c.metric(
            "Severity",
            result[
                "severity"
            ]
        )

        d.metric(
            "Risk Score",
            f"{result['risk_score']}/100"
        )

        if result.get(
            "error"
        ):

            st.warning(
                result[
                    "error"
                ]
            )

        st.subheader(
            "Semantic Telemetry"
        )

        st.code(
            st.session_state[
                "latest_telemetry"
            ]
        )

        # ======================================================
        # TOP 3 MITRE
        # ======================================================

        st.subheader(
            "🎯 Top 3 MITRE ATT&CK Techniques"
        )

        latest_mitre = (
            st.session_state[
                "latest_mitre"
            ]
        )

        if latest_mitre:

            for rank, item in enumerate(
                latest_mitre[:3],
                start=1
            ):

                st.markdown(
                    f"""
                    <div class="attack-card">

                    <h3>
                    #{rank} —
                    {item['id']} —
                    {item['name']}
                    </h3>

                    <p>
                    <b>Retrieval Score:</b>
                    {item.get('similarity', 0):.4f}
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander(
                    f"View {item['id']} details"
                ):

                    st.write(
                        item.get(
                            "description",
                            "No description available."
                        )
                    )

                    if item.get(
                        "phases"
                    ):

                        st.write(
                            "**Tactics:** "
                            +
                            ", ".join(
                                item[
                                    "phases"
                                ]
                            )
                        )

                    if item.get(
                        "platforms"
                    ):

                        st.write(
                            "**Platforms:** "
                            +
                            ", ".join(
                                item[
                                    "platforms"
                                ]
                            )
                        )

        else:

            st.info(
                "No MITRE ATT&CK techniques found."
            )

        # ======================================================
        # VERIFICATION
        # ======================================================

        verification = (
            st.session_state[
                "latest_verification"
            ]
        )

        st.subheader(
            "MITRE Citation Verification"
        )

        v1, v2, v3 = st.columns(3)

        v1.metric(
            "Verified IDs",
            len(
                verification[
                    "verified"
                ]
            )
        )

        v2.metric(
            "Invalid IDs",
            len(
                verification[
                    "fabricated"
                ]
            )
        )

        v3.metric(
            "Fabricated-ID Rate",
            f"{verification['fabricated_rate'] * 100:.2f}%"
        )

        st.subheader(
            "Recommended Actions"
        )

        for action in (
            st.session_state[
                "latest_recommendations"
            ]
        ):

            st.write(
                "• " + action
            )


# ============================================================
# NETWORK DETECTION
# ============================================================

with network_tab:

    st.header(
        "🌐 Network Detection"
    )

    st.write(
        "Upload a CSV containing network telemetry."
    )

    uploaded_file = st.file_uploader(
        "Upload network CSV",
        type=["csv"]
    )

    if uploaded_file:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"{len(df)} rows loaded."
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            if st.button(
                "🚀 Run Network Detection",
                type="primary"
            ):

                predictions = []
                probabilities = []
                severities = []
                risks = []

                progress = st.progress(
                    0
                )

                for index in range(
                    len(df)
                ):

                    row = df.iloc[
                        index
                    ].to_dict()

                    result = predict_network(
                        row
                    )

                    predictions.append(
                        result[
                            "prediction"
                        ]
                    )

                    probabilities.append(
                        result[
                            "probability"
                        ]
                    )

                    severities.append(
                        result[
                            "severity"
                        ]
                    )

                    risks.append(
                        result[
                            "risk_score"
                        ]
                    )

                    progress.progress(
                        int(
                            (
                                (
                                    index + 1
                                )
                                /
                                len(df)
                            )
                            * 100
                        )
                    )

                output = df.copy()

                output[
                    "prediction"
                ] = predictions

                output[
                    "attack_probability"
                ] = probabilities

                output[
                    "severity"
                ] = severities

                output[
                    "risk_score"
                ] = risks

                st.session_state[
                    "network_results"
                ] = output

        except Exception as error:

            st.error(
                f"CSV processing error: {error}"
            )

    if (
        "network_results"
        in st.session_state
    ):

        output = st.session_state[
            "network_results"
        ]

        st.subheader(
            "Detection Results"
        )

        st.dataframe(
            output,
            use_container_width=True
        )

        csv_data = output.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "⬇️ Download Results",
            csv_data,
            "triagemind_network_results.csv",
            "text/csv"
        )


# ============================================================
# MITRE EXPLORER
# ============================================================

with mitre_tab:

    st.header(
        "🎯 MITRE ATT&CK Explorer"
    )

    st.write(
        "Search the local MITRE ATT&CK Enterprise knowledge base."
    )

    if not MITRE_TECHNIQUES:

        st.error(
            "MITRE ATT&CK data was not found."
        )

    else:

        query = st.text_input(
            "Search MITRE ATT&CK",
            placeholder=(
                "Example: PowerShell, "
                "credential dumping, scanning, T1059"
            )
        )

        if query:

            results = get_top3_mitre(
                query
            )

            st.subheader(
                "Top 3 Results"
            )

            for rank, item in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"""
                    ### #{rank}
                    {item['id']} —
                    {item['name']}
                    """
                )

                st.caption(
                    f"Retrieval score: "
                    f"{item['similarity']:.4f}"
                )

                st.write(
                    item[
                        "description"
                    ]
                )

                if item[
                    "phases"
                ]:

                    st.write(
                        "**Tactics:** "
                        +
                        ", ".join(
                            item[
                                "phases"
                            ]
                        )
                    )

                st.divider()


# ============================================================
# SOC INVESTIGATION
# ============================================================

with soc_tab:

    st.header(
        "🛡️ SOC Investigation"
    )

    history = load_history()

    if history.empty:

        st.info(
            "No investigation history available."
        )

    else:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Severity Distribution"
        )

        counts = (
            history[
                "severity"
            ].value_counts()
        )

        st.bar_chart(
            counts
        )

    if (
        "latest_result"
        in st.session_state
    ):

        st.divider()

        report = build_report(

            st.session_state[
                "latest_result"
            ],

            st.session_state[
                "latest_telemetry"
            ],

            st.session_state[
                "latest_mitre"
            ],

            st.session_state[
                "latest_verification"
            ],

            st.session_state[
                "latest_recommendations"
            ],

            analyst_name,

            st.session_state[
                "latest_notes"
            ]
        )

        st.subheader(
            "Latest Security Report"
        )

        st.text_area(
            "Report",
            report,
            height=450
        )

        st.download_button(
            "⬇️ Download Security Report",
            report,
            "triagemind_security_report.txt",
            "text/plain",
            use_container_width=True
        )

        # Download individual Top-3 descriptions.

        for rank, item in enumerate(
            st.session_state[
                "latest_mitre"
            ][:3],
            start=1
        ):

            description = (

                f"Rank #{rank}\n\n"

                f"{item['id']} - "
                f"{item['name']}\n\n"

                f"{item['description']}\n\n"

                f"Tactics: "
                +
                ", ".join(
                    item[
                        "phases"
                    ]
                )
            )

            st.download_button(
                f"⬇️ Download #{rank} "
                f"{item['id']} Description",

                description,

                f"{item['id']}_description.txt",

                "text/plain"
            )


# ============================================================
# DIAGNOSTICS
# ============================================================

with diagnostics_tab:

    st.header(
        "⚙️ Diagnostics"
    )

    status = pd.DataFrame(
        [
            {
                "Component":
                    "ML Classifier",

                "Status":
                    (
                        "Loaded"
                        if classifier is not None
                        else "Missing"
                    )
            },

            {
                "Component":
                    "Feature Columns",

                "Status":
                    (
                        f"{len(feature_columns)} loaded"
                        if feature_columns
                        else "Missing"
                    )
            },

            {
                "Component":
                    "MITRE ATT&CK",

                "Status":
                    (
                        f"{len(MITRE_TECHNIQUES)} techniques"
                        if MITRE_TECHNIQUES
                        else "Missing"
                    )
            },

            {
                "Component":
                    "SQLite",

                "Status":
                    "Enabled"
            },

            {
                "Component":
                    "Top-3 Retrieval",

                "Status":
                    "Enabled"
            },

            {
                "Component":
                    "Input Sanitization",

                "Status":
                    "Enabled"
            },

            {
                "Component":
                    "MITRE ID Verification",

                "Status":
                    "Enabled"
            }
        ]
    )

    st.dataframe(
        status,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Model Information"
    )

    if classifier is not None:

        st.write(
            "Model:",
            type(
                classifier
            ).__name__
        )

        st.write(
            "Number of features:",
            len(
                feature_columns
            )
        )

        with st.expander(
            "View feature columns"
        ):

            st.code(
                "\n".join(
                    feature_columns
                )
            )

    else:

        st.error(
            model_error
        )

    st.divider()

    st.subheader(
        "MITRE Citation Verification Test"
    )

    test_ids = st.text_input(
        "Enter MITRE IDs separated by commas",
        "T1059,T1003,T9999"
    )

    if st.button(
        "Verify IDs"
    ):

        ids = [
            item.strip()
            for item in test_ids.split(
                ","
            )
            if item.strip()
        ]

        verification = (
            verify_mitre_ids(
                ids
            )
        )

        a, b, c = st.columns(3)

        a.metric(
            "Verified",
            len(
                verification[
                    "verified"
                ]
            )
        )

        b.metric(
            "Invalid",
            len(
                verification[
                    "fabricated"
                ]
            )
        )

        c.metric(
            "Fabricated-ID Rate",
            f"{verification['fabricated_rate'] * 100:.2f}%"
        )

        st.write(
            "Verified:",
            verification[
                "verified"
            ]
        )

        st.write(
            "Invalid:",
            verification[
                "fabricated"
            ]
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TriageMind AI | ML Network Detection | "
    "MITRE ATT&CK Top-3 Retrieval | "
    "SOC Investigation"
)
