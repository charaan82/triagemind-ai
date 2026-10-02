# ============================================================
# DASHBOARD
# ============================================================

with dashboard_tab:

    # --------------------------------------------------------
    # DASHBOARD HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            padding: 30px;
            border-radius: 20px;
            background: linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 50%,
                #111827 100%
            );
            border: 1px solid #334155;
            margin-bottom: 25px;
        ">

        <div style="
            font-size: 14px;
            color: #60a5fa;
            font-weight: 600;
            letter-spacing: 1px;
        ">
        SECURITY OPERATIONS CENTER
        </div>

        <h1 style="
            color: white;
            font-size: 42px;
            margin: 8px 0;
        ">
        🛡️ TriageMind AI
        </h1>

        <p style="
            color: #cbd5e1;
            font-size: 18px;
            margin-bottom: 5px;
        ">
        Intelligent Cybersecurity Alert Triage Platform
        </p>

        <p style="
            color: #94a3b8;
            font-size: 14px;
        ">
        Machine Learning • MITRE ATT&CK • Explainable AI •
        SOC Investigation
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.subheader("🟢 System Overview")

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:

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

    with status_col2:

        if MITRE_TECHNIQUES:

            st.success("MITRE ONLINE")

            st.metric(
                "Techniques",
                len(MITRE_TECHNIQUES)
            )

        else:

            st.error("MITRE OFFLINE")

            st.metric(
                "Techniques",
                0
            )

    with status_col3:

        if FEATURE_COLUMNS:

            st.success("FEATURES READY")

            st.metric(
                "Features",
                len(FEATURE_COLUMNS)
            )

        else:

            st.warning("AUTO FEATURES")

            st.metric(
                "Features",
                "Automatic"
            )

    with status_col4:

        if FAISS_PATH:

            st.info("FAISS AVAILABLE")

        else:

            st.info("TF-IDF ACTIVE")

        st.metric(
            "Investigations",
            len(st.session_state.history)
        )

    # --------------------------------------------------------
    # QUICK SECURITY SUMMARY
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Security Operations Summary")

    total_investigations = len(
        st.session_state.history
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        total_risk = pd.to_numeric(
            history_df["Risk Score"],
            errors="coerce"
        ).fillna(0)

        average_risk = (
            total_risk.mean()
            if len(total_risk)
            else 0
        )

        high_risk_count = sum(
            history_df["Severity"].isin(
                [
                    "HIGH",
                    "CRITICAL"
                ]
            )
        )

        latest_severity = (
            history_df.iloc[-1]["Severity"]
        )

    else:

        average_risk = 0
        high_risk_count = 0
        latest_severity = "No alerts"

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Investigations",
            total_investigations
        )

    with c2:

        st.metric(
            "Average Risk",
            f"{average_risk:.1f}/100"
        )

    with c3:

        st.metric(
            "High/Critical Cases",
            high_risk_count
        )

    with c4:

        st.metric(
            "Latest Severity",
            latest_severity
        )

    # --------------------------------------------------------
    # QUICK START
    # --------------------------------------------------------

    st.divider()

    st.subheader("🚀 Quick Start")

    st.write(
        "Use the workflow below to investigate a "
        "security event from detection to response."
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        st.markdown(
            """
            ### 01

            #### 🚨 Detect

            Upload network traffic or enter
            a suspicious security alert.

            **Module:**
            Network Detection
            """
        )

    with q2:

        st.markdown(
            """
            ### 02

            #### 🔎 Triage

            Analyze the alert and identify
            relevant threat behavior.

            **Module:**
            Alert Triage
            """
        )

    with q3:

        st.markdown(
            """
            ### 03

            #### 🎯 Map

            Map suspicious behavior to
            MITRE ATT&CK techniques.

            **Module:**
            MITRE ATT&CK
            """
        )

    with q4:

        st.markdown(
            """
            ### 04

            #### 🛡️ Investigate

            Combine the evidence and
            generate a SOC investigation.

            **Module:**
            SOC Investigation
            """
        )

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.divider()

    st.subheader("🏗️ TriageMind Architecture")

    architecture_col1, architecture_col2 = st.columns(
        [1.4, 1]
    )

    with architecture_col1:

        st.code(
            """
             SECURITY ALERT
                    │
                    ▼
          ┌──────────────────┐
          │  TRIAGEMIND AI   │
          └────────┬─────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
     ML DETECTION      MITRE SEARCH
          │                 │
          ▼                 ▼
     Attack Status     Technique
     Confidence        Mapping
          │                 │
          └────────┬────────┘
                   │
                   ▼
             RISK ENGINE
                   │
                   ▼
          SOC INVESTIGATION
                   │
          ┌────────┴────────┐
          ▼                 ▼
       REPORT            ANALYTICS
            """,
            language="text"
        )

    with architecture_col2:

        st.markdown(
            """
            ### Core Intelligence

            **🤖 Machine Learning**

            Classifies network activity and
            identifies potential attacks.

            **🎯 MITRE ATT&CK**

            Connects observed behavior with
            known adversary techniques.

            **📈 Risk Engine**

            Combines detection confidence,
            attack rate and MITRE relevance.

            **🧠 Explainable AI**

            Displays important model features
            to help analysts understand
            predictions.
            """
        )

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    st.divider()

    st.subheader("🧠 Platform Capabilities")

    cap1, cap2, cap3 = st.columns(3)

    with cap1:

        st.markdown(
            """
            ### 🤖 Machine Learning

            ✓ Network intrusion detection

            ✓ Attack classification

            ✓ Prediction confidence

            ✓ Model evaluation

            ✓ Feature importance

            ✓ Confusion matrix
            """
        )

    with cap2:

        st.markdown(
            """
            ### 🎯 Threat Intelligence

            ✓ MITRE ATT&CK search

            ✓ Technique identification

            ✓ Tactic mapping

            ✓ Alert enrichment

            ✓ Threat context

            ✓ Similarity scoring
            """
        )

    with cap3:

        st.markdown(
            """
            ### 🛡️ SOC Operations

            ✓ Alert triage

            ✓ Risk scoring

            ✓ Investigation workflow

            ✓ Security reports

            ✓ CSV export

            ✓ Investigation history
            """
        )

    # --------------------------------------------------------
    # DATA PIPELINE
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔄 Detection Pipeline")

    pipeline = pd.DataFrame(
        {
            "Stage": [
                "1. Input",
                "2. ML Analysis",
                "3. Threat Mapping",
                "4. Risk Analysis",
                "5. Investigation",
                "6. Reporting",
            ],
            "Purpose": [
                "Security alert / network CSV",
                "Classify suspicious activity",
                "Identify MITRE ATT&CK technique",
                "Calculate security risk",
                "Guide analyst investigation",
                "Generate downloadable report",
            ],
        }
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RECENT INVESTIGATIONS
    # --------------------------------------------------------

    st.divider()

    st.subheader("📋 Recent Investigations")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        # Convert risk to numeric for display
        if "Risk Score" in history_df.columns:

            history_df["Risk Score"] = pd.to_numeric(
                history_df["Risk Score"],
                errors="coerce"
            ).fillna(0)

        st.dataframe(
            history_df.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No investigations have been performed yet. "
            "Go to the SOC Investigation tab to start."
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("🤖 AI Model Information")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        if MODEL is not None:

            st.success(
                "Machine-learning classifier loaded."
            )

            st.write(
                "**Model Type:**",
                type(MODEL).__name__
            )

            if FEATURE_COLUMNS:

                st.write(
                    "**Input Features:**",
                    len(FEATURE_COLUMNS)
                )

        else:

            st.error(
                "Machine-learning model is unavailable."
            )

    with model_col2:

        importance_df = get_feature_importance()

        if importance_df is not None:

            st.write(
                "**Top Model Features**"
            )

            st.bar_chart(
                importance_df.head(8).set_index(
                    "Feature"
                )
            )

        else:

            st.info(
                "Feature importance is unavailable "
                "for the current classifier."
            )

    # --------------------------------------------------------
    # MITRE STATUS
    # --------------------------------------------------------

    st.divider()

    st.subheader("🎯 MITRE ATT&CK Knowledge Base")

    mitre_col1, mitre_col2 = st.columns(2)

    with mitre_col1:

        if MITRE_TECHNIQUES:

            st.success(
                "MITRE ATT&CK knowledge base loaded."
            )

            st.metric(
                "Available Techniques",
                len(MITRE_TECHNIQUES)
            )

        else:

            st.error(
                "MITRE ATT&CK data is unavailable."
            )

    with mitre_col2:

        if FAISS_PATH:

            st.info(
                "FAISS index detected."
            )

            st.write(
                "The project also contains a FAISS "
                "index for retrieval experiments."
            )

        else:

            st.info(
                "TF-IDF retrieval is active."
            )

            st.write(
                "The current dashboard uses lightweight "
                "TF-IDF retrieval and does not require "
                "FAISS for deployment."
            )

    # --------------------------------------------------------
    # PROJECT VALUE
    # --------------------------------------------------------

    st.divider()

    st.subheader("💼 What TriageMind Demonstrates")

    value1, value2, value3, value4 = st.columns(4)

    with value1:

        st.markdown(
            """
            **Cybersecurity**

            Network intrusion detection,
            threat intelligence and
            SOC workflows.
            """
        )

    with value2:

        st.markdown(
            """
            **Machine Learning**

            Classification,
            confidence scoring,
            evaluation and explainability.
            """
        )

    with value3:

        st.markdown(
            """
            **Threat Intelligence**

            MITRE ATT&CK technique
            identification and
            contextual enrichment.
            """
        )

    with value4:

        st.markdown(
            """
            **Deployment**

            Interactive Streamlit
            dashboard with
            downloadable reports.
            """
        )

    # --------------------------------------------------------
    # FOOTER DASHBOARD
    # --------------------------------------------------------

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
