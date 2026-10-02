import os
import json
import joblib
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# TriageMind Configuration
# ============================================================

MODEL_FILE = "triagemind_model.pkl"

MITRE_TECHNIQUES_FILE = "mitre_data/mitre_techniques.json"
MITRE_INDEX_FILE = "mitre_data/mitre_index.faiss"

TOP_K = 3


# ============================================================
# 1. Check Required Files
# ============================================================

def check_files():

    print("\nChecking required files...")
    print("-" * 60)

    required_files = [
        MODEL_FILE,
        MITRE_TECHNIQUES_FILE,
        MITRE_INDEX_FILE
    ]

    all_files_found = True

    for file in required_files:

        if os.path.exists(file):

            print("OK     :", file)

        else:

            print("MISSING:", file)
            all_files_found = False

    return all_files_found


# ============================================================
# 2. Load Random Forest Model
# ============================================================

def load_classifier():

    print("\nLoading Random Forest classifier...")

    try:

        package = joblib.load(MODEL_FILE)

        # New model format
        if isinstance(package, dict):

            model = package["model"]
            feature_columns = package["features"]

        # Old model format
        else:

            model = package
            feature_columns = None

            print(
                "Warning: old model format detected."
            )

        print("Random Forest loaded successfully.")

        if feature_columns is not None:

            print(
                "Number of saved features:",
                len(feature_columns)
            )

        return model, feature_columns

    except Exception as error:

        print(
            "Error loading Random Forest:",
            error
        )

        return None, None


# ============================================================
# 3. Load MITRE Techniques
# ============================================================

def load_mitre_techniques():

    print("\nLoading MITRE ATT&CK techniques...")

    try:

        with open(
            MITRE_TECHNIQUES_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            techniques = json.load(file)

        print(
            "MITRE techniques loaded:",
            len(techniques)
        )

        return techniques

    except Exception as error:

        print(
            "Error loading MITRE techniques:",
            error
        )

        return []


# ============================================================
# 4. Load FAISS Index
# ============================================================

def load_faiss():

    print("\nLoading FAISS index...")

    try:

        index = faiss.read_index(
            MITRE_INDEX_FILE
        )

        print(
            "FAISS loaded successfully."
        )

        print(
            "Vectors in index:",
            index.ntotal
        )

        return index

    except Exception as error:

        print(
            "Error loading FAISS:",
            error
        )

        return None


# ============================================================
# 5. Load Sentence-BERT
# ============================================================

def load_embedding_model():

    print("\nLoading Sentence-BERT...")

    try:

        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print(
            "Sentence-BERT loaded successfully."
        )

        return model

    except Exception as error:

        print(
            "Error loading Sentence-BERT:",
            error
        )

        return None


# ============================================================
# 6. Search MITRE ATT&CK
# ============================================================

def search_mitre(
    alert,
    embedding_model,
    mitre_index,
    techniques,
    top_k=TOP_K
):

    if not alert.strip():

        return []

    # Convert alert text into vector
    query_embedding = embedding_model.encode(
        [alert],
        convert_to_numpy=True
    ).astype("float32")

    # Search FAISS
    distances, results = mitre_index.search(
        query_embedding,
        top_k
    )

    matches = []

    for position, result in enumerate(
        results[0]
    ):

        if result < 0:
            continue

        if result >= len(techniques):
            continue

        technique = techniques[result]

        matches.append({

            "rank": position + 1,

            "id": technique.get(
                "id",
                "Unknown"
            ),

            "name": technique.get(
                "name",
                "Unknown"
            ),

            "description": technique.get(
                "description",
                "No description available."
            ),

            "tactics": technique.get(
                "tactics",
                []
            ),

            "distance": float(
                distances[0][position]
            )
        })

    return matches


# ============================================================
# 7. Display MITRE Results
# ============================================================

def display_mitre_results(matches):

    print("\n")
    print("=" * 60)
    print("MITRE ATT&CK RESULTS")
    print("=" * 60)

    if not matches:

        print(
            "No matching MITRE techniques found."
        )

        return

    for match in matches:

        print(
            f"\n{match['rank']}. "
            f"{match['id']} - "
            f"{match['name']}"
        )

        if match["tactics"]:

            print(
                "Tactics:",
                ", ".join(
                    match["tactics"]
                )
            )

        else:

            print(
                "Tactics: Not available"
            )

        print(
            "Distance:",
            round(
                match["distance"],
                4
            )
        )

        print(
            "\nDescription:"
        )

        print(
            match["description"]
        )

        print("-" * 60)


# ============================================================
# 8. Classify UNSW-NB15 CSV
# ============================================================

def classify_csv(
    csv_file,
    classifier,
    feature_columns
):

    print("\nLoading network dataset...")

    try:

        df = pd.read_csv(
            csv_file
        )

    except Exception as error:

        print(
            "Error reading CSV:",
            error
        )

        return None

    print(
        "Rows loaded:",
        len(df)
    )

    # Remove label if it exists
    if "label" in df.columns:

        df = df.drop(
            columns=["label"]
        )

    # Convert categorical columns
    df = pd.get_dummies(df)

    # Replace missing values
    df = df.fillna(0)

    # Match the features used during training
    if feature_columns is not None:

        # Add missing columns
        for column in feature_columns:

            if column not in df.columns:

                df[column] = 0

        # Remove extra columns
        df = df[
            feature_columns
        ]

    print(
        "Data prepared for prediction."
    )

    try:

        predictions = classifier.predict(
            df
        )

    except Exception as error:

        print(
            "\nPrediction error:",
            error
        )

        return None

    print("\n")
    print("=" * 60)
    print("NETWORK CLASSIFICATION")
    print("=" * 60)

    # Count predictions
    prediction_counts = pd.Series(
        predictions
    ).value_counts()

    for label, count in prediction_counts.items():

        print(
            f"{label}: {count} rows"
        )

    print("\nFirst 10 predictions:")

    for i, prediction in enumerate(
        predictions[:10]
    ):

        print(
            f"Row {i + 1}: {prediction}"
        )

    return predictions


# ============================================================
# 9. Create Unified TriageMind Report
# ============================================================

def create_report(
    alert,
    mitre_matches,
    network_prediction=None
):

    print("\n\n")

    print("=" * 60)
    print("              TRIAGEMIND REPORT")
    print("=" * 60)

    # --------------------------------------------------------
    # Security Alert
    # --------------------------------------------------------

    print("\nSECURITY ALERT")
    print("-" * 60)

    print(alert)

    # --------------------------------------------------------
    # MITRE ATT&CK
    # --------------------------------------------------------

    print("\nMITRE ATT&CK ANALYSIS")
    print("-" * 60)

    if mitre_matches:

        best_match = mitre_matches[0]

        print(
            "Primary Technique:"
        )

        print(
            best_match["id"],
            "-",
            best_match["name"]
        )

        print(
            "\nTactic:"
        )

        if best_match["tactics"]:

            print(
                ", ".join(
                    best_match["tactics"]
                )
            )

        else:

            print(
                "Not available"
            )

        print(
            "\nSimilarity Distance:"
        )

        print(
            round(
                best_match["distance"],
                4
            )
        )

        print(
            "\nDescription:"
        )

        print(
            best_match["description"]
        )

        print(
            "\nOther Relevant Techniques:"
        )

        for match in mitre_matches[1:]:

            print(
                "-",
                match["id"],
                "-",
                match["name"]
            )

    else:

        print(
            "No MITRE technique found."
        )

    # --------------------------------------------------------
    # Network Classification
    # --------------------------------------------------------

    print("\nNETWORK CLASSIFICATION")
    print("-" * 60)

    if network_prediction is not None:

        print(
            "Prediction:",
            network_prediction
        )

    else:

        print(
            "No network classification "
            "was provided."
        )

    # --------------------------------------------------------
    # Analyst Guidance
    # --------------------------------------------------------

    print("\nANALYST GUIDANCE")
    print("-" * 60)

    if mitre_matches:

        best_match = mitre_matches[0]

        print(
            "Investigate activity related to:",
            best_match["name"]
        )

        print(
            "Review the affected host, "
            "process execution history, "
            "and related network activity."
        )

        print(
            "Check additional logs and "
            "network events before making "
            "a final security determination."
        )

    else:

        print(
            "Collect additional evidence "
            "and review the alert manually."
        )

    print("\n")
    print("=" * 60)
    print("             END OF REPORT")
    print("=" * 60)


# ============================================================
# 10. Analyze Text Security Alert
# ============================================================

def analyze_alert(
    alert,
    embedding_model,
    mitre_index,
    techniques
):

    print("\nSearching MITRE ATT&CK...")

    matches = search_mitre(
        alert,
        embedding_model,
        mitre_index,
        techniques
    )

    create_report(
        alert,
        matches
    )

    return matches


# ============================================================
# 11. Main Program
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("                 TRIAGEMIND AI")
    print("=" * 60)

    print(
        "\nStarting TriageMind..."
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not check_files():

        print(
            "\nSome required files are missing."
        )

        print(
            "Please check your project folder."
        )

        return

    # --------------------------------------------------------
    # Load components
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Check components
    # --------------------------------------------------------

    if classifier is None:

        print(
            "\nClassifier could not be loaded."
        )

        return

    if not techniques:

        print(
            "\nMITRE techniques could not be loaded."
        )

        return

    if mitre_index is None:

        print(
            "\nFAISS index could not be loaded."
        )

        return

    if embedding_model is None:

        print(
            "\nSentence-BERT could not be loaded."
        )

        return

    # --------------------------------------------------------
    # Ready
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("             TRIAGEMIND IS READY")
    print("=" * 60)

    # --------------------------------------------------------
    # Menu
    # --------------------------------------------------------

    while True:

        print("\n")
        print("1. Analyze security alert")
        print("2. Classify UNSW-NB15 CSV")
        print("3. Exit")

        choice = input(
            "\nEnter your choice: "
        ).strip()

        # ====================================================
        # Option 1: Security Alert
        # ====================================================

        if choice == "1":

            print("\n")
            print(
                "Example:"
            )

            print(
                "PowerShell was used to execute "
                "a malicious encoded command."
            )

            alert = input(
                "\nEnter security alert:\n"
            ).strip()

            if not alert:

                print(
                    "\nAlert cannot be empty."
                )

                continue

            analyze_alert(
                alert,
                embedding_model,
                mitre_index,
                techniques
            )

        # ====================================================
        # Option 2: Network CSV
        # ====================================================

        elif choice == "2":

            print("\n")
            print(
                "Example:"
            )

            print(
                "dataset/UNSW_NB15_testing-set.csv"
            )

            csv_file = input(
                "\nEnter CSV file path:\n"
            ).strip()

            if not os.path.exists(
                csv_file
            ):

                print(
                    "\nCSV file not found."
                )

                continue

            predictions = classify_csv(
                csv_file,
                classifier,
                feature_columns
            )

            if predictions is not None:

                print(
                    "\nNetwork classification completed."
                )

        # ====================================================
        # Option 3: Exit
        # ====================================================

        elif choice == "3":

            print(
                "\nThank you for using TriageMind."
            )

            break

        # ====================================================
        # Invalid option
        # ====================================================

        else:

            print(
                "\nInvalid choice."
            )

            print(
                "Please enter 1, 2, or 3."
            )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":

    main()