import re
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# 1. Feature Extraction Function
def extract_features(url):
    parsed = urlparse(url)
    features = {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "has_at": int("@" in url),
        "has_hyphen": int("-" in url),
        "has_ip": int(bool(re.search(r'http[s]?://\d{1,3}(\.\d{1,3}){3}', url))),
        "path_length": len(parsed.path),
        "has_https": int(parsed.scheme == "https"),
    }
    return list(features.values())

# 2. Prepare Dataset
# def load_dataset(phishing_file, legit_file):
#     phishing = pd.read_csv(phishing_file)
#     phishing['label'] = 1
#     legit = pd.read_csv(legit_file)
#     legit['label'] = 0

#     df = pd.concat([phishing, legit], ignore_index=True)
#     df = df.sample(frac=1).reset_index(drop=True)  # shuffle
#     return df

def load_dataset(phishing_file, legit_file):
    phishing = pd.read_csv(phishing_file)
    legit = pd.read_csv(legit_file)

    phishing.columns = phishing.columns.str.lower()
    legit.columns = legit.columns.str.lower()

    phishing['label'] = 1
    legit['label'] = 0

    phishing = phishing[['url', 'label']]
    legit = legit[['url', 'label']]

    df = pd.concat([phishing, legit], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df

# 3. Feature Extraction on Dataset
# def extract_features_from_df(df):
#     features = []
#     for url in df['url']:
#         features.append(extract_features(url))
#     return pd.DataFrame(features), df['label']

# def extract_features_from_df(df):
#     features = []
#     labels = []
#     for url, label in zip(df['url'], df['label']):
#         try:
#             feats = extract_features(url)
#             features.append(feats)
#             labels.append(label)
#         except Exception as e:
#             print(f"Skipping invalid URL: {url} — Reason: {e}")
#     return pd.DataFrame(features), pd.Series(labels)

def extract_features_from_df(df):
    features = []
    labels = []
    for url, label in zip(df['url'], df['label']):
        try:
            feats = extract_features(url)
            features.append(feats)
            labels.append(label)
        except Exception as e:
            print(f"[!] Skipping URL due to error: {url}\n    Reason: {e}")
    return pd.DataFrame(features), pd.Series(labels)

# 4. Train and Save Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")

    preds = model.predict(X_test)
    print("Model Evaluation:\n", classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))

# 5. Predict from URL
# def predict_url(url):
#     model = joblib.load("model.pkl")
#     features = extract_features(url)
#     risk_score = model.predict_proba([features])[0][1]
#     print(f"\nURL: {url}")
#     print(f"Phishing Risk Score: {risk_score:.2f}")
#     print("Prediction:", "Phishing ⚠️" if risk_score > 0.5 else "Legitimate ✅")

def predict_url(url):
    if not os.path.exists("model.pkl"):
        print("⚠️ Model not found. Please train the model first (option 1).")
        return
    model = joblib.load("model.pkl")
    features = extract_features(url)
    risk_score = model.predict_proba([features])[0][1]
    print(f"\nURL: {url}")
    print(f"Phishing Risk Score: {risk_score:.2f}")
    print("Prediction:", "Phishing ⚠️" if risk_score > 0.5 else "Legitimate ✅")

# === MAIN ===
if __name__ == "__main__":
    import os

    print("Choose an option:")
    print("1. Train model")
    print("2. Predict URL")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        if not os.path.exists("phishing.csv") or not os.path.exists("legit.csv"):
            print("Dataset files 'phishing.csv' and 'legit.csv' are required in the same directory.")
        else:
            df = load_dataset("phishing.csv", "legit.csv")
            X, y = extract_features_from_df(df)
            train_model(X, y)

    elif choice == '2':
        url_input = input("Enter a URL: ")
        predict_url(url_input)

    else:
        print("Invalid option.")
