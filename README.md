# 🔍 Phishing URL Detection

This is a simple yet effective machine learning project to detect phishing websites just by analyzing their URLs. It uses [LightGBM](https://lightgbm.readthedocs.io/) to classify URLs as **legitimate** or **phishing** based on URL pattern & features.

---

## 📂 Project Structure

```
├── phishing_detector.py     # Main script: train or predict
├── legit_sample.csv         # Rename to legit.csv (# Put original data in given format)
├── phishing_sample.csv      # Rename to phishing.csv (# Put original data in given format)
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files/folders
```

---

## ⚙️ Setup Instructions

### ✅ 1. Clone the Repo

```bash
git clone https://github.com/nazmussakib44/phishing-url-detection.git
cd phishing-url-detection
```

---

### ✅ 2. Create and Activate Virtual Environment

#### 💻 macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 🪟 Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you're on macOS and face `libomp.dylib` issues (LightGBM dependency), run:

```bash
brew install libomp
```

---

### ✅ 4. Run the App

```bash
python phishing_detector.py
```

You'll be prompted to choose:

```
Choose an option:
1. Train model
2. Predict URL
```

---

## 📊 Sample Data

You can modify the following CSVs to feed your own URL datasets:

- `legit_sample.csv` – list of **legitimate** URLs
- `phishing_sample.csv` – list of **phishing** URLs

Each file should contain **one URL per line** without headers.

---

## 📦 Dependencies

- `lightgbm`
- `scikit-learn`
- `joblib`
- `pandas`
- `numpy`
- `tqdm`

All managed via `requirements.txt`.

---

## 💡 Features Extracted from URLs

The model looks at features such as:
- Length of the URL
- Presence of “@” or “-”
- Number of dots
- Use of HTTPS
- Suspicious keywords

---

## 🛡️ Model

The model is trained using **LightGBM** and saved as `model.pkl` after training. **You must train the model at least once before using prediction.**

---

## 📬 Contributing

Feel free to fork the repo, raise issues or submit PRs if you'd like to improve the detection logic or add enhancements.

---

## 📄 License

This project is licensed under the MIT License.
