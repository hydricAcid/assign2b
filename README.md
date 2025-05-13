# 📊 Traffic Flow Prediction Using LSTM, GRU, and CNN

This project processes SCATS traffic flow data and applies three machine learning models (LSTM, GRU, and CNN) to forecast traffic.

---

## 📁 Project Structure

```
project/
├── data/
│   ├── Scats_data_october_2006.xlsx       # Raw input data
│   └── processed/
│       └── dataset.npz                    # Preprocessed sequences
├── models/
│   ├── model_lstm.py                      # LSTM model
│   ├── model_gru.py                       # GRU model
│   └── model_cnn.py                       # CNN model
├── utils/
│   └── preprocess.py                      # Data preprocessing
└── evaluate.py                            # Model evaluation and comparison
```

---

## ⚙️ Setup

Install required packages:

```bash
pip install pandas numpy scikit-learn tensorflow openpyxl
```

---

## 🚀 Run Pipeline

### Step 1: Preprocess data

```bash
python utils/preprocess.py
```

Output: `data/processed/dataset.npz`

### Step 2: Train Models

```bash
python models/model_lstm.py
python models/model_gru.py
python models/model_cnn.py
```

Each command saves a model in `models/`

### Step 3: Evaluate and Compare

```bash
python evaluate.py
```

Output: RMSE and MAE metrics for each model.

---

## 📌 Notes

- All models use a fixed look-back window of 10 time steps.
- Data is normalized per SCATS site using `MinMaxScaler`.
- Evaluation is based on RMSE and MAE.

> Built for COS30019 Assignment 2B, 2025 Semester 1.
