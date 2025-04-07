# EEG Processing & Deep Learning Framework

This repository brings together two powerful tools for working with EEG data:

1. **EEG Epoch Processing Framework**: Preprocess and segment raw EEG recordings into clean, event-locked epochs.
2. **DeepEEG**: Apply deep learning techniques to classify epileptic events from intracranial EEG (iEEG) data.

---

## 📦 Project Overview

### 1. **EEG Epoch Processing Framework**

A modular pipeline to preprocess raw EEG data, extract events, segment into epochs, and save for downstream analysis.

#### 🧩 Key Features

- **Supports .edf EEG data**
- **Band-pass and Notch Filtering** (e.g., 1–250 Hz, 50/60 Hz noise removal)
- **Wavelet Denoising** (optional, preserves signal integrity)
- **Event Extraction** for stimulus-locked segmentation
- **Epoch Creation**: aligns EEG segments to events (e.g., stimulus onsets)
- **Saving in `.fif`** format for easy reuse

#### 🛠 Usage

1. **Customize Preprocessing**  
   Modify parameters in the `CONFIG` dictionary (e.g., filter settings, channels, events).

2. **Run the Pipeline**  
   ```python
   main(CONFIG)
   ```

3. **Load Processed Epochs**  
   ```python
   epochs = load_epochs("sub-01")
   ```

#### 🔬 Applications
- Cognitive neuroscience experiments
- Brain-computer interface (BCI) systems
- Clinical EEG analysis (ERPs, oscillations)

---

### 2. **DeepEEG: Epileptic Event Classification**

A deep learning pipeline to classify epileptic events using iEEG data, leveraging neural networks and spectral features.

#### 🔍 Highlights

- **Event Epoch Features (EEF)**: Extracted features by frequency band and event type
- **Deep Learning Model**: Mish activations, batch norm, dropout for regularization
- **Hyperparameter Optimization**: Optuna for model tuning
- **Performance Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Explainability**: Integrated Gradients & Permutation Importance

#### 📂 Project Structure

```
DeepEEG/
├── data/              # Preprocessed connectivity features
├── notebooks/         # Jupyter notebooks for training/evaluation
├── models/            # Saved trained models
└── README.md          # This file
```

---

## ⚙️ Installation

Make sure to install the following dependencies:

```bash
pip install mne-connectivity optuna tensorpac pytorch-ranger captum
```

---

## 🚀 How to Run

### 1. Load EEG Epochs
```python
epochs = load_epochs("sub-01")
```

### 2. Compute Spectral Connectivity
```python
bands = {"Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 50)}
session_dfs = compute_connectivity_per_session(epochs, bands, "sub-01")
```

### 3. Train XGBoost Classifier
```python
xgb = XGBClassifier(eval_metric='mlogloss', **best_params)
xgb.fit(X_train, y_train)
print(classification_report(y_test, xgb.predict(X_test)))
```

### 4. Train Deep Neural Network
```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### 5. Evaluate Feature Importance
```python
# Integrated Gradients
ig = IntegratedGradients(best_model)
attributions = ig.attribute(inputs, baseline)

# Permutation Importance
importances = permutation_importance(best_model, test_loader, accuracy_score)
```

---

## 📊 Results

| Metric        | XGBoost | Deep Neural Net |
|---------------|---------|------------------|
| Accuracy      | 0.92    | 0.94             |
| Precision     | 0.91    | 0.93             |
| Recall        | 0.92    | 0.94             |
| F1 Score      | 0.91    | 0.93             |
| AUC-ROC       | 0.98    | 0.99             |

---


## 📜 License

This project is released under the **MIT License**.

---

![image.png](attachment:image.png)

