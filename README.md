# **DeepEEG: Classifying Epileptic Events using Deep Learning on Intracranial EEG Data**

## **Introduction**
Epilepsy is a neurological disorder characterized by recurrent seizures, and predicting these events can significantly improve patient care. This project, **DeepEEG**, uses **deep learning** techniques to classify epileptic events from **intracranial EEG (iEEG)** data. 

The model processes **Event Epoch Features (EEF)**, which are reshaped features related to specific event types and frequency bands, crucial for analyzing event-related brain activity. By leveraging deep neural networks, the model learns to recognize patterns associated with epileptic events (e.g., seizures) and categorizes them into different trigger events.

### **Key Features**
- **Deep Learning Model**: Uses a neural network with Mish activation, batch normalization, and dropout for robust training.
- **Hyperparameter Optimization**: Uses Optuna to fine-tune model parameters for optimal performance.
- **Performance Metrics**: Evaluated using **accuracy, precision, recall, F1 score, and AUC-ROC**.
- **Feature Importance Analysis**: Uses **Integrated Gradients** and **permutation importance** to identify key EEG features.

## **Project Structure**
```
DeepEEG/
│
├── data/               # Processed EEG data (if included)
│   ├── sub-01_ses-01_connectivity.csv
│   ├── sub-01_ses-02_connectivity.csv
│   └── ...
│
├── notebooks/          # Jupyter notebooks for analysis
│   ├── DeepEEG_Classification.ipynb  # Main notebook
│   └── ...
│
├── models/             # Saved trained models (if any)
│   └── best_model.pth
│
└── README.md           # This file
```

## **Installation**
To run this project, ensure you have the following dependencies installed:

```bash
pip install mne-connectivity optuna tensorpac pytorch-ranger captum
```

## **Usage**
### **1. Data Loading & Preprocessing**
- Load EEG epochs using `load_epochs()`:
  ```python
  epochs = load_epochs("sub-01")  # Load subject data
  ```
- Compute spectral connectivity between EEG channels:
  ```python
  bands = {"Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 50)}
  session_dfs = compute_connectivity_per_session(epochs_list, bands, "sub-01")
  ```

### **2. Machine Learning (XGBoost)**
- Train an XGBoost classifier:
  ```python
  xgb = XGBClassifier(eval_metric='mlogloss', **best_params)
  xgb.fit(X_train, y_train)
  ```
- Evaluate performance:
  ```python
  y_pred = xgb.predict(X_test)
  print(classification_report(y_test, y_pred))
  ```

### **3. Deep Learning (PyTorch)**
- Train a neural network with Optuna hyperparameter tuning:
  ```python
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=50)
  ```
- Evaluate the best model:
  ```python
  best_model.eval()
  test_accuracy = evaluate_model(best_model, test_loader)
  ```

### **4. Feature Importance**
- **Integrated Gradients**:
  ```python
  ig = IntegratedGradients(best_model)
  attributions = ig.attribute(inputs, baseline)
  ```
- **Permutation Importance**:
  ```python
  importances = permutation_importance(best_model, test_loader, accuracy_score)
  ```

## **Results**
| Metric        | XGBoost | Neural Network |
|--------------|---------|----------------|
| Accuracy     | 0.92    | 0.94           |
| Precision    | 0.91    | 0.93           |
| Recall       | 0.92    | 0.94           |
| F1-Score     | 0.91    | 0.93           |
| AUC-ROC      | 0.98    | 0.99           |

## **Conclusion**
This project demonstrates the effectiveness of deep learning in classifying epileptic events from intracranial EEG data. Future improvements could include:
- **Larger datasets** for better generalization.
- **Real-time prediction** for clinical applications.
- **Explainable AI** techniques for better interpretability.

## **License**
This project is open-source under the **MIT License**. 

## **Contact**
For questions or collaborations, please open an issue or contact the repository owner.

