# 🛢️ Volve Reservoir Anomaly Detection & Production Forecasting Pipeline

An end-to-end Machine Learning pipeline applied to historical production logs from Equinor's public **Volve Field dataset**. This project combines operational anomaly classification (predictive maintenance) with time-series production forecasting, benchmarking tree-based ML against traditional empirical Arps Decline Curve Analysis (DCA).

---

## 📌 Executive Summary

* **Predictive Maintenance Model:** Built an anomaly detection pipeline to identify operational disturbances (unplanned shut-ins, pressure drops) and water breakthrough events, achieving an **ROC-AUC score of 0.916**.
* **Production Forecasting Model:** Formulated a time-series regression model predicting long-term oil output. Achieved an **$R^2$ score of 0.469**, outperforming industry-standard empirical Arps Decline Curve Analysis ($R^2 = 0.255$) by **~84%**.
* **Imbalance & Optimization:** Leveraged SMOTE oversampling and Stratified Cross-Validation to handle severe target imbalance in operational failure events.

---

## 📊 Key Performance Metrics

### 1. Anomaly Classification (Predictive Maintenance)
| Metric | Class 0 (Normal Operation) | Class 1 (Anomaly / Failure) | Overall |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.47 | 0.95 | — |
| **Recall** | 0.67 | 0.91 | — |
| **F1-Score** | 0.55 | **0.93** | — |
| **ROC-AUC** | — | — | **0.916** |

### 2. Time-Series Production Forecasting Benchmark
| Model | Target Variable | $R^2$ Score | Performance Comparison |
| :--- | :--- | :---: | :--- |
| **Classical Arps DCA** | Daily Oil Volume (`BORE_OIL_VOL`) | 0.255 | Baseline empirical model |
| **Tuned XGBoost Regressor** | Daily Oil Volume (`BORE_OIL_VOL`) | **0.469** | **+84% improvement** over Arps |

---

## 🛠️ Tech Stack & Methods

* **Language & Environment:** Python 3.x, Google Colab / Jupyter Notebooks
* **Data Processing & Analytics:** `pandas`, `numpy`, `scipy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE)
* **Visualization:** `matplotlib`, `seaborn`

---

## 🚀 Workflow Architecture

1. **Data Preprocessing & Cleaning:** Filtered active production windows (`ON_STREAM_HRS > 0`) and extracted reservoir depletion parameters across Volve wellbores.
2. **Feature Engineering:** Calculated dynamic fluid ratios (GOR, WOR) and time-series rolling statistics to capture transient reservoir depletion behavior.
3. **Imbalance Handling:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to rebalance rare downhole operational failure classes.
4. **Benchmarking:** Evaluated hyperparameter-tuned XGBoost against non-linear hyperbolic Arps decline curve fits.

---
