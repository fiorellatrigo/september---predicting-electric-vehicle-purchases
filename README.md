## Kaggle S6E9: EV Purchase Prediction

### 1. The Problem

Automakers and dealerships face inefficient marketing ROI when trying to identify high-propensity Electric Vehicle (EV) buyers. Generic marketing campaigns fail to account for the complex interplay between a consumer's financial readiness, commuting stress, and environmental concerns. This lack of targeting results in wasted ad spend and missed conversion opportunities for sales teams aiming to transition consumers to electric models.

### 2. Objective

The primary objective of this project is to build a highly optimized predictive classification model to forecast whether a consumer will purchase an EV (`Will_Buy_EV`). The goal is to maximize the ROC AUC metric to accurately rank potential buyers, ensuring that marketing and sales resources are directed toward the highest-probability leads.

### 3. Methodology

**Data & Target**

* **Source:** Kaggle Playground Series S6E9 competition dataset.
* **Features:** Tabular data encompassing demographic, logistical (daily commute, charging stations), and behavioral (environmental concern) variables.
* **Target Metric:** `Will_Buy_EV` (Binary classification evaluated via ROC AUC).

**Experimental & Validation Strategy**

* **Baseline Establishment:** Evaluated a tabular trifecta consisting of LightGBM, XGBoost, and CatBoost to establish strong out-of-fold (OOF) performance metrics.
* **Feature Engineering & Pruning:** Iteratively tested advanced techniques including Target Encoding, K-Means clustering, and domain-logic feature crossing (Eco Readiness and Commute Stress). Conducted Feature Importance evaluation to prune noisy, low-contributing variables (e.g., `Gender`, `Total_Charging_Stations`).
* **Hyperparameter Optimization:** Automated the tuning of the pruned dataset using Optuna (20 trials) to optimize tree depth, learning rate, and subsampling specifically for the streamlined feature space.
* **Ensemble Testing:** Experimented with Level-2 Stacking Classifiers and Multi-Layer Perceptrons (Neural Networks) to capture non-linearities missed by tree models.

### 4. Findings

**Experimental Performance Summary**

* **Feature Pruning Efficacy:** Removing weak predictors (`Total_Charging_Stations`, `Current_Car_Type`, `Gender`) successfully eliminated noise without degrading the baseline XGBoost ROC AUC (0.94175).
* **Algorithm Superiority:** Deep Learning (MLP) and Target Encoding severely underperformed on this specific tabular structure, causing the ROC AUC to drop below 0.940.
* **Final Optimization:** The pruned and Optuna-optimized XGBoost model achieved a peak local OOF ROC AUC of 0.94183 and a robust Public Leaderboard score of 0.94178.

### 5. Recommendations

* **Production Deployment:** Deploy the lightweight, pruned XGBoost model as the primary scoring engine for incoming leads to rank EV purchase propensity efficiently.
* **Data Collection Strategy:** Deprioritize the collection of broad demographic data (like Gender) in lead-gen forms, focusing instead on high-signal behavioral and financial indicators.
* **Pipeline Integration:** Utilize the saved `joblib` model objects to establish an automated daily scoring pipeline for the sales CRM.

### 6. Technologies

* **Language:** Python 3.10
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** XGBoost, LightGBM, CatBoost, Scikit-learn, TensorFlow/Keras
* **Optimization:** Optuna

**Project Structure**

```text
├── data/                  # Local directory for train and test CSV files
├── models/                # Serialized model objects (.pkl, .keras) and OOF predictions
├── notebook/              # Jupyter Notebook containing iterative modeling and optimization
├── README.md              # Executive summary and project documentation
└── submission.csv         # Final Kaggle submission file

```

**Installation & Environment Setup**
This project uses an isolated Python environment. To replicate this setup, run the following commands in your terminal:

1. Clone the repository

```bash
git clone https://github.com/fiorellatrigo/
september---predicting-electric-vehicle-purchases.git
cd september---predicting-electric-vehicle-purchases

```

2. Create and activate the virtual environment

```bash
# Windows:
python -m venv .venv
.\.venv\Scripts\activate

# Mac/Linux:
python -m venv .venv
source .venv/bin/activate

```

3. Install dependencies

```bash
pip install -r requirements.txt

```
