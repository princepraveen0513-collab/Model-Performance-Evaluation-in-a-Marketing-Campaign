# W06 – Marketing Campaign Model Evaluation

## Understanding how to assess a Machine Learning model: Key performance metrics and their implications explored.

**Notebook:** `modelEvaluation.ipynb`  
**Title (in notebook):** Assignment : Model Performance Evaluation in a Marketing Campaign
This project evaluates and compares multiple classification models for a **marketing campaign response** task using a **synthetic dataset**. It demonstrates end-to-end steps: data generation, preprocessing, model selection with cross-validation, and performance evaluation via metrics and diagnostic plots.
---
## 🎯 Objectives
- Generate a **synthetic binary classification** dataset suitable for a marketing scenario.
- Train and compare **Logistic Regression**, **Support Vector Machine (SVC)**, and **K-Nearest Neighbors (KNN)**.
- Use **`GridSearchCV`** to tune hyperparameters (e.g., KNN neighbors; LR/SVM grids defined in the notebook).
- Evaluate models with **accuracy, precision, recall, F1, ROC-AUC**, and visualize **confusion matrices, ROC, and PR curves**.
---
## 🧪 Data
- Created via **`sklearn.datasets.make_classification`** with:
  - `n_samples = 1000`, `n_features = 2`
  - `n_informative = 2`, `n_redundant = 0`
  - `n_clusters_per_class = 1`
  - `random_state = 42`
- Split: **70/30** with `train_test_split(..., test_size=0.3, random_state=42)`
- **Standardization** with `StandardScaler()` applied to train/test features.

> This setup provides a clean, controlled benchmark to focus on modeling and evaluation techniques.
---
## 🧠 Models & Tuning
- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- **SVM (SVC)** (`sklearn.svm.SVC`)
- **KNN** (`sklearn.neighbors.KNeighborsClassifier`)
  - Example grid (from the notebook):
    ```python
    param_grid_knn = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
    }
    ```

- **Search strategy:** `GridSearchCV` with `random_state=42` used in splits; see notebook cells for each model’s search setup and best parameters.
---
## 📈 Evaluation
**Metrics computed in the notebook:**
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`
- `confusion_matrix` and plots for **Confusion Matrix**, **ROC**, and **Precision–Recall**

> Run the notebook to reproduce the exact numbers. Results may vary slightly across environments due to solver/kernel details.

## 📁 Repository Structure
```
.
├── modelEvaluation.ipynb        # Main analysis notebook
└── README.md              # This file
```

> Plots are generated inline by the notebook. If you want them in the repo, add an `images/` folder and save figures there.
---
## ⚙️ Environment & Setup
**Core Libraries:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`

Install:
```bash
pip install -U numpy pandas scikit-learn matplotlib
```

Run:
```bash
jupyter notebook "modelEvaluation.ipynb"
```
---
## 🧭 Interpretation & Tips
- With only **two informative features**, decision boundaries are easy to visualize—use the ROC/PR plots to compare models beyond accuracy.
- For **KNN**, scaling matters; keeping `StandardScaler` in the pipeline is important.
- For **SVM**, try both linear and RBF kernels; tune `C` (and `gamma` for RBF).
- For **Logistic Regression**, adjust `C` and consider penalty/solver options if you extend the grid.
- If moving to real marketing data, consider:
  - **Class imbalance** handling (stratified CV, class weights, resampling)
  - **Feature engineering** (interaction terms, embeddings for categorical variables)
  - **Threshold tuning** based on campaign costs (optimize precision/recall trade-off)
---
## 🔁 Reproducibility
- Deterministic dataset generation and split via `random_state=42`.
- Hyperparameter search defined in the notebook to document chosen settings.
---
## 📚 References
- scikit-learn: datasets, preprocessing, model selection, and metrics
- General guides on model evaluation and ROC/PR analysis
