# 🏠 Advanced Real Estate Valuation using ML Stacking

**Predicting home prices smarter — with a blend of EDA, feature engineering, and a custom stacked regressor to outperform standard models by 6.5%.**  

## 📌 Project Summary

Buying or selling a home involves one big question: *What’s it really worth?*

This project tackles that with machine learning — using **Stacked Regression models** (Lasso, Ridge, ElasticNet, LightGBM) to estimate house prices with **high accuracy and interpretability**. With extensive feature engineering and domain-aware preprocessing, the final model achieved a **6.5% improvement in prediction accuracy** on a dataset of 81 features.

> ✨ **Business Impact**: Helps buyers, sellers, and agents make informed decisions, improves market efficiency, and supports smarter real estate investments.

---

## 🚀 Highlights

| Feature | Description |
|--------|-------------|
| 🔍 **Exploratory Data Analysis (EDA)** | Uncovered key valuation drivers like `GrLivArea`, `OverallQual`, and neighborhood impact through plots and heatmaps. |
| 🧹 **Data Cleaning** | Addressed missing data using smart domain-specific imputations (e.g., median LotFrontage by neighborhood). |
| 🧠 **Feature Engineering** | Handled 57 categorical columns (including 27 ordinal ones) with Label Encoding and One-Hot Encoding. |
| ⚙️ **Transformations** | Applied log-normalization and Box-Cox to reduce skewness, boosting model precision by 10%. |
| 🔗 **Model Stacking** | Combined Ridge, Lasso, ElasticNet, and LightGBM under a Ridge meta-model. |
| 📈 **Performance** | Achieved the lowest RMSE and MAE compared to other baselines like Random Forest and Gradient Boosting. |
| 🛠️ **Tech Stack** | `Pandas`, `Seaborn`, `Scikit-learn`, `LightGBM`, `Matplotlib`, `Joblib` |

---

## 📊 Results

| Model | RMSE | MAE |
|-------|------|-----|
| Linear Regression | 25,754 | 18,840 |
| Random Forest | 24,208 | 16,723 |
| Gradient Boosting | 21,156 | 14,926 |
| 🏆 **Stacked Regressor** | **~19,800** | **~13,900** (approx., validated through cross-validation) |

---

## 🧐 Skills Demonstrated

- **Machine Learning**: Model selection, hyperparameter tuning (GridSearchCV), model ensembling
- **Data Engineering**: Cleaning, imputation, outlier detection, label and one-hot encoding
- **Statistical Techniques**: Box-Cox transformation, log-normalization, skew analysis
- **Model Evaluation**: RMSE, MAE, k-fold cross-validation
- **Data Visualization**: Correlation heatmaps, boxplots, scatter plots
- **Tools**: `Jupyter`, `Git`, `Python`, `LightGBM`, `Scikit-learn`, `Joblib`

---

## 🧪 How It Works

1. **Preprocessing**:
   - Drop high-null columns (>15%)
   - Fill numerical with medians, categorical with modes or "None"
   - Identify & remove domain-defined outliers

2. **Transformation**:
   - Normalize skewed features with Box-Cox
   - Apply log transform to target (`SalePrice`)

3. **Modeling**:
   - Encode categorical features
   - Scale numerical features
   - Train multiple models and evaluate performance
   - Stack top-performing models into final regressor

4. **Prediction**:
   - Test dataset predictions stored with all model outputs for comparison.

---

## 🔮 Future Work

- Addressing multicollinearity with advanced feature selection (e.g., VIF)
- Integrating time-series trends (year of sale, month effects)
- Augmenting with external data like economic indices or property tax records
- Deployment with Flask/Streamlit for real-time user interaction
