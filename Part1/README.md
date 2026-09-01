Video Recording: https://youtu.be/aU8Xfas0fwM 



# Part 1: NBA Player Salary Prediction using CRISP-DM Methodology

## 📋 Project Overview

This project applies the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to predict NBA player salaries based on historical performance statistics. It is structured as a comprehensive educational walkthrough designed for a master's-level data science program, demonstrating best practices in data science project management and machine learning.

**Objective:** Build a predictive model that estimates NBA player salaries from player statistics including scoring, rebounds, assists, efficiency metrics, and other performance indicators.

**Primary Model:** Gradient Boosting Regression  
**Test Performance:** RMSE = $5.83M, MAE = $3.69M, R² = 0.755  
**Dataset:** 7,296 NBA player-season records with real salaries from 2010–2025

The deployed interface separates the ML model's **historical contract
benchmark** from a **salary-cap-aware fair value**. The latter incorporates
peer-performance percentile, contract start year, years of service, MVP,
All-NBA and All-Star status, maximum-salary tier, and an illustrative four-year
contract value. See [FAIR_VALUE_METHOD.md](FAIR_VALUE_METHOD.md) for assumptions.

Read the polished project story in
[NBA_Fair_Value_Medium_Article.html](NBA_Fair_Value_Medium_Article.html).

Prepare and record the end-to-end project walkthrough with
[YOUTUBE_VIDEO_PRODUCTION_GUIDE.md](YOUTUBE_VIDEO_PRODUCTION_GUIDE.md).

---

## 🚀 Quick Start

### Run the Complete Pipeline (One Command)

```powershell
cd .\Part1
.\run.ps1
```

This single command will:
1. Execute all 7 CRISP-DM phases
2. Train and evaluate the best-performing model
3. Generate visualizations and reports
4. Launch the interactive web UI at **http://127.0.0.1:8006**

### Requirements

- **Python 3.10+** available on `PATH` (or in the active virtual environment)
- **Operating System:** Windows (PowerShell 5.1+)
- Required Python packages:
  - pandas, numpy
  - scikit-learn
  - matplotlib, seaborn
  - scipy.stats
  - joblib

`run.ps1` automatically installs or verifies these packages from `requirements.txt` before running the pipeline.

---

## 📊 CRISP-DM Methodology Overview

This project is structured around the 6 phases of the CRISP-DM framework:

### Phase 1: Business Understanding (Chunk 1)
- Define the business problem: predicting NBA salary trends
- Identify key performance drivers
- Establish success criteria and metrics
- Load and profile initial dataset

### Phase 2: Data Understanding (Chunk 2)
- **Exploratory Data Analysis (EDA)**
- Distribution analysis: histograms, boxplots
- Relationship analysis: correlation matrices, scatter plots
- Outlier identification and domain knowledge integration
- Feature importance initial assessment
- **Outputs:** 14 visualizations, EDA insights JSON

### Phase 3: Data Preparation (Chunk 3)
- Handle missing values (none in this dataset)
- Standardization: Z-score normalization of all statistics (mean=0, std=1)
- Player demographic encoding: Position and Team one-hot encoding
- Chronological train/test split (80/20) - ensures temporal integrity
- **Output:** 13 standardized NBA statistics ready for modeling

### Phase 4a: Baseline & Feature Selection (Chunk 4)
- Establish baseline with Linear Regression
- Apply three feature selection methods:
  - **SelectKBest:** Univariate statistical test
  - **RFE:** Recursive Feature Elimination
  - **Tree-based:** Random Forest feature importance
- Reduce from 34 → 13 optimal features
- **Best Features:** GP, MIN, PTS, FG%, 3PT%, FT%, REB, AST, STL, BLK, TO, PF, +/-

### Phase 4b: Multiple Algorithms & Hyperparameter Tuning (Chunk 5)
- Compare 8 regression algorithms:
  - Linear Regression, Ridge, Lasso, ElasticNet
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors, Random Forest, Gradient Boosting
- GridSearchCV for optimal hyperparameter tuning
- Overfitting analysis and model ranking
- **Winner:** Gradient Boosting with the best chronological holdout performance
- **Output:** Model comparison table, best model saved

### Phase 5: Evaluation & Diagnostics (Chunk 6)
- Comprehensive performance metrics: RMSE, MAE, R²
- Residual analysis: distribution, Q-Q plots, normality tests
- Actual vs. Predicted visualization
- Error quantiles: P90, P95 statistics
- Model diagnostics and assumption validation

### Phase 6: Deployment & Recommendations (Chunk 7)
- Summary of findings and business insights
- Model deployment guidance
- Limitations and recommendations for improvement
- Integration with web UI for practical usage

---

## 📁 Project Structure

```
Part1/
├── README.md                                    # This file
├── run.ps1                                      # One-command launcher (all 7 phases + UI)
│
├── CRISP-DM Chunk Scripts (Sequential):
├── 01_CRISP_DM_Chunk1_Business_and_Data_Understanding.py
├── 02_CRISP_DM_Chunk2_EDA_Visualization.py
├── 03_CRISP_DM_Chunk3_Data_Preparation.py
├── 04_CRISP_DM_Chunk4_Baseline_and_Feature_Selection.py
├── 05_CRISP_DM_Chunk5_Multiple_Algorithms.py
├── 06_CRISP_DM_Chunk6_Final_Evaluation_and_Diagnostics.py
├── 07_CRISP_DM_Chunk7_Deployment_and_Recommendations.py
│
├── Web UI & Model Server:
├── salary_ui_server.py                          # Interactive browser-based prediction interface
│
├── Output Reports (PDF & HTML):
├── NBA_Salary_Prediction_Medium_Article.pdf     # Research paper format (PDF)
├── NBA_Salary_Prediction_Medium_Article.html    # Web-readable version
│
├── data/                                         # Artifacts and datasets
│   ├── 01_raw_data.csv                          # Original dataset (500 samples, 13 NBA stats)
│   ├── 02_prepared_data_full.csv                # After data prep (standardized stats)
│   ├── 03_X_train_features.csv                  # Training features (400, 13)
│   ├── 03_y_train_target.csv                    # Training targets (400,)
│   ├── 04_X_test_features.csv                   # Test features (100, 13)
│   ├── 04_y_test_target.csv                     # Test targets (100,)
│   ├── 05_X_train_optimal.csv                   # Optimized training features (400, 13)
│   ├── 05_y_train_optimal.csv                   # Training targets (400,)
│   ├── 06_X_test_optimal.csv                    # Optimized test features (100, 13)
│   ├── 06_y_test_optimal.csv                    # Test targets (100,)
│   ├── best_model.pkl                           # Trained SVR model (serialized)
│   ├── best_model_metadata.json                 # Model hyperparameters & config
│   ├── model_results.csv                        # Performance metrics for all 8 algorithms
│   ├── model_comparison_summary.txt             # Algorithm rankings
│   ├── metadata.json                            # Dataset metadata
│   ├── 06_evaluation_summary.txt                # Final evaluation report
│   ├── 06_model_evaluation_report.json          # Detailed JSON evaluation
│   ├── preparation_metadata.json                # Feature engineering details
│   ├── optimal_features_metadata.json           # Selected feature list
│   └── 07_deployment_recommendations.txt        # Business recommendations
│
└── visualizations/                              # Generated plots
    ├── Numeric_Features_Distributions.png
    ├── Categorical_Features_Analysis.png
    ├── Correlation_Matrix.png
    ├── Target_Variable_Distribution.png
    ├── Relationships_with_Target.png
    ├── Data_Preparation_Summary.png
    ├── Feature_Selection_Analysis.png
    ├── Algorithm_Comparison.png
    ├── 06_actual_vs_predicted.png
    ├── 06_residuals_vs_fitted.png
    ├── 06_qq_plot.png
    ├── 06_residual_distribution.png
    ├── eda_insights.json
    └── data_quality_report.csv
```

---

## 📈 Model Performance

### Best Model: Gradient Boosting Regression

```
Test RMSE (Root Mean Squared Error):  $5.8258M
Test MAE (Mean Absolute Error):       $3.6881M
Test R² (Coefficient of Determination): 0.7546

Hyperparameters:
  - Learning rate: 0.05
  - Maximum depth: 3
  - Estimators: 200
  - Subsample: 1.0
```

### All Algorithms Compared

| Algorithm | Test RMSE | Test R² | Overfitting |
|-----------|-----------|---------|-------------|
| **Gradient Boosting** | **5.8258** | **0.7546** | **0.0078** |
| Random Forest | 5.8477 | 0.7527 | 0.1390 |
| SVR | 5.9720 | 0.7421 | 0.0082 |
| KNN | 6.5013 | 0.6943 | 0.3057 |
| Linear Regression | 7.6777 | 0.5737 | 0.0201 |
| Ridge | 7.6778 | 0.5737 | 0.0201 |
| Lasso | 7.6783 | 0.5736 | 0.0200 |
| ElasticNet | 7.6862 | 0.5728 | 0.0193 |

**Selection Criteria:**
- Best test RMSE ($5.8258M)
- Best test R² (0.7546)
- Small train/test R² gap (0.0078)
- Evaluated on the newest chronological holdout seasons

### Residual Analysis

```
Mean:          -0.012
Std Dev:       0.630
Min:           -6.217
Max:           0.220
Median:        0.048

Error Quantiles:
  - Mean Abs Error (MAE):     $0.124M
  - Median Abs Error:         $0.051M
  - 90th Percentile Error:    $0.136M
  - 95th Percentile Error:    $0.153M
```

**Key Insights:**
- Residuals approximately normally distributed
- Some heteroscedasticity present (expected for salary data)
- Model provides reasonable salary estimates within ±$0.12M tolerance

---

## 🎯 Key Features (13 NBA Statistics)

The model uses these standard NBA performance statistics:

1. **GP** - Games Played (durability/availability)
2. **MIN** - Minutes per Game (playing time)
3. **PTS** - Points per Game (scoring volume)
4. **FG%** - Field Goal Percentage (shooting efficiency)
5. **3PT%** - 3-Point Percentage (modern shooting skill)
6. **FT%** - Free Throw Percentage (shooting consistency)
7. **REB** - Rebounds per Game (rebounding ability)
8. **AST** - Assists per Game (playmaking ability)
9. **STL** - Steals per Game (defensive skill)
10. **BLK** - Blocks per Game (shot-blocking defense)
11. **TO** - Turnovers per Game (ball security)
12. **PF** - Personal Fouls (foul propensity)
13. **+/-** - Plus/Minus (on-court impact)

---

## 💻 Web UI: Interactive Model Usage

### Launch the Web UI

The UI starts automatically when you run `.\run.ps1`, but you can also launch it separately:

```powershell
cd .\Part1
python salary_ui_server.py
```

Then open your browser to: **http://127.0.0.1:8006**

### Using the UI

1. **Enter Player Statistics:** Fill in the 13 NBA stat values using a simple form
2. **Submit Prediction:** Click "Predict Salary"
3. **View Estimate:** The model returns an estimated salary in millions of dollars

### Example Input

```
GP (Games Played):           82
MIN (Minutes per Game):      28.0
PTS (Points per Game):       18.5
FG% (Field Goal %):          0.450
3PT% (3-Point %):            0.350
FT% (Free Throw %):          0.800
REB (Rebounds per Game):     4.1
AST (Assists per Game):      3.2
STL (Steals per Game):       1.1
BLK (Blocks per Game):       1.5
TO (Turnovers per Game):     2.3
PF (Personal Fouls):         2.8
+/- (Plus/Minus):           +2.5
```

**Predicted Salary:** ~$8.5M annually

---

## 📖 Running Individual Chunks

If you want to run specific CRISP-DM phases:

```powershell
# Phase 1: Business Understanding
python 01_CRISP_DM_Chunk1_Business_and_Data_Understanding.py

# Phase 2: Exploratory Data Analysis
python 02_CRISP_DM_Chunk2_EDA_Visualization.py

# Phase 3: Data Preparation
python 03_CRISP_DM_Chunk3_Data_Preparation.py

# Phase 4a: Baseline & Feature Selection
python 04_CRISP_DM_Chunk4_Baseline_and_Feature_Selection.py

# Phase 4b: Multiple Algorithms
python 05_CRISP_DM_Chunk5_Multiple_Algorithms.py

# Phase 5: Final Evaluation
python 06_CRISP_DM_Chunk6_Final_Evaluation_and_Diagnostics.py

# Phase 6: Deployment
python 07_CRISP_DM_Chunk7_Deployment_and_Recommendations.py
```

**Note:** Scripts should be run in order, as each depends on outputs from previous phases.

---

## 📊 Understanding the Outputs

### Visualizations (14 PNG files)

- **Numeric_Features_Distributions.png** - Histograms and distributions of numeric features
- **Categorical_Features_Analysis.png** - Position and Team breakdowns
- **Correlation_Matrix.png** - Heatmap of feature correlations
- **Target_Variable_Distribution.png** - Salary distribution analysis
- **Relationships_with_Target.png** - Scatter plots showing feature-salary relationships
- **Data_Preparation_Summary.png** - Before/after data transformation
- **Feature_Selection_Analysis.png** - Feature importance rankings
- **Algorithm_Comparison.png** - Bar chart comparing 8 algorithms
- **06_actual_vs_predicted.png** - Model predictions vs. actual salaries
- **06_residuals_vs_fitted.png** - Residual scatter plot
- **06_qq_plot.png** - Q-Q plot for normality assessment
- **06_residual_distribution.png** - Histogram of residuals

### Data Files

- **01_raw_data.csv** - Original 500 samples
- **02_prepared_data_full.csv** - After data preparation (13 standardized NBA stats)
- **05_X_train_optimal.csv / 06_X_test_optimal.csv** - Final 13-stat matrices
- **best_model.pkl** - Trained SVR model (load with joblib.load())
- **model_results.csv** - Detailed metrics for all 8 algorithms

### JSON Reports

- **best_model_metadata.json** - Model configuration and hyperparameters
- **06_model_evaluation_report.json** - Comprehensive evaluation metrics
- **eda_insights.json** - Statistical insights from EDA phase
- **preparation_metadata.json** - Feature engineering transformation details

### Text Reports

- **06_evaluation_summary.txt** - Executive summary of model performance
- **model_comparison_summary.txt** - Algorithm rankings and analysis
- **07_deployment_recommendations.txt** - Business recommendations

---

## 🔍 Model Interpretation

### What the Model Learns

The SVR model creates a non-linear decision boundary that captures:

1. **Scoring Impact** - Points Per Game (PTS) is a major salary driver
2. **Defensive Value** - Blocks (BLK) and Steals (STL) add premium value
3. **Efficiency** - Ball security (low Turnovers/TO) is valued
4. **Playing Time** - Minutes per Game (MIN) indicates role and market value
5. **Shooting Skills** - Field goal %, 3-point %, and free throw % all matter
6. **Availability** - Games Played (GP) shows durability
7. **All-around Contribution** - Rebounds, Assists, and on-court impact (+/-) round out the picture

### Limitations

- **No Exact Prediction:** Model is approximate; salaries are influenced by non-statistical factors (brand, market, agent negotiation)
- **Historical Data:** Based on past salary trends; new contracts may differ
- **Benchmark Only:** Use for salary ballpark estimates, not precise contract prediction
- **Market Dynamics:** Doesn't capture salary cap changes or roster construction strategies

### Typical Error Range

- **68% of predictions** within ±$0.625M of actual (1 std dev)
- **95% of predictions** within ±$1.26M of actual (2 std dev)
- **P90 error** at $0.136M; **P95 error** at $0.153M

---

## 🎓 Learning Objectives

This project demonstrates:

1. **CRISP-DM Workflow:** Complete end-to-end data science methodology
2. **EDA Best Practices:** Systematic exploration and visualization
3. **Feature Engineering:** Domain-informed feature creation and selection
4. **Model Comparison:** Systematic evaluation of multiple algorithms
5. **Hyperparameter Tuning:** GridSearchCV and cross-validation
6. **Model Diagnostics:** Residual analysis and assumption testing
7. **Deployment:** Web UI for practical model usage
8. **Reproducibility:** Automated pipeline with serialized model

---

## 📚 References & Resources

### CRISP-DM Framework
- [CRISP-DM Wikipedia](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- [IBM CRISP-DM Guide](https://www.ibm.com/cloud/learn/crisp-dm)

### Scikit-Learn Documentation
- [Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Feature Selection Methods](https://scikit-learn.org/stable/modules/feature_selection.html)

### Python Data Science Stack
- Pandas: Data manipulation and analysis
- NumPy: Numerical computing
- Scikit-Learn: Machine learning algorithms
- Matplotlib & Seaborn: Visualization

---

## 🔧 Troubleshooting

### Python Path Issues

If `run.ps1` fails with "python not found":

1. Update the Python executable path in `run.ps1`:
   ```powershell
   $pythonExe = "C:/Users/James/AppData/Local/Python/pythoncore-3.14-64/python.exe"
   ```

2. Or use global Python:
   ```powershell
   $pythonExe = "python"
   ```

### Web UI Won't Start

1. Check if port 8006 is available
2. Run directly: `python salary_ui_server.py`
3. Verify all data files exist in the `data/` folder

### Missing Dependencies

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib
```

---

## 📝 File Naming Convention

All generated artifacts follow a naming pattern:

- **01_* :** Chunk 1 outputs (Business Understanding)
- **02_* :** Chunk 2 outputs (EDA)
- **03_* :** Chunk 3 outputs (Data Preparation)
- **04_* :** Chunk 4 outputs (Baseline & Feature Selection)
- **05_* :** Chunk 5 outputs (Multiple Algorithms)
- **06_* :** Chunk 6 outputs (Evaluation & Diagnostics)
- **07_* :** Chunk 7 outputs (Deployment)

This naming ensures artifacts are automatically organized by CRISP-DM phase.

---

## 🎯 Next Steps

1. **Run the complete pipeline:** `.\run.ps1`
2. **Explore visualizations:** Open PNG files in `visualizations/`
3. **Read the report:** Open `NBA_Salary_Prediction_Medium_Article.pdf`
4. **Try the web UI:** Navigate to `http://127.0.0.1:8006`
5. **Experiment:** Modify player statistics to see salary predictions
6. **Learn:** Review each chunk script to understand CRISP-DM implementation

---

## 📄 License & Attribution

This project is part of CMPE255 (Data Mining) Master's assignment at San José State University.

**Created:** 2026  
**Framework:** CRISP-DM Methodology  
**Institution:** San José State University - College of Engineering  

---

## ✅ Checklist for Using This Project

- [ ] Python 3.14+ installed
- [ ] All dependencies installed (`pip install pandas numpy scikit-learn ...`)
- [ ] Run `.\run.ps1` from Part1 directory
- [ ] Wait for model training (2-5 minutes depending on system)
- [ ] Open web UI at http://127.0.0.1:8006
- [ ] Review visualizations in `visualizations/` folder
- [ ] Read PDF report: `NBA_Salary_Prediction_Medium_Article.pdf`
- [ ] Experiment with salary predictions in the web interface

---

**Last Updated:** September 2026  
**Status:** ✅ Complete and Ready for Use
