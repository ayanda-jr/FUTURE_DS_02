# FUTURE_DS_02 — Customer Retention & Churn Analysis

**Task:** Data Science & Analytics – Task 2 (2026) — Future Interns  
**Dataset:** Telco Customer Churn (Kaggle) — 7,043 customers, 21 features  
**Tools:** Power BI, Python (scikit-learn), DAX

## Repository Structure

```
FUTURE_DS_02/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── telco-churn-analysis.pbix               # Power BI dashboard (build from guide)
├── README.md
├── python/
│   ├── churn_model.py                     # Standalone churn prediction model
│   ├── powerbi_churn_visual.py            # Power BI Python visual script
│   └── requirements.txt                   # Python dependencies
└── docs/
    ├── powerbi-build-guide.md             # Step-by-step Power BI setup
    └── insights-report.md                 # Stakeholder insights & recommendations
```

## How to Use

### Power BI Dashboard
1. Open `WA_Fn-UseC_-Telco-Customer-Churn.csv` in Power BI Desktop
2. Follow `docs/powerbi-build-guide.md` for data cleaning, DAX measures, and visual setup
3. For the predictive model visual, use `python/powerbi_churn_visual.py` as the Python script
4. Save as `telco-churn-analysis.pbix`

### Standalone Python Model
```bash
cd python
pip install -r requirements.txt
python churn_model.py
```
Outputs: `model_performance.png`, `feature_importance.png`

## Key Findings

- **26.5%** overall churn rate (1,869 of 7,043 customers)
- **42.7%** churn for month-to-month contracts vs **2.8%** for two-year
- **40%** of churn happens in the first 6 months
- Customers without Online Security churn at **41.8%** vs **14.6%** with it
- Predictive model (Random Forest) achieves **0.85 AUC-ROC**

## Deliverables

- Interactive Power BI dashboard with 5 analysis pages
- Predictive churn model with feature importance
- Stakeholder insights report with 6 actionable recommendations

## About

Customer retention analysis for Future Interns Data Science & Analytics — Task 2 (2026).
