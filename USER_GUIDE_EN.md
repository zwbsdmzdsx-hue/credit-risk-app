# Credit Risk Assessment System - User Guide

**COMP5572 Consumer Credit Risk Prediction Project**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Getting Started](#2-getting-started)
3. [Single Customer Assessment](#3-single-customer-assessment)
4. [Batch Prediction](#4-batch-prediction)
5. [Model Report](#5-model-report)
6. [Prediction History](#6-prediction-history)
7. [System Settings](#7-system-settings)
8. [FAQ](#8-faq)

---

## 1. System Overview

### 1.1 Features

This system is a **machine learning-based credit risk assessment tool** that provides:

| Feature | Description |
|---------|-------------|
| ✅ Single Assessment | Input customer info, get real-time default probability and risk level |
| ✅ Batch Prediction | Upload CSV file, process multiple customers at once |
| ✅ Decision Explanation | SHAP analysis shows which factors affect risk |
| ✅ Model Transparency | View model performance and feature importance |
| ✅ History Records | Automatically saves all prediction records |

### 1.2 Technical Background

- **Algorithm**: LightGBM Gradient Boosting
- **Training Data**: 108,657 real credit records
- **Accuracy**: AUC = 0.76+ (Out-of-Fold)
- **Languages**: English / 中文

---

## 2. Getting Started

### 2.1 Launch the Application

**Local Run**:
```bash
cd app
streamlit run app.py
```

Or use Python module:
```bash
python -m streamlit run app.py
```

The application will open automatically in your browser (or visit `http://localhost:8501` manually).

### 2.2 Interface Overview

> **[Screenshot Placeholder 1: Main Interface]**
> Description: Initial application view showing sidebar and main content area

**Interface Components**:

```
┌─────────────────────────────────────────────────────────┐
│  🏦 Credit Risk Assessment                              │
├─────────────┬───────────────────────────────────────────┤
│             │                                           │
│  Sidebar    │         Main Content Area                 │
│             │                                           │
│  • Pages    │    Displays current page content           │
│  • Settings │                                           │
│  • Model    │                                           │
│             │                                           │
└─────────────┴───────────────────────────────────────────┘
```

**Sidebar Functions**:
- Page Selection: Switch between different pages
- Advanced Settings: Language toggle, threshold adjustment
- Model Info: View current model performance

---

## 3. Single Customer Assessment

### 3.1 Access the Page

Click **📝 Single Prediction** in the sidebar.

> **[Screenshot Placeholder 2: Single Prediction Page Initial State]**
> Description: Initial state of single customer prediction page with all input forms

### 3.2 Enter Customer Information

#### 3.2.1 Basic Information

> **[Screenshot Placeholder 3: Basic Information Input Area]**
> Description: Expanded "Basic Information" form with gender, age, marital status fields

| Field | Description | Range |
|------|-------------|-------|
| Gender | Customer gender | Male / Female |
| Age | Customer age (years) | 18-100 |
| Marital Status | Marital status | Married, Single, Civil Marriage, Widow, Separated |
| Children | Number of children | 0-20 |
| Family Members | Total family size | 1+ |

**Note**: Age and employment duration are automatically converted to negative days (internal format).

#### 3.2.2 Financial Information

> **[Screenshot Placeholder 4: Financial Information Input Area]**
> Description: Expanded "Financial Information" form with income, loan amount, monthly payment

| Field | Description | Unit |
|------|-------------|------|
| Annual Income | Customer annual income | Currency |
| Loan Amount | Loan application amount | Currency |
| Monthly Payment | Monthly payment amount | Currency |
| Goods Price | Purchase price | Currency |

**Calculation Formulas**:
- Payment-to-Income Ratio = Monthly Payment / Annual Income
- Credit-to-Income Ratio = Loan Amount / Annual Income

#### 3.2.3 Employment Information

> **[Screenshot Placeholder 5: Employment Information Input Area]**
> Description: Expanded "Employment Information" form with income type, occupation, work years

| Field | Description | Options |
|------|-------------|--------|
| Income Type | Source of income | Working, State servant, Pensioner, Commercial associate |
| Occupation Type | Occupation category | Laborers, Core staff, Sales staff, Managers, Drivers, Other |
| Employment Years | Current work duration | 0-50 years |

#### 3.2.4 Asset Information

> **[Screenshot Placeholder 6: Asset Information Input Area]**
> Description: Expanded "Asset Information" form with property, vehicle, etc.

| Field | Description | Options |
|------|-------------|--------|
| Owns Car | Vehicle ownership | Y / N |
| Owns Realty | Property ownership | Y / N |
| Car Age | Vehicle usage years | 0+ years |
| Housing Type | Living situation | House/apartment, Rented, With parents |

#### 3.2.5 External Credit Scores

> **[Screenshot Placeholder 7: External Credit Scores Input Area]**
> Description: Expanded "External Credit Scores" form with three sliders

| Field | Description | Range |
|------|-------------|-------|
| EXT_SOURCE_1 | External credit score 1 | 0.0 - 1.0 |
| EXT_SOURCE_2 | External credit score 2 | 0.0 - 1.0 |
| EXT_SOURCE_3 | External credit score 3 | 0.0 - 1.0 |

**Tip**: These are comprehensive scores from external credit bureaus, similar to credit scores. Higher scores indicate better credit.

### 3.3 Start Assessment

#### 3.3.1 Submit Assessment

After filling all information, click **🔍 Start Assessment** button.

> **[Screenshot Placeholder 8: Assessment Result Interface]**
> Description: Complete assessment results showing default probability, risk level, decision recommendation

#### 3.3.2 Result Interpretation

**Top Metric Cards**:
- **Default Probability**: Customer's default likelihood (percentage)
- **Risk Level**: Low Risk ✅ / Medium Risk ⚠️ / High Risk ❌
- **Decision**: Approve / Review Required / Reject

> **[Screenshot Placeholder 9: Risk Level Gauge]**
> Description: Colored risk level gauge (green=low risk, yellow=medium risk, red=high risk)

**Risk Level Classification**:

| Default Probability | Risk Level | Decision | Color |
|---------------------|-----------|----------|-------|
| < Threshold×33% | Low Risk | Approve | 🟢 Green |
| Threshold×33% ~ Threshold | Medium Risk | Review | 🟡 Yellow |
| ≥ Threshold | High Risk | Reject | 🔴 Red |

**Default Threshold**: 0.24 (24%)

#### 3.3.3 Decision Analysis

> **[Screenshot Placeholder 10: SHAP Decision Analysis]**
> Description: "Positive Factors" and "Risk Factors" lists

**Positive Factors** (green ✅): Factors that reduce default risk
- Example: High income, stable job, property ownership

**Risk Factors** (red ⚠️): Factors that increase default risk
- Example: Low income, high monthly payment, low credit score

**SHAP Value Interpretation**:
- Negative value = Reduces risk (favorable)
- Positive value = Increases risk (unfavorable)
- Larger absolute value = Greater impact

### 3.4 Quick Test

Click **📋 Load Sample Data** to auto-fill test data for quick experience.

---

## 4. Batch Prediction

### 4.1 Access Batch Prediction Page

Click **📤 Batch Prediction** in the sidebar.

> **[Screenshot Placeholder 11: Batch Prediction Page Initial State]**
> Description: Three-step interface of batch prediction page

### 4.2 Step 1: Download CSV Template

> **[Screenshot Placeholder 12: CSV Template Download Button]**
> Description: "Download CSV Template" button

Click **📥 Download CSV Template** to get the standard template file.

**Template Fields**:
```
SK_ID_CURR, CODE_GENDER, DAYS_BIRTH, NAME_EDUCATION_TYPE,
NAME_FAMILY_STATUS, CNT_CHILDREN, CNT_FAM_MEMBERS,
AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
NAME_INCOME_TYPE, OCCUPATION_TYPE, DAYS_EMPLOYED,
FLAG_OWN_CAR, FLAG_OWN_REALTY, OWN_CAR_AGE,
NAME_HOUSING_TYPE, REGION_POPULATION_RELATIVE,
REGION_RATING_CLIENT, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3,
...
```

### 4.3 Step 2: Upload Data File

> **[Screenshot Placeholder 13: File Upload Area]**
> Description: CSV file upload interface

1. Prepare CSV file matching template format
2. Click **Select CSV file** to upload
3. System displays data preview and row count

> **[Screenshot Placeholder 14: Data Preview Table]**
> Description: First few rows preview of uploaded data

### 4.4 Step 3: Start Prediction

> **[Screenshot Placeholder 15: Batch Prediction Result Interface]**
> Description: Statistics and result table after batch prediction completion

Click **🚀 Batch Predict** to start processing.

**Prediction Results**:
- **Summary**: Low/Medium/High risk customer counts
- **Result Table**: Contains SK_ID_CURR, TARGET_PROB, RISK_LEVEL, DECISION

### 4.5 Download Results

Click **💾 Download Results CSV** to save prediction results.

---

## 5. Model Report

### 5.1 View Model Report

Click **📊 Model Report** in the sidebar.

> **[Screenshot Placeholder 16: Model Report Page]**
> Description: Model performance metrics, feature importance, fairness analysis

### 5.2 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| AUC | ~0.76 | Model discrimination ability (0.5=random, 1.0=perfect) |
| F1 Score | ~0.25 | Harmonic mean of precision and recall |
| Best Threshold | 0.24 | Optimal decision boundary for classification |

> **[Screenshot Placeholder 17: Model Performance Metric Cards]**
> Description: Three Metric cards showing performance indicators

### 5.3 Feature Importance

> **[Screenshot Placeholder 18: Feature Importance Bar Chart]**
> Description: Horizontal bar chart of Top 15 features

**Most Important Features**:
1. EXT_SOURCE_2/3 - External credit scores
2. AMT_ANNUITY - Monthly payment amount
3. DAYS_EMPLOYED - Employment duration
4. DAYS_BIRTH - Age

**Interpretation**: Higher feature importance means greater impact on default risk.

### 5.4 Fairness Analysis

> **[Screenshot Placeholder 19: Fairness Analysis Metrics]**
> Description: Gender DI and Age DI assessment results

| Metric | Value | Status |
|--------|-------|--------|
| Gender DI | 0.8139 | ✅ Acceptable |
| Age DI | 0.7743 | ⚠️ Needs Improvement |

**DI (Disparate Impact Ratio)**:
- DI ≥ 0.8: No significant discrimination
- DI < 0.8: Discrimination risk exists

---

## 6. Prediction History

### 6.1 View History Records

Click **📜 Prediction History** in the sidebar.

> **[Screenshot Placeholder 20: Prediction History Page]**
> Description: History record list and statistics

**Summary Statistics**:
- Total Records
- Single Predictions count
- Batch Predictions count

### 6.2 History Record Details

Each record contains:
- **Timestamp**: Prediction time
- **Mode**: single / batch
- **Input Summary**: Customer basic information
- **Result**: Default probability, risk level, decision

> **[Screenshot Placeholder 21: Expanded History Record Detail]**
> Description: One expanded history record with complete information

### 6.3 Clear History

Click **🗑️ Clear History** to delete all history data.

---

## 7. System Settings

### 7.1 Advanced Settings

Click **⚙️ Advanced Settings** in the sidebar.

> **[Screenshot Placeholder 22: Advanced Settings Expanded]**
> Description: Language selection and threshold adjustment settings

### 7.2 Language Switching

**Language / 语言**:
- 🇨🇳 中文
- 🇺🇸 English

Interface updates immediately after switching language.

> **[Screenshot Placeholder 23: Chinese/English Interface Comparison]**
> Description: Side-by-side comparison of same page in Chinese and English

### 7.3 Threshold Adjustment

**Classification Threshold**: Controls risk level strictness

| Threshold | Effect |
|-----------|--------|
| Smaller value | Stricter (more customers marked as high risk) |
| Larger value | More lenient (more customers marked as low risk) |

**Default Value**: 0.24 (24%)

**Adjustment Method**: Drag slider to adjust threshold

> **[Screenshot Placeholder 24: Threshold Adjustment Slider]**
> Description: Threshold adjustment slider control

**Impact**:
- Changes low/medium/high risk boundaries
- Affects all prediction results in real-time

### 7.4 Model Information

Model information displayed at sidebar bottom:
- **AUC**: Model accuracy
- **F1**: Overall score
- **Threshold**: Current threshold value

> **[Screenshot Placeholder 25: Sidebar Model Information Area]**
> Description: Model information card at sidebar bottom

---

## 8. FAQ

### Q1: How accurate is the system's prediction?

**A**: The model has an AUC of approximately 0.76, meaning:
- Correctly distinguishes default/non-default for 76% of customers
- Still has 24% uncertainty
- **Recommendation**: Use predictions as reference, combine with human judgment

### Q2: Why are external credit scores most important?

**A**: EXT_SOURCE are comprehensive scores from external credit bureaus (like central bank credit centers), integrating:
- Historical repayment records
- Debt status
- Credit history length
- Inquiry count

These are the most effective information for predicting default.

### Q3: How can I improve my approval chances?

Based on feature importance, recommendations:
1. **Improve credit score**: Make payments on time, reduce debt
2. **Provide income proof**: Submit complete income documentation
3. **Lower monthly payment**: Extend loan term
4. **Provide asset proof**: Property, vehicle, etc.

### Q4: How many records can batch prediction handle?

**A**: Recommend not exceeding 10,000 records, as more may:
- Take longer to process
- Use more memory
- Cause browser timeout

### Q5: Where is prediction history saved?

**A**: Saved in `predictions_history/history.json` file in the application directory.

### Q6: How to backup data?

**A**:
1. Regularly export prediction history (copy history.json)
2. Download batch prediction results (CSV format)
3. Backup entire application folder

### Q7: Does the system require internet connection?

**A**:
- Local run: No internet required
- Streamlit Cloud deployment: Internet required for access

### Q8: Why are some fields negative numbers?

**A**: Like DAYS_BIRTH (birth days) using negative numbers:
- DAYS_BIRTH = -10000 → Approximately 27 years old
- This is the dataset's encoding method, system handles conversion automatically

---

## Appendix

### A. Field Description Table

| Field Name | English Name | Data Type | Description |
|------------|--------------|-----------|-------------|
| SK_ID_CURR | Customer ID | Numeric | Unique identifier |
| CODE_GENDER | Gender | Categorical | M=Male, F=Female |
| DAYS_BIRTH | Age | Numeric | Negative days |
| NAME_EDUCATION_TYPE | Education Level | Categorical | Education level |
| NAME_FAMILY_STATUS | Marital Status | Categorical | Marital status |
| CNT_CHILDREN | Children Count | Numeric | Number of children |
| CNT_FAM_MEMBERS | Family Members | Numeric | Total family size |
| AMT_INCOME_TOTAL | Annual Income | Numeric | Annual total income |
| AMT_CREDIT | Loan Amount | Numeric | Loan application amount |
| AMT_ANNUITY | Monthly Payment | Numeric | Monthly payment amount |
| AMT_GOODS_PRICE | Goods Price | Numeric | Purchase price |
| NAME_INCOME_TYPE | Income Type | Categorical | Income source |
| OCCUPATION_TYPE | Occupation Type | Categorical | Occupation category |
| DAYS_EMPLOYED | Employment Duration | Numeric | Negative days |
| FLAG_OWN_CAR | Owns Car | Categorical | Y/N |
| FLAG_OWN_REALTY | Owns Realty | Categorical | Y/N |
| OWN_CAR_AGE | Car Age | Numeric | Vehicle usage years |
| NAME_HOUSING_TYPE | Housing Type | Categorical | Living situation |
| EXT_SOURCE_1 | External Score 1 | Numeric | 0-1 range |
| EXT_SOURCE_2 | External Score 2 | Numeric | 0-1 range |
| EXT_SOURCE_3 | External Score 3 | Numeric | 0-1 range |

### B. Risk Level Details

| Level | Probability Range | Decision | Recommended Action |
|-------|-------------------|----------|-------------------|
| Low Risk | < 8% | Approve | Normal process |
| Medium Risk | 8% - 24% | Review | Consider additional collateral or reduce loan amount |
| High Risk | > 24% | Reject | Loan not recommended |

*Note: Threshold can be adjusted in settings*

### C. Technical Support

If you encounter issues, please:
1. Check FAQ section of this guide
2. Verify input data format is correct
3. Check browser console for error messages
4. Contact technical support team

---

**Document Version**: v1.0
**Last Updated**: 2026-04-05
**Applicable System Version**: Credit Risk Assessment System v1.0
