#  Customer Churn Prediction for Telco Company

##  Project Overview

### Purpose & Outcome
**Purpose**: To identify the factors that contribute to customer churn and develop a predictive model to identify at-risk customers in the telecommunications industry.

**Outcome**: A comprehensive machine learning system that can predict customer churn with high accuracy, along with actionable insights to reduce churn rates and improve customer retention strategies.

### Business Value
- **Proactive Customer Retention**: Identify at-risk customers before they churn
- **Data-Driven Decisions**: Make informed business decisions based on predictive analytics
- **Revenue Protection**: Reduce revenue loss by preventing customer churn
- **Resource Optimization**: Allocate retention resources more efficiently

##  Project Structure

```
Final_Project/
├── README.md                           # Comprehensive project documentation
├── Telco-Customer-Churn-Dataset.csv    # Original dataset from Kaggle
├── telco_clean.csv                     # Preprocessed and cleaned dataset
├── telco_preprocessing.ipynb           # Data preprocessing and EDA
├── telco_modeling.ipynb                # Model development and comparison
├── save_model.py                       # Model training and serialization
├── predict_churn.py                    # Production prediction script
├── sample.csv                          # Sample data for testing
├── sample_with_predictions.csv         # Prediction results
├── logistic_regression_model.pkl       # Trained Logistic Regression model
├── minmax_scaler.pkl                   # Fitted MinMaxScaler
├── feature_columns.pkl                 # Feature column names
├── Telco_Churn_Analysis.pbix           # Power BI interactive dashboard
└── Telco_Churn_Analysis.pdf            # Power BI static report
```

##  Data Visualization & Business Intelligence

### Power BI Dashboard

**Dashboard Files**: 
- **Power BI File**: `Telco_Churn_Analysis.pbix` (Interactive dashboard)
- **Dashboard Report**: `Telco_Churn_Analysis.pdf` (Static report for sharing)
- **Data Source**: `telco_clean.csv` and `sample_with_predictions.csv`

#### Dashboard Overview
Our comprehensive Power BI dashboard provides interactive business intelligence insights for customer churn analysis, enabling stakeholders to explore data visually and make data-driven decisions.

#### Key Dashboard Pages

**1. Executive Summary**
- High-level churn metrics and trends
- Key performance indicators (KPIs)
- Executive-level insights and recommendations

**2. Customer Demographics**
- Age and gender distribution analysis
- Senior citizen churn patterns
- Partner and dependents impact on retention

**3. Service Analysis**
- Internet service adoption rates
- Phone service penetration
- Streaming services usage patterns
- Service bundle effectiveness

**4. Contract & Payment Insights**
- Contract type distribution and churn rates
- Payment method preferences and risk levels
- Paperless billing adoption trends

**5. Revenue & Tenure Analysis**
- Monthly charges distribution
- Total charges patterns
- Customer tenure analysis
- Revenue retention by customer segment

**6. Churn Prediction Results**
- Model performance metrics
- Risk level distribution
- Prediction accuracy tracking
- Business impact measurement

#### Interactive Features
- **Real-time Filtering**: Filter by any customer attribute
- **Drill-down Capabilities**: Explore data at multiple levels
- **Cross-filtering**: Select one element to filter related charts
- **Export Functionality**: Download reports and visualizations
- **Mobile Responsive**: Access dashboard on any device

#### Dashboard Maintenance
- **File Management**: Power BI (.pbix) file for interactive analysis
- **Report Export**: PDF format for easy sharing and presentation
- **Data Integration**: Direct connection to cleaned dataset and prediction results
- **Version Control**: Track dashboard changes and improvements


##  Dataset Information

### Source
**Kaggle Dataset**: [Telecom Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Provider**: IBM Sample Data Sets
- **Size**: 7,043 customers
- **Features**: 20 customer attributes
- **Target Variable**: Churn (Yes/No)

### Data Structure

| Feature | Type | Description | Values |
|---------|------|-------------|---------|
| `customerID` | String | Unique customer identifier | - |
| `gender` | Categorical | Customer gender | Male, Female |
| `SeniorCitizen` | Binary | Senior citizen status | 0, 1 |
| `Partner` | Binary | Has partner | Yes, No |
| `Dependents` | Binary | Has dependents | Yes, No |
| `tenure` | Numerical | Length of customer tenure (months) | 0-72 |
| `PhoneService` | Binary | Has phone service | Yes, No |
| `MultipleLines` | Categorical | Multiple lines service | Yes, No, No phone service |
| `InternetService` | Categorical | Internet service type | DSL, Fiber optic, No |
| `OnlineSecurity` | Categorical | Online security service | Yes, No, No internet service |
| `OnlineBackup` | Categorical | Online backup service | Yes, No, No internet service |
| `DeviceProtection` | Categorical | Device protection service | Yes, No, No internet service |
| `TechSupport` | Categorical | Tech support service | Yes, No, No internet service |
| `StreamingTV` | Categorical | Streaming TV service | Yes, No, No internet service |
| `StreamingMovies` | Categorical | Streaming movies service | Yes, No, No internet service |
| `Contract` | Categorical | Contract type | Month-to-month, One year, Two year |
| `PaperlessBilling` | Binary | Paperless billing | Yes, No |
| `PaymentMethod` | Categorical | Payment method | Electronic check, Mailed check, Bank transfer, Credit card |
| `MonthlyCharges` | Numerical | Monthly charges ($) | 18.25-118.75 |
| `TotalCharges` | Numerical | Total charges ($) | 18.8-8684.8 |
| `Churn` | Binary | Customer churn status | Yes, No |

##  Analysis Workflow

### Phase 1: Data Preprocessing & EDA

#### Data Preprocessing (`telco_preprocessing.ipynb`)
- **Data Cleaning**: Handle missing values, outliers, and data inconsistencies
- **Data Quality Assessment**: Check data types, distributions, and correlations
- **Feature Engineering**: Create derived features and handle categorical variables

#### Exploratory Data Analysis (EDA)
**Tools Used**: 
- **Power BI** - Interactive dashboards and business intelligence insights
- **Python (Matplotlib/Seaborn)** - Statistical analysis and technical visualizations

**Power BI Dashboard Components**:
- **Customer Demographics Analysis**: Age distribution, gender breakdown, senior citizen analysis
- **Service Usage Patterns**: Internet service adoption, phone service penetration, streaming services
- **Contract & Payment Analysis**: Contract type distribution, payment method preferences
- **Churn Rate Analysis**: Churn rates by different customer segments
- **Revenue Analysis**: Monthly charges distribution, total charges patterns
- **Tenure Analysis**: Customer loyalty patterns and retention insights

**Key Visualizations Created**:
- Interactive charts showing churn rates by contract type
- Bar charts for service adoption across different segments
- Heat maps for correlation analysis
- Time series analysis for customer tenure patterns
- Geographic distribution (if applicable)
- Real-time filtering and drill-down capabilities

#### Key Insights from Power BI EDA:

** Overview Dashboard Findings:**
- **Overall Churn Rate**: 26.54% (1,869 out of 7,043 customers)
- **Revenue Impact**: $139.13K monthly revenue lost (30.50% of total revenue)
- **Demographics**: Balanced gender distribution (50.48% Female, 49.52% Male)
- **Senior Citizens**: Higher churn risk among non-senior customers (1,393 vs 476 churns)

** Account Details Analysis:**

**Contract Type Impact:**
- **Month-to-month**: 42.71% churn rate (highest risk)
- **One year**: 11.27% churn rate
- **Two year**: 2.83% churn rate (lowest risk)

**Tenure Analysis:**
- **<1 year**: 48.28% churn rate (highest risk)
- **<2 years**: 29.51% churn rate
- **<3 years**: 22.03% churn rate
- **<4 years**: 19.52% churn rate
- **<5 years**: 15.00% churn rate
- **<6 years**: 8.30% churn rate
- **≥6 years**: 1.66% churn rate (lowest risk)

** Payment & Billing Analysis:**
- **Paperless Billing**: 33.57% churn rate vs 16.33% (traditional billing)
- **Electronic Check**: 45.29% churn rate (highest risk payment method)
- **Credit Card (automatic)**: 15.24% churn rate (lowest risk)
- **Bank Transfer (automatic)**: 16.71% churn rate
- **Mailed Check**: 19.11% churn rate

** Revenue Patterns:**
- **Active Customers**: Average $2,549.91 total charges
- **Churned Customers**: Average $1,531.80 total charges
- **Monthly Charges**: Churned customers pay higher monthly fees ($74.44 vs $61.27)

** Service Usage Analysis:**

**Main Services:**
- **Fiber Optic**: 41.89% churn rate (highest risk)
- **DSL**: 18.96% churn rate
- **No Internet**: 7.40% churn rate (lowest risk)
- **Phone Service**: 26.71% churn rate vs 24.93% (no phone)
- **Multiple Lines**: 28.61% churn rate (highest among phone services)

**Additional Services:**
- **Online Security**: 41.77% churn rate (without service) vs 14.61% (with service)
- **Tech Support**: 41.64% churn rate (without service) vs 15.17% (with service)
- **Online Backup**: 39.93% churn rate (without service) vs 21.53% (with service)
- **Device Protection**: 39.13% churn rate (without service) vs 22.50% (with service)

** Streaming Services:**
- **Streaming Movies**: 33.68% churn rate (without service) vs 29.94% (with service)
- **Streaming TV**: 33.52% churn rate (without service) vs 30.07% (with service)

### Phase 2: Model Development (`telco_modeling.ipynb`)

#### Feature Engineering Pipeline
1. **Label Encoding**: Convert binary categorical variables
   - `Partner`, `Dependents`, `PhoneService`, `Churn`, `PaperlessBilling` → 0/1
   - `gender` → Female=1, Male=0

2. **One-Hot Encoding**: Convert multi-category variables
   - `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`
   - `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
   - `Contract`, `PaymentMethod`

3. **Feature Scaling**: MinMaxScaler for numerical features
   - `MonthlyCharges`, `TotalCharges`, `tenure` → [0,1] range

#### Model Comparison Framework
We implemented and compared three state-of-the-art algorithms:

1. **Logistic Regression**
   - **Advantages**: Interpretable, fast, good baseline
   - **Hyperparameters**: C=100, penalty='l2', solver='liblinear'
   - **Performance**: Accuracy=80.34%, AUC=84.11%

2. **Random Forest**
   - **Advantages**: Robust, handles non-linear relationships
   - **Hyperparameters**: n_estimators=200, max_depth=10, min_samples_split=5
   - **Performance**: Accuracy=80.41%, AUC=84.18%

3. **XGBoost**
   - **Advantages**: High performance, handles missing values
   - **Hyperparameters**: n_estimators=300, max_depth=3, learning_rate=0.01
   - **Performance**: Accuracy=79.99%, AUC=84.58%

### Phase 3: Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- **Cross-Validation**: 5-fold cross-validation for hyperparameter tuning
- **Overfitting Analysis**: Train vs Test performance comparison
- **Feature Importance**: Coefficient analysis and feature ranking

### Phase 4: Production Deployment (`save_model.py`, `predict_churn.py`)
- **Model Serialization**: Save trained models and preprocessing objects
- **Prediction Pipeline**: End-to-end prediction system
- **Risk Classification**: Low/Medium/High risk categorization

##  Model Performance Analysis

### Comprehensive Model Comparison

| Model | Test Accuracy | Test AUC | Precision | Recall | F1-Score | Overfitting Gap |
|-------|---------------|----------|-----------|--------|----------|-----------------|
| **Logistic Regression** | **80.34%** | **84.11%** | **65.30%** | **55.35%** | **59.91%** | **0.28%** |
| Random Forest | 80.41% | 84.18% | 66.33% | 54.55% | 59.05% | 4.36% |
| XGBoost | 79.99% | 84.58% | 64.29% | 48.66% | 54.95% | 0.84% |

### Model Selection Rationale
** Recommended Model: Logistic Regression**
- **Overall Score**: 81.26% (weighted combination of all metrics)
- **Key Advantages**:
  - Minimal overfitting (0.28% gap)
  - High interpretability for business stakeholders
  - Consistent performance across metrics
  - Fast prediction speed for production use


##  Implementation Guide

### Prerequisites
```bash
# Required Python packages
pip install pandas scikit-learn numpy matplotlib seaborn xgboost
```

### Quick Start
```bash
# 1. Train the model
python save_model.py

# 2. Make predictions
python predict_churn.py
```



### Detailed Usage

#### Training New Model
```python
# The save_model.py script will:
# - Load and preprocess telco_clean.csv
# - Train Logistic Regression with optimal parameters
# - Save model, scaler, and feature columns
python save_model.py
```

#### Making Predictions
```python
# The predict_churn.py script will:
# - Load trained model and preprocessing objects
# - Process new customer data (sample.csv)
# - Generate churn predictions and risk levels
# - Save results to sample_with_predictions.csv
python predict_churn.py
```

##  Risk Classification System

### Risk Levels
- **Low Risk** (Green): Churn probability < 30%
  - Recommended action: Standard retention programs
- **Medium Risk** (Yellow): Churn probability 30-70%
  - Recommended action: Targeted retention campaigns
- **High Risk** (Red): Churn probability > 70%
  - Recommended action: Immediate intervention and personalized offers

### Output Format
```csv
customerID,tenure,Contract,MonthlyCharges,Churn_Prediction,Churn_Probability,Risk_Level
CUST0001,1,Month-to-month,29.85,1,0.8234,High
CUST0002,5,Month-to-month,56.95,1,0.6542,Medium
```

##  Business Recommendations

### Immediate Actions
1. **Target New Customers**: Implement onboarding programs for customers with <12 months tenure
2. **Contract Incentives**: Offer discounts for upgrading to longer-term contracts
3. **Payment Method Promotion**: Encourage automatic payment methods
4. **Service Bundling**: Package internet security and tech support with core services

### Strategic Initiatives
1. **Customer Success Programs**: Proactive support for high-risk segments
2. **Pricing Strategy Review**: Analyze pricing for fiber optic vs DSL services
3. **Retention Analytics**: Implement real-time churn prediction dashboard
4. **A/B Testing Framework**: Test retention strategies on different segments

### Long-term Strategy
1. **Predictive Analytics Integration**: Embed churn prediction into CRM systems
2. **Customer Journey Optimization**: Improve touchpoints based on churn drivers
3. **Product Development**: Develop features addressing identified pain points
4. **Competitive Analysis**: Monitor competitor offerings in high-churn segments

##  Technical Architecture

### Data Pipeline
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Model Evaluation → Production Deployment
```

### Model Pipeline
```
New Customer Data → Preprocessing → Feature Scaling → Prediction → Risk Classification → Business Actions
```

### Power BI Integration
```
Data Source → Power BI Desktop → Interactive Dashboard → PDF Export → Stakeholder Sharing
```

**Integration Points**:
- **Data Source**: Direct connection to cleaned dataset (`telco_clean.csv`)
- **Model Results**: Integration with prediction outputs (`sample_with_predictions.csv`)
- **File Export**: PDF format for easy distribution and presentation
- **Interactive Analysis**: .pbix file for detailed exploration and filtering

## Customization Options

### Adding New Features
1. Update `feature_columns.pkl` with new feature names
2. Modify preprocessing pipeline in `predict_churn.py`
3. Retrain model with `save_model.py`

### Adjusting Risk Thresholds
```python
# In predict_churn.py, modify risk classification:
df['Risk_Level'] = ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' 
                    for p in probabilities]
```

### Model Selection
```python
# Choose different models by modifying save_model.py:
# - Random Forest: RandomForestClassifier()
# - XGBoost: XGBClassifier()
# - Logistic Regression: LogisticRegression()
```

##  Performance Monitoring

### Key Metrics to Track
- **Model Accuracy**: Maintain >80% accuracy
- **Prediction Volume**: Monitor daily prediction counts
- **Risk Distribution**: Track Low/Medium/High risk ratios
- **Business Impact**: Measure retention rate improvements

### Model Retraining Schedule
- **Monthly**: Retrain with new data
- **Quarterly**: Full model evaluation and comparison
- **Annually**: Feature importance review and business strategy alignment


### Model Performance Visualizations

**Technical Charts from Python Analysis:**
- **ROC Curves Comparison**: Logistic Regression vs Random Forest vs XGBoost
- **Feature Importance Plots**: Top 15 features ranked by importance
- **Confusion Matrix**: Model prediction accuracy visualization
- **Model Comparison Charts**: Performance metrics across algorithms


