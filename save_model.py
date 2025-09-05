import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("telco_clean.csv")

# Label Encoding
columes_to_label_encode = ['Partner', 'Dependents', 'PhoneService', 'Churn', 'PaperlessBilling']
def label_encoding(df, columns):
    for col in columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

df = label_encoding(df, columes_to_label_encode)
df["gender"] = df["gender"].map({"Female":1, "Male":0})

# One Hot Encoding
columes_to_one_hot_encode = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                             'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df_ohe = pd.get_dummies(df, columns=columes_to_one_hot_encode)

# Feature Scaling
numerical_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']
df_mms = pd.DataFrame(df_ohe, columns=numerical_columns)
df_remaining = df_ohe.drop(columns=numerical_columns)

mms = MinMaxScaler(feature_range=(0,1))
rescaled_feature = mms.fit_transform(df_mms)    
rescaled_feature_df = pd.DataFrame(rescaled_feature, columns=numerical_columns, index=df_remaining.index)
df = pd.concat([rescaled_feature_df,df_remaining],axis=1)

# Drop customerID
df = df.drop(columns=['customerID'])


# Train model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

X = df.drop(columns = "Churn")
y = df.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Sử dụng best parameters từ notebook
best_params = {'C': 100, 'penalty': 'l2', 'solver': 'liblinear'}
lr_best_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
lr_best_model.fit(X_train, y_train)

# Save the trained model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_best_model, f)

# Save the scaler
with open('minmax_scaler.pkl', 'wb') as f:
    pickle.dump(mms, f)

# Save feature columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

#  Test model loading
with open('logistic_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('minmax_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    loaded_features = pickle.load(f)

# Model performance summary
from sklearn.metrics import accuracy_score, roc_auc_score

train_accuracy = loaded_model.score(X_train, y_train)
test_accuracy = loaded_model.score(X_test, y_test)
train_auc = roc_auc_score(y_train, loaded_model.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, loaded_model.predict_proba(X_test)[:, 1])

# Summary
print("Tất cả files đã được lưu")
