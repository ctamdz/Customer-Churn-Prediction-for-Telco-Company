import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_models():
    """Load trained model, scaler and feature columns"""
    try:
        with open('logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('minmax_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        print(" Models loaded successfully!")
        return model, scaler, feature_columns
    except FileNotFoundError:
        print(" Model files not found! Please train model first.")
        exit()

def preprocess_data(df, scaler, feature_columns):
    """Preprocess data same as training pipeline"""
    processed_df = df.copy()
    
    # Label encoding
    label_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in label_cols:
        processed_df[col] = processed_df[col].map({'Yes': 1, 'No': 0})
    
    processed_df["gender"] = processed_df["gender"].map({"Female": 1, "Male": 0})
    
    # One-hot encoding
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaymentMethod']
    processed_df = pd.get_dummies(processed_df, columns=ohe_cols)
    
    # Scale numerical features
    numerical_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
    numerical_data = processed_df[numerical_cols]
    other_data = processed_df.drop(columns=numerical_cols)
    
    scaled_features = scaler.transform(numerical_data)
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols, index=other_data.index)
    processed_df = pd.concat([scaled_df, other_data], axis=1)
    
    # Drop customerID for prediction
    processed_df = processed_df.drop(columns=['customerID'])
    
    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    # Reorder columns to match training data
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)
    
    print(" Data preprocessing completed!")
    return processed_df

def predict_churn(model, data):
    """Make churn predictions"""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]
    print(" Predictions completed!")
    return predictions, probabilities

def main():
    # Load models
    model, scaler, feature_columns = load_models()
    
    # Load data
    try:
        df = pd.read_csv('sample.csv')
        print(f" Data loaded: {df.shape[0]} customers")
    except FileNotFoundError:
        print(" sample.csv not found!")
        return
    
    # Preprocess data
    processed_data = preprocess_data(df, scaler, feature_columns)
    
    # Make predictions
    predictions, probabilities = predict_churn(model, processed_data)
    
    # Add results to original dataframe
    df['Churn_Prediction'] = predictions
    df['Churn_Probability'] = probabilities.round(4)
    df['Risk_Level'] = ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' 
                        for p in probabilities]
    
    # Display results
    print("\n=== KẾT QUẢ DỰ ĐOÁN ===")
    result_cols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges', 
                   'Churn_Prediction', 'Churn_Probability', 'Risk_Level']
    print(df[result_cols].to_string(index=False))
    
    # Summary statistics
    total = len(df)
    churn_count = sum(predictions)
    churn_rate = churn_count / total * 100
    
    print(f"\n=== THỐNG KÊ ===")
    print(f"Tổng khách hàng: {total}")
    print(f"Dự đoán churn: {churn_count} ({churn_rate:.1f}%)")
    
    risk_counts = df['Risk_Level'].value_counts()
    print(f"\nPhân bố rủi ro:")
    for risk, count in risk_counts.items():
        print(f"- {risk}: {count} ({count/total*100:.1f}%)")
    
    # Save results
    df.to_csv('sample_with_predictions.csv', index=False)
    print(f"\n Kết quả đã lưu vào 'sample_with_predictions.csv'")

if __name__ == "__main__":
    main()
