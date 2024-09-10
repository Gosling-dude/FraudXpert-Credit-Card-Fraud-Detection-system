import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Function to preprocess data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Scaling transaction amount
    scaler = StandardScaler()
    df['TransactionAmount'] = scaler.fit_transform(df[['TransactionAmount']])
    
    # Feature selection
    features = df[['TransactionAmount', 'TransactionTime', 'CustomerID']]
    labels = df['Fraud']
    
    # Handling class imbalance with undersampling
    under_sampler = RandomUnderSampler(sampling_strategy=1.0)
    features_res, labels_res = under_sampler.fit_resample(features, labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features_res, labels_res, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
