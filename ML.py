# Import necessary libraries
## We use sklearn library for RandomForestRegressor implementation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
filepath = 'data\catA_train.csv'
dataset = pd.read_csv(filepath)
# Modification of Dataset & EDA
columns_to_drop = ["error", "Fiscal Year End", "Sales (Global Ultimate Total USD)", "Web Address","Square Footage", "Company Description", "PostCode",
                   "8-Digit SIC Code", "8-Digit SIC Description", "AccountID",
                   "Parent Company", "City", "Country", "Address", "Address1", "Industry", "Region", "Parent Country", "Global Ultimate Country", "Company"]

dataset = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], errors='ignore')

# Extract features and target variable
X = dataset.drop(['Sales (Domestic Ultimate Total USD)'], axis=1)  # Features
y = dataset['Sales (Domestic Ultimate Total USD)']  # Target variable


### ML: Random Forest Regressor
# Split the dataset into training and testing sets
# Using 20% data for testing and 80% data for training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing (standardization) and the Random Forest Regressor
MLmodel = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('regressor', RandomForestRegressor())  # Random Forest Regressor
])

# Train the model
MLmodel.fit(X_train, y_train)

# Make predictions on the test set
y_pred = MLmodel.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')





def testing_hidden_data(hidden_data: pd.DataFrame) -> list:
    result = [] 
    
    file_path = 'data\catA_train.csv'
    dataset = pd.read_csv(file_path)
    X = dataset.drop(['Sales (Domestic Ultimate Total USD)'], axis=1)  # Features
    y = dataset['Sales (Domestic Ultimate Total USD)']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('regressor', RandomForestRegressor())  # Random Forest Regressor
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared Score: {r2}')
    
    return result

##################
#Test_Hidden_Data#
##################

# This cell should output a list of predictions.
test_df = pd.read_csv(filepath)
test_df = test_df.drop(columns=['Sales (Domestic Ultimate Total USD)'])
print(testing_hidden_data(test_df))