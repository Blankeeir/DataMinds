# Import necessary libraries
## We use sklearn library for RandomForestRegressor implementation
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Save the base model to an HDF5 file

# Load the dataset
filepath = 'data/catA_train.csv'
dataset = pd.read_csv(filepath)

# Modification of Dataset & EDA
columns_to_drop = ["AccountID","Company","Industry","8-Digit SIC Code","8-Digit SIC Description","Entity Type","Parent Company","Parent Country","Ownership Type","Company Description","Sales (Global Ultimate Total USD)","Fiscal Year End","Global Ultimate Company","Global Ultimate Country","Domestic Ultimate Company"]

dataset = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], errors='ignore')
dataset = dataset[dataset["Company Status (Active/Inactive)"] == "Active"]
dataset["Import/Export Status"] = dataset["Import/Export Status"].replace({'': '0','Imports':1, 'Exports':2,'Both Imports & Exports': 3})
dataset = dataset.drop(["Company Status (Active/Inactive)"], axis=1)


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
result = list(y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("predicted domestic sales figures: ",result)
print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

joblib.dump(MLmodel, 'mlmodel.h5')


def testing_hidden_data(hidden_data: pd.DataFrame) -> list:
    dataset = hidden_data
    '''# Modification of Dataset & EDA
    columns_to_drop = ["AccountID","Company","Industry","8-Digit SIC Code","8-Digit SIC Description","Entity Type","Parent Company","Parent Country","Ownership Type","Company Description","Sales (Global Ultimate Total USD)","Fiscal Year End","Global Ultimate Company","Global Ultimate Country","Domestic Ultimate Company"]

    dataset = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], errors='ignore')
    dataset = dataset[dataset["Company Status (Active/Inactive)"] == "Active"]
    dataset["Import/Export Status"] = dataset["Import/Export Status"].replace({'': '0','Imports':1, 'Exports':2,'Both Imports & Exports': 3})
    dataset = dataset.drop(["Company Status (Active/Inactive)"], axis=1)


    # Extract features and target variable
    X = dataset  # Features
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
    result = list(y_pred)'''

    
    loaded_model = joblib.load('./mlmodel.h5')
    result = list(loaded_model.predict(dataset))
    return result


##################
#Test_Hidden_Data#
##################

# This cell should output a list of predictions.
test_df = pd.read_csv(filepath)
test_df = test_df.drop(columns=['Sales (Domestic Ultimate Total USD)'])
#print(testing_hidden_data(test_df))