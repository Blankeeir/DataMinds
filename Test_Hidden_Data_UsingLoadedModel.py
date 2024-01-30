###This File is for test only 
###It's irrelevant to Datathon
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

filepath = ''

def testing_hidden_data(hidden_data: pd.DataFrame) -> list:
    dataset = hidden_data
    columns_to_drop = ["AccountID","Company","Industry","8-Digit SIC Code","8-Digit SIC Description","Entity Type","Parent Company","Parent Country","Ownership Type","Company Description","Sales (Global Ultimate Total USD)","Fiscal Year End","Global Ultimate Company","Global Ultimate Country","Domestic Ultimate Company"]
    dataset = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], errors='ignore')
    dataset = dataset[dataset["Company Status (Active/Inactive)"] == "Active"]
    dataset["Import/Export Status"] = dataset["Import/Export Status"].replace({'': '0','Imports':1, 'Exports':2,'Both Imports & Exports': 3})
    dataset = dataset.drop(["Company Status (Active/Inactive)"], axis=1)
    loaded_model = joblib.load('./mlmodel.h5')
    result = list(loaded_model.predict(dataset))
    return result


##################
#Test_Hidden_Data#
##################

#This cell should output a list of predictions.
test_df = pd.read_csv(filepath)
test_df = test_df.drop(columns=['Sales (Domestic Ultimate Total USD)'])
print(testing_hidden_data(test_df))