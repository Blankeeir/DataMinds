# DataMinds
## Prediction Model for sales figures using Using Random Forest Regressor for ML Model and MSE & R-Square for model testing

## Preparations:
```ruby
pip3 install pandas
pip3 install matplotlib
pip3 install numpy
pip3 install scikit-learn
```

## Training and testing MLmodel 
```
python ML.py
```
You will find a mlmodel.h5 created in your directory

#Implement testing procedure
```ruby
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
```

## This test should output a list of predictions.
Make sure your dataset matches the format of catA_train.csv

```ruby

test_df = pd.read_csv(filepath)
test_df = test_df.drop(columns=['Sales (Domestic Ultimate Total USD)'])
print(testing_hidden_data(test_df))

```
### Do note that upon receiving requirements to pull down the original dataset for confidential purposes, the /data directory no longer exists. If you need original dataset format to test our model, please contact us at e1300538@u.nus.edu 

### All Use of data is subject to approval of organizors



