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
##Implement testing procedure
```ruby
def testing_hidden_data(hidden_data: pd.DataFrame) -> list:
    dataset = hidden_data
    loaded_model = joblib.load('./mlmodel.h5')
    result = list(loaded_model.predict(dataset))
    return result

# This cell should output a list of predictions.
test_df = pd.read_csv(filepath)
test_df = test_df.drop(columns=['Sales (Domestic Ultimate Total USD)'])
print(testing_hidden_data(test_df))
```



