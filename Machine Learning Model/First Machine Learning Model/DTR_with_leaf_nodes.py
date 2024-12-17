from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd


# training model
training_model = pd.read_csv("train.csv")

y = training_model.SalePrice

feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = training_model[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# finding correct max leaf nodes
def get_mlf(train_X, train_y, val_X, val_y):
  candidates = []
  list_mae = []
  for i in range(5, 505, 5):
    candidates.append(i)
    
    dtr_ln_model = DecisionTreeRegressor(max_leaf_nodes=i, random_state=1)
    dtr_ln_model.fit(train_X, train_y)

    predictions = dtr_ln_model.predict(val_X)
    mae = mean_absolute_error(predictions, val_y)
    list_mae.append(mae)

  for i in range(len(candidates)):
    if list_mae[i] == min(list_mae):
      return candidates[i]

  
mlf = get_mlf(train_X, train_y, val_X, val_y)

dtr_ln_model = DecisionTreeRegressor(max_leaf_nodes=mlf, random_state=1)
dtr_ln_model.fit(X, y)
  



  