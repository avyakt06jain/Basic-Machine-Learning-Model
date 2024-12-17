from sklearn.tree import DecisionTreeRegressor 
import pandas as pd


# training model
training_model = pd.read_csv("train.csv")

train_y = training_model.SalePrice

feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
train_X = training_model[feature_names]


dtr_wln_model = DecisionTreeRegressor(random_state=0)
dtr_wln_model.fit(train_X, train_y)
