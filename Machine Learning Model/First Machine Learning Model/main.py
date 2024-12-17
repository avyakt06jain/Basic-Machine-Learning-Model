from DTR_without_leaf_nodes import dtr_wln_model
from DTR_with_leaf_nodes import dtr_ln_model
from RFR import rf_model
import pandas as pd

# testing model
testing_model = pd.read_csv("test.csv")

feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
test_X = testing_model[feature_names]

# prediction with dtr without leaf node model
dtr_wln_model_predictions = dtr_wln_model.predict(test_X)
output_dtr_wln_model = pd.DataFrame({"Id": testing_model.Id,
                                     "SalePrice": dtr_wln_model_predictions})
output_dtr_wln_model.to_csv("Submission of DTRWLN.csv", index=False)

# prediction with dtr with leaf node model
dtr_ln_model_predictions = dtr_ln_model.predict(test_X)
output_dtr_ln_model = pd.DataFrame({"Id": testing_model.Id,
                                   "SalePrice": dtr_ln_model_predictions})
output_dtr_ln_model.to_csv("Submission of DTRLN.csv", index=False)

# prediction with rf model
rf_model_predictions = rf_model.predict(test_X)
output_rf_model = pd.DataFrame({"Id": testing_model.Id,
                                "SalePrice": rf_model_predictions})
output_rf_model.to_csv("Submission of RF.csv", index=False)






