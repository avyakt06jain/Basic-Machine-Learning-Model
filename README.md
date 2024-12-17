# Basic Machine Learning Model with DecisionTreeRegressor and RandomForestRegressor
This project demonstrates a simple implementation of a **Decision Tree Regressor** and a **Random Forest Regressor** using Python's `sklearn` library. The goal is to train these models to predict a target variable (`SalePrice`) based on several features of a dataset.

## Dataset
The project uses two datasets:
- **train.csv**: The training dataset.
- **test.csv**: The testing dataset.

The datasets contain the following columns:
- **LotArea**: Lot size in square feet.
- **YearBuilt**: The year the house was built.
- **1stFlrSF**: First-floor square footage.
- **2ndFlrSF**: Second-floor square footage.
- **FullBath**: Number of full bathrooms.
- **BedroomAbvGr**: Number of bedrooms above ground.
- **TotRmsAbvGrd**: Total number of rooms above ground.
- **SalePrice**: The price at which the house was sold (target variable).

## Models Implemented
### 1. Decision Tree Regressor
The **Decision Tree Regressor** is a simple, interpretable machine learning model that can be used for regression tasks. It works by recursively splitting the data into subsets based on feature values to minimize prediction error.

#### a. Decision Tree Regressor without `max_leaf_nodes`
This model builds a full decision tree without restricting the number of leaf nodes.

#### b. Decision Tree Regressor with `max_leaf_nodes`
The `max_leaf_nodes` parameter controls the maximum number of leaf nodes in the decision tree. Limiting the number of leaf nodes helps to prevent overfitting by restricting the model's complexity.

- **Benefits of using max_leaf_nodes**:
  - Reduces overfitting by simplifying the model.
  - Faster training and prediction times for larger datasets.

---
### 2. Random Forest Regressor
The **Random Forest Regressor** is an ensemble method that builds multiple decision trees and combines their outputs for improved prediction accuracy. Each tree in the forest is trained on a random subset of the data (bootstrapping) and considers a random subset of features at each split.

- **Key Advantages of Random Forest**:
  - **Reduces overfitting**: By averaging the predictions of multiple trees.
  - **Handles feature randomness**: Considers only a subset of features, making the model robust to irrelevant features.
  - **Improved accuracy**: Typically achieves better performance than a single decision tree.

---

## Output
The script will output:
1. The first five rows of the target variable (`SalePrice`) and features.
2. A summary description of the feature data.
3. Predictions for the first five rows of the dataset using the following models:
   - Decision Tree Regressor without `max_leaf_nodes`
   - Decision Tree Regressor with `max_leaf_nodes`
   - Random Forest Regressor

---

## Conclusion
This project demonstrates the usage of **DecisionTreeRegressor** (with and without `max_leaf_nodes`) and **RandomForestRegressor** to predict the `SalePrice` column of a housing dataset. By controlling parameters like `max_leaf_nodes` in the Decision Tree and using an ensemble approach with Random Forest, we can achieve better model performance and reduce overfitting.

---
