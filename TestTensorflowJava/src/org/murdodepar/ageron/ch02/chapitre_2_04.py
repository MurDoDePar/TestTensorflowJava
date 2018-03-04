import os
import pandas as pd

HOUSING_PATH = os.path.join(".\datasets", "housing")
print(HOUSING_PATH)

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
print("------------------housing.head")
#housing.head()
print(housing.head())
print("------------------housing.info")
housing.info()
print("------------------ocean_proximity")
#housing["ocean_proximity"].value_counts()
print(housing["ocean_proximity"].value_counts())
print("------------------housing.describe")
#housing.describe()
print(housing.describe)
