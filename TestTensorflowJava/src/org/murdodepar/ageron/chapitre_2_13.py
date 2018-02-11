import hashlib
import numpy as np
import sklearn
import scipy
import chapitre_2_04 as ch

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

# This version supports both Python 2 and Python 3, instead of just Python 3.
def test_set_check(identifier, test_ratio, hash):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

housing_with_id = ch.housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = ch.housing["longitude"] * 1000 + ch.housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

test_set.head()

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(ch.housing, test_size=0.2, random_state=42)

test_set.head()

ch.housing["median_income"].hist()
print(ch.housing["median_income"].hist())

# Divide by 1.5 to limit the number of income categories
ch.housing["income_cat"] = np.ceil(ch.housing["median_income"] / 1.5)
# Label those above 5 as 5
ch.housing["income_cat"].where(ch.housing["income_cat"] < 5, 5.0, inplace=True)

ch.housing["income_cat"].value_counts()

ch.housing["income_cat"].hist()