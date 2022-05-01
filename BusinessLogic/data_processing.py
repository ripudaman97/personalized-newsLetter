
from sklearn import preprocessing
import pandas as pd


def preprocess_credit_data():

    # loading data
    credit_data = pd.read_csv ("../raw_data.csv")

    # normalizing and scaling data

    normalized_credit_data = preprocessing.MinMaxScaler()
    col_names = credit_data.columns
    d = normalized_credit_data.fit_transform(credit_data)

    scaled_credit_data = pd.DataFrame(d,columns=col_names)

    # print(scaled_credit_data)

    return scaled_credit_data
