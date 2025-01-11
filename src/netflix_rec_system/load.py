import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from netflix_rec_system.pre_processing import clean


def load_data(path):
    data = pd.read_csv(path)
    print(data.head())


    print(data.isnull().sum())
    # lets clear the null values for better analysis.
    # The dataset contains null values, but before removing the null values,
    # let’s select the columns that we can use to build a Netflix recommendation system:
    data = data[["Title", "Description", "Content Type", "Genres"]]
    print(data.head())


    data = data.dropna()


    data["Title"] = data["Title"].apply(clean)
    print(data.head())
    return data

