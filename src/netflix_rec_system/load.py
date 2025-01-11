import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("data/netflixData.csv")
print(data.head())



print(data.isnull().sum())
# lets clear the null values for better analysis.
# The dataset contains null values, but before removing the null values, 
# let’s select the columns that we can use to build a Netflix recommendation system:
data = data[["Title", "Description", "Content Type", "Genres"]]
print(data.head())


data = data.dropna()
# let’s drop the rows containing null values and move further: