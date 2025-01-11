from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from netflix_rec_system.load import load_data

path = "data/netflixData.csv"
data = load_data(path)
feature = data["Genres"].tolist()

# Create an instance of TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Fit and transform the vectorizer on our corpus
tfidf_matrix = tfidf.fit_transform(feature)

# Compute the cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix)