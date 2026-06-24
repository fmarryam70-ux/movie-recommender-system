# ==========================================
# 📌 RECOMMENDER SYSTEMS
# 📌 MovieLens Latest Small Dataset
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==========================================
# 📊 DATA LOADING + EXPLORATION
# ==========================================

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

print("\n🔹 First 5 rows:")
print(ratings.head())

print("\n🔹 Dataset Shape:", ratings.shape)

print("\n🔹 Unique Users:", ratings['userId'].nunique())
print("🔹 Unique Movies:", ratings['movieId'].nunique())

# --------------------------
# 📊 GRAPH 1: Ratings Distribution
# --------------------------
plt.figure()
plt.hist(ratings['rating'], bins=10)
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# --------------------------
# 📊 GRAPH 2: Top Rated Movies
# --------------------------
top_movies = ratings['movieId'].value_counts().head(10)

plt.figure()
top_movies.plot(kind='bar')
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Movie ID")
plt.ylabel("Number of Ratings")
plt.show()

# --------------------------
# 📊 GRAPH 3: User Activity
# --------------------------
plt.figure()
ratings['userId'].value_counts().hist()
plt.title("Ratings per User Distribution")
plt.xlabel("Number of Ratings")
plt.ylabel("Users Count")
plt.show()

# --------------------------
# 📊 USER-ITEM MATRIX
# --------------------------
user_item = ratings.pivot_table(index='userId',
                                 columns='movieId',
                                 values='rating')

print("\n🔹 User-Item Matrix Sample:")
print(user_item.head())

# --------------------------
# 📊 Sparsity Calculation
# --------------------------
total = user_item.shape[0] * user_item.shape[1]
missing = user_item.isna().sum().sum()
sparsity = 1 - ((total - missing) / total)

print("\n🔹 Sparsity:", sparsity)

# ==========================================
# 🤝 USER-BASED COLLABORATIVE FILTERING
# ==========================================

user_item_filled = user_item.fillna(0)

user_similarity = cosine_similarity(user_item_filled)

user_sim_df = pd.DataFrame(user_similarity,
                            index=user_item.index,
                            columns=user_item.index)

target_user = 1

print("\n🔹 Top Similar Users:")
print(user_sim_df[target_user].sort_values(ascending=False)[1:6])

# --------------------------
# Prediction Function
# --------------------------
def predict_user_cf(user, movie):
    sim_scores = user_sim_df[user]

    num = 0
    den = 0

    for u in user_item.index:
        if not np.isnan(user_item.loc[u, movie]):
            num += sim_scores[u] * user_item.loc[u, movie]
            den += abs(sim_scores[u])

    if den == 0:
        return 0
    return num / den

# Recommendations
user_cf_scores = {}

for movie in user_item.columns:
    if np.isnan(user_item.loc[target_user, movie]):
        user_cf_scores[movie] = predict_user_cf(target_user, movie)

top_user_cf = sorted(user_cf_scores.items(),
                     key=lambda x: x[1],
                     reverse=True)[:5]

print("\n🔹 User-Based CF Recommendations:")
print(top_user_cf)

# ==========================================
# 🎬  ITEM-BASED COLLABORATIVE FILTERING
# ==========================================

item_user = user_item.T.fillna(0)

item_similarity = cosine_similarity(item_user)

item_sim_df = pd.DataFrame(item_similarity,
                           index=item_user.index,
                           columns=item_user.index)

def predict_item_cf(user, movie):
    sim_scores = item_sim_df[movie]

    num = 0
    den = 0

    for m in user_item.columns:
        if not np.isnan(user_item.loc[user, m]):
            num += sim_scores[m] * user_item.loc[user, m]
            den += abs(sim_scores[m])

    if den == 0:
        return 0
    return num / den

item_cf_scores = {}

for movie in user_item.columns:
    if np.isnan(user_item.loc[target_user, movie]):
        item_cf_scores[movie] = predict_item_cf(target_user, movie)

top_item_cf = sorted(item_cf_scores.items(),
                     key=lambda x: x[1],
                     reverse=True)[:5]

print("\n🔹 Item-Based CF Recommendations:")
print(top_item_cf)

# ==========================================
# 🎥 CONTENT-BASED FILTERING
# ==========================================

movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(movies['genres'])

content_sim = cosine_similarity(genre_matrix)

content_sim_df = pd.DataFrame(content_sim,
                              index=movies['movieId'],
                              columns=movies['movieId'])

# --------------------------
# Movie Recommendation Function
# --------------------------
def recommend_movie(title):
    movie_id = movies[movies['title'] == title]['movieId'].values[0]

    scores = content_sim_df[movie_id].sort_values(ascending=False)[1:11]

    return movies[movies['movieId'].isin(scores.index)]['title']

print("\n🔹 Content-Based Recommendations:")
print(recommend_movie("Toy Story (1995)"))

# ==========================================
# 📈  EVALUATION
# ==========================================

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# --------------------------
# Baseline Predictor
# --------------------------
def baseline_pred(row):
    movie = row['movieId']

    if movie in user_item.columns:
        return user_item[movie].mean()
    return 0

test['pred'] = test.apply(baseline_pred, axis=1)

rmse = np.sqrt(mean_squared_error(test['rating'], test['pred']))

print("\n🔹 RMSE:", rmse)

# --------------------------
# Precision@K
# --------------------------
def precision_at_k(recommended, relevant, k=5):
    recommended = recommended[:k]
    return len(set(recommended) & set(relevant)) / k

relevant = ratings[(ratings['userId'] == target_user) &
                   (ratings['rating'] >= 4)]['movieId'].tolist()

recommended = [m[0] for m in top_user_cf]

print("\n🔹 Precision@5:", precision_at_k(recommended, relevant, 5))

