# 🎬 Movie Recommender System

A complete Movie Recommendation System built using the MovieLens dataset, implementing multiple recommendation techniques and evaluated with standard metrics.

## 🚀 What This Project Does

- Explores and analyzes the MovieLens dataset (100,836 ratings, 610 users, 9,724 movies)
- Builds a User-Item Rating Matrix and calculates sparsity
- Implements User-Based Collaborative Filtering using Cosine Similarity
- Implements Item-Based Collaborative Filtering
- Implements Content-Based Filtering using TF-IDF and Cosine Similarity
- Evaluates performance using RMSE and Precision@K metrics

## 🛠️ Technologies Used

- Python
- Pandas — Data manipulation
- NumPy — Numerical computations
- Scikit-learn — TF-IDF, Cosine Similarity, Train-Test Split, RMSE
- Matplotlib — Data visualizations

## 📊 Dataset

- **Source:** MovieLens Latest Small Dataset
- **Ratings:** 100,836
- **Users:** 610
- **Movies:** 9,724
- **Sparsity:** 98.3%

## 📁 Project Structure

| File | Description |
|------|-------------|
| recommender_system.py | Complete implementation |
| ratings_per_distribution.png | Ratings distribution graph |
| ratings per user distribution.png | Ratings per user graph |
| top_10 rated movies.png | Top 10 most rated movies graph |

## ⚙️ How It Works

```
MovieLens Dataset
      ↓
Data Exploration & Visualization
      ↓
User-Item Rating Matrix
      ↓
┌─────────────────────────────────┐
│  User-Based Collaborative       │
│  Item-Based Collaborative       │
│  Content-Based (TF-IDF)         │
└─────────────────────────────────┘
      ↓
Evaluation (RMSE + Precision@K)
```


## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
```
pip install pandas numpy scikit-learn matplotlib
```
3. Download MovieLens dataset (ratings.csv, movies.csv)
4. Run the script:
```
python recommender_system.py
```

## 👩‍💻 Author

**Maryam Fatima**  
Computer Engineering Student @ UET Taxila  
Python | Machine Learning | NLP | Web Scraping | Data Analysis
