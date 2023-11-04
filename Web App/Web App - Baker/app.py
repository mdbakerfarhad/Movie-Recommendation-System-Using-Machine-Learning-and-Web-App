import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load movie data and user ratings
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocess the data and create a user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Item-based collaborative filtering model
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

@app.route('/')
def index():
    return render_template('index.html', movie_names=movies['title'])

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]

    # Get movie recommendations based on item similarity
    recommendations = get_movie_recommendations(movie_id)

    return render_template('recommendations.html',movie_name=movie_name, recommendations=recommendations)

def get_movie_recommendations(movie_id, num_recommendations=10):
    # Find movies most similar to the selected movie
    similar_movies = list(item_similarity_df[movie_id].sort_values(ascending=False).index[1:num_recommendations+1])
    recommended_movies = movies[movies['movieId'].isin(similar_movies)]['title'].tolist()
    return recommended_movies

if __name__ == '__main__':
    app.run(debug=True)
