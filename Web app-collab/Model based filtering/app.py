import requests
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
from flask import Flask,render_template, request

app=Flask(__name__)

movies= pd.read_csv('movies.csv')
ratings=pd.read_csv("ratings.csv")

user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

n_factors = 100
svd = TruncatedSVD(n_components=n_factors, random_state=42)
user_matrix = svd.fit_transform(user_item_matrix)
item_matrix = svd.components_

item_factors_df = pd.DataFrame(item_matrix, columns=user_item_matrix.columns)


def get_movie_recommendations(movie_name, top_n=10):
    
    movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]
    # Get the latent factors for the given movie
    movie_factors = item_factors_df[movie_id]

    # Find similar movies based on cosine similarity with the given movie
    similar_movies = item_factors_df.T.dot(movie_factors)

    # Sort the movies by similarity in descending order
    similar_movies = similar_movies.sort_values(ascending=False)

    # Exclude the given movie from the recommendations
    similar_movies = similar_movies.drop(movie_id,errors='ignore')

    # Get the top N recommended movies
    top_recommendations = similar_movies.head(top_n)

    valid_recommendations = top_recommendations.index[top_recommendations.index.isin(movies.index)]
    # Get movie names corresponding to valid movie IDs
    recommended_names = movies.loc[valid_recommendations, 'title'].tolist()

    # Return the recommended movie names
    return recommended_names
 
@app.route('/')

def index():
    movie_names=movies['title'].tolist()
    return render_template('index.html',movie_names=movie_names)
 
 
@app.route('/recommend', methods=['POST'])

def recommend():
    
    movie_name = request.form["movie_name"]
    recommended_movie_titles = get_movie_recommendations(movie_name,top_n=10)

    return render_template('recommendations.html',movie_name=movie_name,

                           recommended_movie_titles=recommended_movie_titles)


if __name__ == '__main__':
    app.run(debug=True)