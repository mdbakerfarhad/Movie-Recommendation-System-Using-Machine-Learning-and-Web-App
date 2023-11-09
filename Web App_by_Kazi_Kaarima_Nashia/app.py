import requests
import pandas as pd
import pickle
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask,render_template, request


app=Flask(__name__)

movies= pd.read_csv("movies.csv")

# Create a TF-IDF matrix for movie genres
tfidf= TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate cosine similarity
similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendation(title):
    
    
    idx= movies[movies['title']==title].index[0]

    # get pairwise similarity scores of all movies with the selected movie

    sim_scores = list(enumerate(similarity[idx]))

    # sort the movies based on similarity scores in descending order

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get top 20 similar movies (excluding the selected movie)

    sim_scores = sim_scores[1:10]

    # get titles and posters of the recommended movies

    movie_indices = [i[0] for i in sim_scores]

    movie_titles = movies['title'].iloc[movie_indices].tolist()
    return movie_titles


@app.route('/')

def index():
    movie_names=movies['title'].tolist()
    return render_template('index.html',movie_names=movie_names)
 
 
@app.route('/recommend', methods=['POST'])

def recommend():
    
    movie_name = request.form["movie_name"]

    recommended_movie_titles = get_recommendation(movie_name)

    return render_template('recommendations.html',movie_name=movie_name,

                           recommended_movie_titles=recommended_movie_titles)


if __name__ == '__main__':
    app.run(debug=True)