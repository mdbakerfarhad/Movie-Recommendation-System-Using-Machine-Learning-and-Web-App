import requests
import pandas as pd
import pickle
from flask import Flask,render_template, request

app=Flask(__name__)

movies= pd.read_csv("movies.csv")
movies['title']= movies['title'].apply(lambda x: ' '.join(x.split()[:-1]))

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

similarity = pickle.load(open('cosine_sim.pkl', 'rb'))

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
    
    movie_genres = movies['genres'].iloc[movie_indices].tolist()
    
    return movie_titles, movie_genres


@app.route('/')

def index():
    return render_template('index.html')
 
 
@app.route('/recommend', methods=['POST'])

def recommend():
    
    movie_title = request.form["movie_name"]
    formatted_title = movie_title.title()

    recommended_movie_titles, recommended_movie_genres = get_recommendation(formatted_title)

    return render_template('index.html',

                           recommended_movie_titles=recommended_movie_titles,

                           recommended_movie_genres=recommended_movie_genres)


if __name__ == '__main__':
    app.run(debug=True)