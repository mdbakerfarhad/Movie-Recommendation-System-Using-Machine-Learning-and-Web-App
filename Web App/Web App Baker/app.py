# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify, render_template
# import pickle
# app = Flask(__name__)

# #----------------------------------------------Machine Learning Script for recommendation of movies-----------------------------------------------
# ratings = pd.read_csv('ratings.csv')
# movies = pd.read_csv('movies.csv')
# # we basically need the rating and userID
# # A DataFrame object has two axes: “axis 0” and “axis 1”. “axis 0” represents rows and “axis 1” represents columns
# ratings = pd.merge(movies,ratings).drop(['genres','timestamp'],axis=1)

# user_ratings = ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
# # Let's drop/remove the movies which have less than 10 users who rated it and fill remaining NaN with 0
# user_ratings = user_ratings.dropna(thresh=10,axis=1).fillna(0)

# # we will build our similarity matrix
# # user inbuilt method for in-build correlation
# # it will automatically standardize the ratings
# item_similarity_df = user_ratings.corr(method='pearson')
# item_similarity_df.to_csv('item_similarity_df.csv') 

# # make recommendations
# # this method will return a similarity score
# def get_similar_movies(movie_name,user_rating):
#     # subtract user_rating (used for scaling) by mean of rating to accomodate disliked movies
#     similar_score = item_similarity_df[movie_name]*(float(user_rating)-2.5)
#     similar_movies = similar_score.sort_values(ascending=False)
#     return similar_movies
  
# def getRecommendations(movie,rating):
#     try:
#         similar_movies = pd.DataFrame()
#         similar_movies = similar_movies.append(get_similar_movies(movie,rating),ignore_index=True)
#         all_recommend = similar_movies.sum().sort_values(ascending=False)
#         m = all_recommend[0:15].to_string()
#         m = m.split("\n")
#         l=[]
#         for i in m:
#             i = i.split("  ")
#             l.append(i[0])
#         return l
#         s = ""
#         for i in l:
#             s+=i+"  "
#         return(s)
#     except:
#         return("Sorry No suggestions available !")


# #----------------------------------------------------------------------------------------------------------------------------------------------------

# # root api direct to index.html (home page)
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/recommend',methods=['POST'])
# def recommend():
#     '''
#     For rendering results on HTML GUI

#     '''
#     features = [str(x) for x in request.form.values()]
#     print(features)
#     movie_name = str(features[0])
#     movie_rating = float(features[1]) 
#     print(movie_name)
#     output = getRecommendations(movie_name)
#     return render_template('index.html', recommended_movie=output)

# if __name__ == "__main__":
#     app.run(debug=True)


# Import required libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Initialize the Flask app
app = Flask(__name__)

# Load movie data (for demonstration purposes)
movies = pd.read_csv('movies.csv')

# Create a TF-IDF vectorizer and compute the TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Define a function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Check if the provided title exists in the movies DataFrame
    if title not in movies['title'].values:
        return ["Movie not found"]

    idx = movies.index[movies['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Define a route to render the home page
@app.route('/')
def home():
    movie_titles = movies['title'].tolist()
    return render_template('index.html', movie_titles=movie_titles)

# Define a route to handle movie recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form.get('title')
    recommended_movies = get_recommendations(title)
    return render_template('recommendations.html', title=title, recommendations=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)













