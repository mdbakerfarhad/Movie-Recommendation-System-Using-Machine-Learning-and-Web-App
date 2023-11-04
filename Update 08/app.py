import pickle
import streamlit as st
import requests


st.header("Movie Recommendation System Using Machine Learning")
st.caption("By Md. Baker, Kazi Kaarima Nashia, Fatema Akter Rimi")


movies=pickle.load(open('movielist.pkl','rb')) 
similarity = pickle.load(open('similarity.pkl','rb'))
movielist=movies['title'].values
st.selectbox("Type or select a movie name", movielist)


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
 
    for i in distances[1:6]:
     
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names


if st.button('Show Recommendation'):
    recommended_movie_names = recommend(movielist)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write(recommended_movie_names)
        
    with col2:
        st.write(recommended_movie_names[1])
        

    with col3:
        st.write(recommended_movie_names[2])
      
    with col4:
        st.write(recommended_movie_names[3])
       
    with col5:
        st.write(recommended_movie_names[0])
       