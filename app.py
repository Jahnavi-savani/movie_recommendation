import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load CSV files
movies = pd.read_csv("movie.csv")
tags = pd.read_csv("tag.csv")

# Merge tags for each movie
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(str(i) for i in x)).reset_index()
movie_data = pd.merge(movies, movie_tags, on='movieId', how='left')
movie_data['tag'] = movie_data['tag'].fillna('')

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tag_matrix = tfidf.fit_transform(movie_data['tag'])

# Title lookup dictionary
title_to_index = {title.lower(): idx for idx, title in enumerate(movie_data['title'])}
titles = movie_data['title'].tolist()

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Hybrid Movie Recommender")

movie_input = st.text_input("Enter a movie title", placeholder="e.g. The Matrix")

def recommend_movies(title, top_n=5):
    matches = get_close_matches(title.lower(), title_to_index.keys(), n=1, cutoff=0.6)
    if not matches:
        return []
    idx = title_to_index[matches[0]]

    cosine_sim = cosine_similarity(tag_matrix[idx], tag_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
    recommended = movie_data.iloc[similar_indices]['title'].tolist()
    return recommended

if st.button("Recommend"):
    if movie_input:
        recommendations = recommend_movies(movie_input)
        if recommendations:
            st.success("Top Recommendations:")
            for i, title in enumerate(recommendations, start=1):
                st.write(f"{i}. {title}")
        else:
            st.warning("No similar movies found. Try another title.")
    else:
        st.info("Please enter a movie title above.")
