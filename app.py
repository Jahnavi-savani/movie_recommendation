# import streamlit as st
# import pandas as pd
# from difflib import get_close_matches
# from surprise import SVD, Dataset, Reader, accuracy
# from surprise.model_selection import train_test_split
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity

# #Load Datasets 
# movies = pd.read_csv("movie.csv")
# ratings = pd.read_csv("rating.csv")
# genome_scores = pd.read_csv("genome_scores.csv")
# genome_tags = pd.read_csv("genome_tags.csv")

# # Preprocess
# tag_data = pd.merge(genome_scores, genome_tags, on="tagId")
# movie_tag_matrix = tag_data.pivot_table(index="movieId", columns="tag", values="relevance", fill_value=0)
# movie_features = pd.merge(movies[['movieId', 'title']], movie_tag_matrix, on="movieId")
# movie_index = pd.Series(movie_features.index, index=movie_features['title'].str.lower())

# # Hybrid Recommender 
# @st.cache_data
# def recommend_movies(title, top_n=5):
#     matches = get_close_matches(title.lower(), movie_index.keys(), n=1, cutoff=0.6)
#     if not matches:
#         return []
    
#     idx = movie_index.get[matches[0]]
#     if idx is None:
#         return []

#     cos_sim = cosine_similarity([movie_tag_matrix.values[idx]], movie_tag_matrix.values)[0]
#     sim_scores = list(enumerate(cos_sim))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
#     recommended_titles = movie_features.iloc[[i[0] for i in sim_scores]]['title'].tolist()
#     return recommended_titles

# # Streamlit UI 
# st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
# st.title("🎬 Movie Recommendation System")

# st.subheader("🔍 Search a Movie")
# movie_input = st.text_input("Enter a movie title (e.g., 'The Notebook')")

# if st.button("Recommend"):
#     if movie_input.strip() == "":
#         st.warning("Please enter a movie title.")
#     else:
#         recommendations = recommend_movies(movie_input)
#         if not recommendations:
#             st.error("No recommendations found.")
#         else:
#             st.success("Top Recommendations:")
#             for i, title in enumerate(recommendations, 1):
#                 st.write(f"{i}. {title}")

# # Evaluation
# @st.cache_resource
# def evaluate_model():
#     reader = Reader(rating_scale=(0.5, 5.0))
#     data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#     trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
#     model = SVD()
#     model.fit(trainset)
#     predictions = model.test(testset)
    
#     rmse = accuracy.rmse(predictions, verbose=False)
#     mae = accuracy.mae(predictions, verbose=False)
#     return rmse, mae

# # ========== Display Evaluation ==========
# # st.subheader("📊 Model Evaluation")
# # rmse, mae = evaluate_model()  # Must return RMSE, MAE
# # st.metric("RMSE", f"{rmse:.4f}")
# # st.metric("MAE", f"{mae:.4f}")
# # # rmse, mae = evaluate_model()
# # st.metric("RMSE", f"{rmse:.4f}")
# # st.metric("MAE", f"{mae:.4f}")



# # app.py

# # import streamlit as st
# # from test import hybrid_recommendations, evaluate_model_rmse_mae, get_poster_url

# # st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# # st.title("🎥 Movie Recommendation System")
# # st.markdown("Get hybrid recommendations based on both **content** and **user behavior**.")

# # # Sidebar input
# # st.sidebar.header("🔍 Select Movie & User")
# # movie_name = st.text_input("Enter a movie name:", value="The Notebook")
# # user_id = st.number_input("Enter user ID (e.g., 500):", min_value=1, step=1, value=500)
# # top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
# # alpha = st.sidebar.slider("Hybrid alpha (0 = content, 1 = collaborative):", 0.0, 1.0, 0.5)

# # if st.button("Recommend"):
# #     with st.spinner("Fetching recommendations..."):
# #         results = hybrid_recommendations(movie_name, user_id, top_n=top_n)
# #         if results:
# #             st.subheader(f"🎯 Top {top_n} Recommendations for User {user_id}")
# #             for title, score in results:
# #                 poster = get_poster_url(title)
# #                 cols = st.columns([1, 4])
# #                 with cols[0]:
# #                     if poster:
# #                         st.image(poster, width=100)
# #                     else:
# #                         st.write("📷 No image")
# #                 with cols[1]:
# #                     st.markdown(f"**{title}**")
# #                     st.markdown(f"Predicted Score: `{score:.2f}`")
# #         else:
# #             st.error("No recommendations found. Try a different movie or user.")

# # # Metrics section
# # st.markdown("---")
# # st.subheader("📊 Evaluation Metrics")
# # rmse, mae = evaluate_model_rmse_mae()
# # st.metric("RMSE", f"{rmse:.4f}")
# # st.metric("MAE", f"{mae:.4f}")

import streamlit as st
import pandas as pd
from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load data
movie = pd.read_csv("movie.csv")
genome_scores = pd.read_csv("genome_scores.csv")

# Merge and create a movie-tag matrix
movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)

# Reindex movie_tag_matrix using titles for easy access
movie_index = movie.set_index('title')['movieId'].to_dict()
movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_tag_matrix.index)}
index_to_title = movie.set_index('movieId')['title'].to_dict()

# Streamlit UI
st.set_page_config(page_title="🎬 Search a Movie", layout="centered")
st.title("🔍 Search a Movie")
st.markdown("Enter a movie title (e.g., 'The Notebook')")

movie_input = st.text_input("", placeholder="e.g., The Notebook")

@st.cache_data
def recommend_movies(title, top_n=5):
    matches = get_close_matches(title.lower(), [t.lower() for t in movie_index.keys()], n=1, cutoff=0.6)
    if not matches:
        return []
    
    # Retrieve the original title
    matched_title = [t for t in movie_index if t.lower() == matches[0]][0]
    movie_id = movie_index[matched_title]
    
    if movie_id not in movie_tag_matrix.index:
        return []
    
    idx = movie_id_to_index[movie_id]
    
    # Calculate cosine similarity
    cos_sim = cosine_similarity([movie_tag_matrix.iloc[idx]], movie_tag_matrix)[0]
    sim_scores = list(enumerate(cos_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommended_titles = [index_to_title[movie_tag_matrix.index[i[0]]] for i in sim_scores]
    return recommended_titles

if st.button("Recommend"):
    if movie_input:
        recommendations = recommend_movies(movie_input)
        if recommendations:
            st.success("Top Recommendations:")
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"{i}. {rec}")
        else:
            st.error("No recommendations found. Try another title.")
    else:
        st.warning("Please enter a movie title.")
