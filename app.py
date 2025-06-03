import streamlit as st
import pickle
import pandas as pd
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Load data and model
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('recommender.pkl', 'rb'))

st.set_page_config(page_title="Bollywood Movie Recommender", layout="centered")
st.title("🎬 Bollywood Movie Recommender")

# --- Utility Functions ---

@st.cache_data(show_spinner=False)
def get_tmdb_config():
    url = f"https://api.themoviedb.org/3/configuration?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        config = response.json()
        base_url = config['images']['base_url']
        size = 'w500'
        return base_url, size
    except Exception as e:
        st.error("Could not connect to TMDB API. Please check your internet connection or API key.")
        return None, None

def safe_get(url, max_retries=3, delay=2, timeout=10):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            else:
                print(f"Non-200 status: {response.status_code} for {url}")
                print(response.text)
        except requests.ConnectionError as e:
            print(f"ConnectionError: {e} for {url}")
        if attempt < max_retries - 1:
            time.sleep(delay)
    return None

@st.cache_data(show_spinner=False)
def get_poster_url(movie_title, year=None, language='hi'):
    base_url, size = get_tmdb_config()
    if base_url is None or size is None:
        return None
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = safe_get(search_url)
    if response is None:
        return None
    data = response.json()
    # Try best match: language and year
    for result in data.get("results", []):
        if (not year or (result.get("release_date") and result["release_date"].startswith(str(year)))) \
           and result.get("original_language") == language:
            poster_path = result.get("poster_path")
            if poster_path:
                return f"{base_url}{size}{poster_path}"
    # Fallback: first result with a poster
    for result in data.get("results", []):
        poster_path = result.get("poster_path")
        if poster_path:
            return f"{base_url}{size}{poster_path}"
    return None

# --- Recommendation Functions ---

def recommend(movie):
    movie = movie.lower()
    index = movies[movies['Movie Name'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]]['Movie Name'] for i in sorted_movies]

def plot_similarity(movie_name):
    index = movies[movies['Movie Name'] == movie_name].index[0]
    scores = similarity[index]
    top_indices = (-scores).argsort()[1:6]
    fig, ax = plt.subplots()
    colors = cm.viridis(np.linspace(0, 1, len(top_indices)))
    ax.barh(movies.iloc[top_indices]['Movie Name'], scores[top_indices], color=colors)
    ax.set_xlabel('Similarity Score')
    ax.set_title(f'Similarity to {movie_name}')
    st.pyplot(fig)

# --- Streamlit UI ---

tab1, tab2 = st.tabs(["Movie-Based", "Genre-Based"])

with tab1:
    st.subheader("Get recommendations based on a movie")
    selected_movie = st.selectbox(
        "Select or type a movie name:",
        sorted(movies['Movie Name'].unique()),
        placeholder="Type a movie name...",
        key="movie_select"
    )

    if st.button("Get Movie Recommendations", key="movie_btn"):
        closest_match = process.extractOne(selected_movie, movies['Movie Name'].tolist())
        if closest_match and closest_match[1] >= 70:
            recommendations = recommend(closest_match[0])
            st.subheader(f"🎥 Because you liked **{closest_match[0]}**, you might also enjoy:")

            cols = st.columns(3)
            for i, (col, movie) in enumerate(zip(cols * 2, recommendations), 1):
                with col:
                    movie_data = movies[movies['Movie Name'] == movie].iloc[0]
                    year = movie_data['Year'] if 'Year' in movies.columns else None
                    st.markdown(f"**{i}. {movie}**")
                    st.caption(f"Genre: {movie_data['Genre']}")
                    if year:
                        st.caption(f"Year: {year}")
                    if 'Rating' in movies.columns:
                        st.caption(f"⭐ {movie_data['Rating']}/10")
                    # Fetch and display poster
                    poster_url = get_poster_url(movie, year=year, language='hi')
                    time.sleep(0.3)  # Respect TMDB rate limit
                    if poster_url:
                        st.image(poster_url, caption=movie, use_container_width=True)
                    else:
                        st.caption("Poster not available.")
                    st.markdown("---")
            plot_similarity(closest_match[0])
        else:
            st.error("❌ Movie not found. Please check the spelling or try a different name.")

with tab2:
    st.subheader("Browse movies by genre")
    all_genres = sorted(movies['Genre'].unique())
    selected_genre = st.selectbox(
        "Select a genre:",
        all_genres,
        key="genre_select"
    )
    sort_by = st.radio("Sort by:", ["A-Z", "Year", "Rating"], horizontal=True)
    genre_movies = movies[movies['Genre'] == selected_genre]
    if sort_by == "Year" and 'Year' in movies.columns:
        genre_movies = genre_movies.sort_values('Year', ascending=False)
    elif sort_by == "Rating" and 'Rating' in movies.columns:
        genre_movies = genre_movies.sort_values('Rating', ascending=False)
    else:
        genre_movies = genre_movies.sort_values('Movie Name')
    if st.button("Show Genre Movies", key="genre_btn"):
        genre_movies = genre_movies['Movie Name'].tolist()
        if genre_movies:
            st.subheader(f"🎭 Movies in the {selected_genre} genre:")
            for i, movie in enumerate(genre_movies[:10], 1):
                movie_data = movies[movies['Movie Name'] == movie].iloc[0]
                year = movie_data['Year'] if 'Year' in movies.columns else None
                st.markdown(f"**{i}. {movie}**")
                if year:
                    st.markdown(f"Year: {year}")
                if 'Rating' in movies.columns:
                    st.markdown(f"Rating: ⭐ {movie_data['Rating']}/10")
                poster_url = get_poster_url(movie, year=year, language='hi')
                time.sleep(0.3)  # Respect TMDB rate limit
                if poster_url:
                    st.image(poster_url, caption=movie, use_container_width=True)
                else:
                    st.caption("Poster not available.")
                st.markdown("---")
        else:
            st.warning(f"No movies found in the {selected_genre} genre.")
