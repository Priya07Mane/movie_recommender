import streamlit as st
import pickle
import pandas as pd
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import requests
from functools import lru_cache
import os
import json

TMDB_API_KEY = "5bf323a67be44067a3d3232aa2f9dd5c"

DEFAULT_POSTER = "https://via.placeholder.com/150x225?text=No+Poster"

POSTER_CACHE_FILE = "poster_cache.json"
# Load existing cache
if os.path.exists(POSTER_CACHE_FILE):
    with open(POSTER_CACHE_FILE) as f:
        poster_cache = json.load(f)
else:
    poster_cache = {}

def get_persistent_poster(movie_name):
    if movie_name in poster_cache:
        return poster_cache[movie_name]
    
    poster_url = fetch_poster(movie_name, TMDB_API_KEY)
    poster_cache[movie_name] = poster_url
    
    # Save to disk
    with open(POSTER_CACHE_FILE, 'w') as f:
        json.dump(poster_cache, f)
    
    return poster_url

# Load data and model
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('recommender.pkl', 'rb'))

# Set title
st.set_page_config(page_title="Bollywood Movie Recommender", layout="centered")
st.title("🎬 Bollywood Movie Recommender")

# Recommendation function (used by all tabs)
def recommend(movie):
    movie = movie.lower()
    index = movies[movies['Movie Name'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]]['Movie Name'] for i in sorted_movies]

# Function to plot similarity scores
def plot_similarity(movie_name):
    index = movies[movies['Movie Name'] == movie_name].index[0]
    scores = similarity[index]
    top_indices = (-scores).argsort()[1:6]
    
    fig, ax = plt.subplots()
    ax.barh(movies.iloc[top_indices]['Movie Name'], scores[top_indices])
    ax.set_xlabel('Similarity Score')
    ax.set_title(f'Similarity to {movie_name}')
    st.pyplot(fig)
 
@lru_cache(maxsize=500)
def fetch_poster(movie_name, api_key):
    try:
        year = movies[movies['Movie Name'] == movie_name]['Year'].values[0] if 'Year' in movies.columns else ""
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name} {year}&include_adult=false"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('results'):
            for result in data['results'][:3]:  # Check first 3 results
                if result.get('poster_path'):
                    return f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
        return DEFAULT_POSTER
    except Exception as e:
        st.error(f"poster for {movie_name} not found")
        return DEFAULT_POSTER

# Create tabs for different recommendation methods
tab1, tab2 = st.tabs(["Movie-Based", "Genre-Based"])

with tab1:
    # Movie-based recommendations
    st.subheader("Get recommendations based on a movie")
    selected_movie = st.selectbox(
        "Select or type a movie name:",
        sorted(movies['Movie Name'].unique()),
        placeholder="Type a movie name...",
        key="movie_select"
    )
  
    if st.button("Get Movie Recommendations", key="movie_btn"):
        # Fuzzy matching
        closest_match = process.extractOne(selected_movie, movies['Movie Name'].tolist())
        if closest_match[1] >= 70:
            recommendations = recommend(closest_match[0])
            st.subheader(f"🎥 Because you liked **{closest_match[0]}**, you might also enjoy:")

            cols = st.columns(3)  # 3 columns for better mobile responsiveness
            for i, (col, movie) in enumerate(zip(cols * 2, recommendations), 1):  # Wrap columns if >3
                with col:
                    movie_data = movies[movies['Movie Name'] == movie].iloc[0]
                    poster_url = fetch_poster(movie, TMDB_API_KEY)
                    
                    # Movie Card
                    with st.container():
                        # Poster (center-aligned)
                        st.image(poster_url if poster_url and "None" not in poster_url else DEFAULT_POSTER, 
                                width=150, use_container_width='always')
                        
                        # Movie Details
                        st.markdown(f"**{i}. {movie}**")
                        st.caption(f"Genre: {movie_data['Genre']}")
                        
                        if 'Year' in movies.columns:
                            st.caption(f"Year: {movie_data['Year']}")
                        if 'Rating' in movies.columns:
                            st.caption(f"⭐ {movie_data['Rating']}/10")
                        
                        st.markdown("---")  # Divider
            
            plot_similarity(closest_match[0])
        else:
            st.error("❌ Movie not found. Please check the spelling or try a different name.")

with tab2:
    # Genre-based filtering
    st.subheader("Browse movies by genre")
    
    # Get unique genres (assuming single genre per movie)
    all_genres = sorted(movies['Genre'].unique())
    selected_genre = st.selectbox(
        "Select a genre:",
        all_genres,
        key="genre_select"
    )
    # Add sorting options
    sort_by = st.radio("Sort by:", ["A-Z", "Year", "Rating"], horizontal=True)
    
    genre_movies = movies[movies['Genre'] == selected_genre]
    
    if sort_by == "Year" and 'Year' in movies.columns:
        genre_movies = genre_movies.sort_values('Year', ascending=False)
    elif sort_by == "Rating" and 'Rating' in movies.columns:
        genre_movies = genre_movies.sort_values('Rating', ascending=False)
    else:
        genre_movies = genre_movies.sort_values('Movie Name')
    

    if st.button("Show Genre Movies", key="genre_btn"):
        genre_movies = movies[movies['Genre'] == selected_genre]['Movie Name'].tolist()
        if genre_movies:
            st.subheader(f"🎭 Movies in the {selected_genre} genre:")
            for i, movie in enumerate(genre_movies[:15], 1):  # Show first 15 to avoid clutter
                movie_data = movies[movies['Movie Name'] == movie].iloc[0]
                poster_url = fetch_poster(movie, TMDB_API_KEY)
                
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if poster_url:
                            st.image(poster_url or DEFAULT_POSTER, width=120)
                    
                    with col2:
                        st.markdown(f"**{i}. {movie}**")
                        if 'Year' in movies.columns:
                            st.markdown(f"Year: {movie_data['Year']}")
                        if 'Rating' in movies.columns:
                            st.markdown(f"Rating: ⭐ {movie_data['Rating']}/10")
                    
                    st.markdown("---")
        else:
            st.warning(f"No movies found in the {selected_genre} genre.")
       