
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# TMDB API CONFIGURATION
# ---------------------------------------------------------

try:
    TMDB_ACCESS_TOKEN = st.secrets["TMDB_ACCESS_TOKEN"]
except (KeyError, FileNotFoundError):
    st.error(
        "TMDB token not found. Create .streamlit/secrets.toml "
        "and add your TMDB Read Access Token."
    )
    st.stop()


# ---------------------------------------------------------
# FETCH MOVIE POSTERS FROM TMDB
# ---------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()

        poster_path = data.get("poster_path")

        if poster_path:
            return (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )

    except (requests.RequestException, ValueError):
        return None

    return None


def show_poster(movie_id):

    poster = fetch_poster(movie_id)

    if poster:
        st.image(
            poster,
            use_container_width=True
        )
    else:
        st.write("Poster unavailable")


# ---------------------------------------------------------
# FRONTEND STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    @import url(
    'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(
              ellipse at 10% 0%,
              rgba(120, 20, 35, .22),
              transparent 35%
          ),
          #090b10;
        color: #f5f5f5;
    }

    [data-testid="stHeader"] {
        background: rgba(9,11,16,0.8);
    }

    .block-container {
        padding-top: 1.5rem;
        max-width: 1450px;
    }

    .brand {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #fff;
        margin: 0;
    }

    .brand span {
        color: #e50914;
    }

    .eyebrow {
        color: #e50914;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-size: .72rem;
        font-weight: 800;
    }

    .hero-title {
        font-size: clamp(2.2rem, 5vw, 4.3rem);
        line-height: 1.03;
        letter-spacing: -2px;
        font-weight: 800;
        margin: .4rem 0 .8rem;
        color: #fff;
    }

    .muted {
        color: #a5a7b0;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #fff;
        margin: 1.3rem 0 .2rem;
    }

    .section-subtitle {
        color: #9296a2;
        font-size: .9rem;
        margin-bottom: 1rem;
    }

    .movie-title {
        color: #fff;
        font-size: .95rem;
        font-weight: 700;
        line-height: 1.25;
        min-height: 2.4rem;
        margin-top: .55rem;
    }

    .movie-meta {
        color: #9da1ad;
        font-size: .78rem;
    }

    .tag {
        display: inline-block;
        border: 1px solid #353945;
        color: #c8cad1;
        border-radius: 30px;
        padding: 4px 10px;
        font-size: .72rem;
        margin: 2px;
    }

    .detail-panel {
        background: linear-gradient(
            120deg,
            #171a23,
            #101219
        );
        border: 1px solid #292d38;
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin: .5rem 0 1.2rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,.35);
    }

    div[data-baseweb="select"] > div {
        background: #171a23;
        border-color: #363b48;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------------

def parse_names(value, limit=None):

    if not isinstance(value, str) or not value.strip():
        return []

    try:

        records = ast.literal_eval(value)

        names = [
            item.get("name", "").replace(" ", "")
            for item in records
            if isinstance(item, dict) and item.get("name")
        ]

        return names[:limit] if limit else names

    except (ValueError, SyntaxError, TypeError):
        return []


def get_director(value):

    if not isinstance(value, str) or not value.strip():
        return []

    try:

        for person in ast.literal_eval(value):

            if (
                isinstance(person, dict)
                and person.get("job") == "Director"
            ):

                name = person.get("name", "")

                return [name.replace(" ", "")] if name else []

    except (ValueError, SyntaxError, TypeError):
        pass

    return []


# ---------------------------------------------------------
# BUILD RECOMMENDATION ENGINE
# ---------------------------------------------------------

@st.cache_data(
    show_spinner="Building the movie recommendation engine..."
)
def load_recommender(movies_path, credits_path):

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    movie_required = {
        "id", "title", "genres", "keywords", "overview"
    }

    credits_required = {
        "title", "cast", "crew"
    }

    missing_movies = movie_required - set(movies.columns)
    missing_credits = credits_required - set(credits.columns)

    if missing_movies or missing_credits:

        raise ValueError(
            f"Missing columns in movies CSV: {sorted(missing_movies)}; "
            f"missing columns in credits CSV: {sorted(missing_credits)}"
        )

    # Merge movie details with cast and crew.

    df = movies.merge(
        credits[["title", "cast", "crew"]],
        on="title",
        how="inner"
    )

    df = df.dropna(
        subset=[
            "id",
            "title",
            "genres",
            "keywords",
            "overview",
            "cast",
            "crew"
        ]
    ).copy()

    optional = [
        col for col in [
            "release_date",
            "vote_average",
            "vote_count",
            "runtime"
        ]
        if col in df.columns
    ]

    df = df[
        [
            "id",
            "title",
            "genres",
            "keywords",
            "overview",
            "cast",
            "crew"
        ] + optional
    ].copy()

    df = df.drop_duplicates(
        subset=["id"]
    ).reset_index(drop=True)

    # Extract features.

    df["genres_list"] = df["genres"].apply(parse_names)

    df["keywords_list"] = df["keywords"].apply(parse_names)

    df["cast_list"] = df["cast"].apply(
        lambda x: parse_names(x, limit=3)
    )

    df["director_list"] = df["crew"].apply(get_director)

    df["overview_list"] = df["overview"].fillna("").apply(
        lambda x: x.split()
    )

    # Combine all features into tags.

    df["tags"] = (
        df["overview_list"]
        + df["genres_list"]
        + df["keywords_list"]
        + df["cast_list"]
        + df["director_list"]
    )

    df["tags"] = df["tags"].apply(
        lambda words: " ".join(words).lower()
    )

    # Apply stemming.

    stemmer = PorterStemmer()

    df["tags"] = df["tags"].apply(
        lambda text: " ".join(
            stemmer.stem(word)
            for word in text.split()
        )
    )

    # Text vectorization.

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(df["tags"])

    return df, vectors


def year_of(row):

    release_date = row.get("release_date")

    if isinstance(release_date, str) and len(release_date) >= 4:
        return release_date[:4]

    return "Year unavailable"


# ---------------------------------------------------------
# FIND DATASET FILES
# ---------------------------------------------------------

possible_pairs = [

    (
        Path("dataset/tmdb_5000_movies.csv"),
        Path("dataset/tmdb_5000_credits.csv")
    ),

    (
        Path("tmdb_5000_movies.csv"),
        Path("tmdb_5000_credits.csv")
    ),

    (
        Path("data/tmdb_5000_movies.csv"),
        Path("data/tmdb_5000_credits.csv")
    )

]

dataset_pair = next(
    (
        (m, c)
        for m, c in possible_pairs
        if m.exists() and c.exists()
    ),
    None
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

left, right = st.columns([3, 1])

with left:

    st.markdown(
        '<p class="brand">CINE<span>MATCH</span> 🎬</p>',
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        '<p style="text-align:right;color:#a5a7b0;padding-top:12px;">'
        'CONTENT-BASED DISCOVERY</p>',
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="eyebrow">Your next favourite movie is waiting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Find films that<br>feel like your films.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="muted">Choose a movie you love. CineMatch will discover '
    'films with similar stories, genres, keywords, cast and directors.</p>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

if dataset_pair is None:

    st.error("Movie dataset files were not found.")

    st.markdown(
        """
        Put both CSV files in this folder:

        ```text
        Movies_Recommendation_System/
        ├── app.py
        └── dataset/
            ├── tmdb_5000_movies.csv
            └── tmdb_5000_credits.csv
        ```
        """
    )

    st.stop()


try:

    df, vectors = load_recommender(
        str(dataset_pair[0]),
        str(dataset_pair[1])
    )

except Exception as exc:

    st.error(f"Could not prepare the movie data: {exc}")
    st.stop()


if df.empty:

    st.error("No movies are available after preprocessing.")
    st.stop()


# ---------------------------------------------------------
# MOVIE SEARCH
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Explore the collection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Search by movie title to get similar-movie suggestions.</div>',
    unsafe_allow_html=True
)

titles = df["title"].astype(str).tolist()

default_index = (
    titles.index("The Dark Knight")
    if "The Dark Knight" in titles
    else 0
)

search_col, info_col = st.columns([4, 1])

with search_col:

    selected_title = st.selectbox(
        "Search for a movie",
        options=titles,
        index=default_index,
        placeholder="Start typing a movie title...",
        label_visibility="collapsed"
    )

with info_col:

    st.metric(
        "Movies indexed",
        f"{len(df):,}"
    )


selected_index = df.index[
    df["title"] == selected_title
][0]

selected_movie = df.iloc[selected_index]


# ---------------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------------

scores = cosine_similarity(
    vectors[selected_index],
    vectors
).flatten()

ranked_indices = np.argsort(scores)[::-1]

ranked_indices = [
    i for i in ranked_indices
    if i != selected_index
][:10]


# ---------------------------------------------------------
# SELECTED MOVIE DETAILS
# ---------------------------------------------------------

poster_col, details_col = st.columns(
    [1, 3],
    gap="large"
)

with poster_col:

    show_poster(selected_movie["id"])


with details_col:

    st.markdown(
        '<div class="detail-panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        f"### {selected_movie['title']}"
    )

    meta = [year_of(selected_movie)]

    if pd.notna(selected_movie.get("vote_average")):

        meta.append(
            f"⭐ {float(selected_movie['vote_average']):.1f}/10"
        )

    if pd.notna(selected_movie.get("runtime")):

        meta.append(
            f"{int(selected_movie['runtime'])} min"
        )

    st.caption("  •  ".join(meta))

    overview = selected_movie.get("overview", "")

    st.write(
        overview
        if isinstance(overview, str) and overview
        else "No overview available."
    )

    genres = selected_movie.get("genres_list", [])

    if genres:

        st.markdown(
            "".join(
                f'<span class="tag">{g}</span>'
                for g in genres
            ),
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# RECOMMENDATION GALLERY
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Because you watched this</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="section-subtitle">Top similar movies for '
    f'<b>{selected_movie["title"]}</b>, ranked using cosine similarity.</div>',
    unsafe_allow_html=True
)


def display_recommendations(movie_indices):

    columns = st.columns(5, gap="medium")

    for position, movie_index in enumerate(movie_indices):

        movie = df.iloc[movie_index]

        with columns[position]:

            show_poster(movie["id"])

            st.markdown(
                f'<div class="movie-title">{movie["title"]}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="movie-meta">{year_of(movie)}'
                f' &nbsp;·&nbsp; Similarity {scores[movie_index]:.0%}</div>',
                unsafe_allow_html=True
            )


display_recommendations(
    ranked_indices[:5]
)


if len(ranked_indices) > 5:

    st.markdown(
        '<div class="section-title">More like this</div>',
        unsafe_allow_html=True
    )

    display_recommendations(
        ranked_indices[5:10]
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "CineMatch is an educational movie-discovery project. "
    "Recommendations are based on similarity between movie metadata, "
    "not Netflix viewing history. Movie metadata comes from the "
    "TMDB 5000 dataset. Poster images are served by TMDB."
)
