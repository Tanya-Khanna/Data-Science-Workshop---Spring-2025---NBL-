import pandas as pd
import weaviate
import requests
import base64
import toml
import os
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5
from weaviate.classes.config import Configure, DataType, Property

weaviate_api_key = ""
weaviate_url = ""
cohere_api_key = ""
tmdb_api_key = ""

# --- Function to fetch poster via TMDB ---
def fetch_tmdb_poster(imdb_id, tmdb_api_key):
    try:
        url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={tmdb_api_key}&external_source=imdb_id"
        response = requests.get(url)
        data = response.json()

        if data.get("movie_results"):
            poster_path = data["movie_results"][0].get("poster_path")
            if poster_path:
                full_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                img_data = requests.get(full_url).content
                return base64.b64encode(img_data).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Couldn't fetch poster for {imdb_id}: {e}")
    return None

# --- Connect to Weaviate Cloud ---
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
    headers={"X-Cohere-Api-Key": cohere_api_key}
)

# Delete existing MovieDemo collection if it exists
client.collections.delete(["MovieDemo"])

# Create MovieDemo collection
movies = client.collections.create(
    name="MovieDemo",
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="overview", data_type=DataType.TEXT),
        Property(name="tagline", data_type=DataType.TEXT),
        Property(name="movie_id", data_type=DataType.INT, skip_vectorization=True),
        Property(name="release_year", data_type=DataType.INT),
        Property(name="genres", data_type=DataType.TEXT_ARRAY),
        Property(name="vote_average", data_type=DataType.NUMBER),
        Property(name="vote_count", data_type=DataType.INT),
        Property(name="revenue", data_type=DataType.INT),
        Property(name="budget", data_type=DataType.INT),
        Property(name="poster", data_type=DataType.BLOB),
    ],
    vectorizer_config=Configure.Vectorizer.text2vec_cohere(),
    vector_index_config=Configure.VectorIndex.hnsw(
        quantizer=Configure.VectorIndex.Quantizer.bq()
    ),
    generative_config=Configure.Generative.cohere(model="command-r-plus"),
)

# Load movies JSON
json_file_path = os.path.join(os.getcwd(), "helpers/data/1950_2024_movies_info.json")
movies_df = pd.read_json(json_file_path)

# Add movies in batch
with movies.batch.fixed_size(100) as batch:
    for i, movie_row in tqdm(movies_df.iterrows(), total=len(movies_df)):
        try:
            date_object = datetime.strptime(movie_row["release_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            
            props = {
                "title": movie_row["title"],
                "overview": movie_row["overview"],
                "tagline": movie_row["tagline"],
                "vote_count": movie_row["vote_count"],
                "vote_average": movie_row["vote_average"],
                "revenue": movie_row["revenue"],
                "budget": movie_row["budget"],
                "movie_id": movie_row["id"],
                "release_year": date_object.year,
                "genres": [genre["name"] for genre in movie_row["genres"]],
            }

            # Fetch and add poster
            poster_b64 = fetch_tmdb_poster(movie_row["imdb_id"], tmdb_api_key)
            if poster_b64:
                props["poster"] = poster_b64

            batch.add_object(properties=props, uuid=generate_uuid5(movie_row["id"]))

        except Exception as e:
            print(f"[ERROR] Failed to process movie {movie_row.get('title', '')}: {e}")
            continue

# Close client
client.close()
