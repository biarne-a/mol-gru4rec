import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from typing import Set, Dict, Any
from collections import Counter

# Paths to input files
movies_file: str = "data/ml-1m/movies.csv"
links_file: str = "data/links.csv"
output_file: str = "data/ml-1m/movie_metadata.json"

# Step 1: Read 'movies.dat' and extract unique movie IDs
def get_movie_data(movies_file: str) -> tuple[set[str], dict[str, str], dict[str, str]]:
    movies_df: pd.DataFrame = pd.read_csv(movies_file, sep="::", engine="python", header=None, names=["movieId", "title", "genres"])
    unique_movie_ids = set(movies_df["movieId"].astype(str))
    titles_dict = {str(row["movieId"]): row["title"] for _, row in movies_df.iterrows()}
    genres_dict = {str(row["movieId"]): row["genres"].split("|") for _, row in movies_df.iterrows()}
    return unique_movie_ids, titles_dict, genres_dict


# Step 2: Read 'links.csv' and create a mapping of movieId to imdbId
def get_movie_to_imdb_mapping(links_file: str) -> Dict[str, str]:
    links_df: pd.DataFrame = pd.read_csv(links_file)
    return {str(row["movieId"].astype(int)): str(row["imdbId"].astype(int)).zfill(7) for _, row in links_df.iterrows()}

# Step 3: Filter mapping to only include movieIds from 'movies.dat'
def filter_mapping(movie_ids: Set[str], movie_to_imdb: Dict[str, str]) -> Dict[str, str]:
    return {movieId: imdbId for movieId, imdbId in movie_to_imdb.items() if movieId in movie_ids}

# Step 4: Scrape IMDb for metadata
def scrape_imdb_metadata(filtered_mapping: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    base_url: str = "https://www.imdb.com/title/tt"

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    }
    home_request = requests.get("https://www.imdb.com", headers=headers)

    for movieId, imdbId in tqdm(filtered_mapping.items(), total=len(filtered_mapping)):
        url: str = f"{base_url}{imdbId}"

        response: requests.Response = requests.get(url, headers=headers, cookies=home_request.cookies)
        if response.status_code != 200:
            print(f"Failed to fetch data for movieId {movieId} (imdbId {imdbId})")
            continue

        soup: BeautifulSoup = BeautifulSoup(response.content, "html.parser")

        # Extract title
        title_element = soup.find("span", class_="hero__primary-text")
        title: str = title_element.text.strip() if title_element else None

        # Extract categories
        categories_element = soup.find("div", class_="ipc-chip-list__scroller")
        categories: list[str] = [chip.text.strip() for chip in categories_element.find_all("span")] if categories_element else []

        # Extract short description
        description_element = soup.find("span", class_="sc-42125d72-2 mmImo")
        description: str = description_element.text.strip() if description_element else None

        # Store metadata
        metadata[movieId] = {
            "imdbId": imdbId,
            "title": title,
            "categories": categories,
            "description": description
        }

    return metadata

# Step 5: Save metadata to a JSON file
def save_metadata(
    metadata: Dict[str, Dict[str, Any]],
    movie_ids: Set[str],
    titles_dict: dict[str, str],
    genres_dict: dict[str, str],
    output_file: str
) -> None:
    # Fill in missing titles and categories
    for movie_id, movie_info in metadata.items():
        if not movie_info.get("title") and movie_id in titles_dict:
            movie_info["title"] = titles_dict[movie_id]
        if not movie_info.get("categories") and movie_id in genres_dict:
            movie_info["categories"] = genres_dict[movie_id]
    # Add missing movies (the ones for which scrapping failed)
    handled_ids = set(metadata.keys())
    missing_ids = movie_ids.difference(handled_ids)
    for movie_id in missing_ids:
        metadata[movie_id] = {
            "title": titles_dict.get(movie_id, None),
            "categories": genres_dict.get(movie_id, []),
        }

    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=4)


def analyze_missing_fields(file_path: str) -> None:
    # Load the JSON data
    with open(file_path, "r") as f:
        movie_data = json.load(f)

    # Initialize a counter for missing fields
    missing_fields_count = Counter()

    # Iterate through each movie and check for missing fields
    movie_ids_with_missing_info = []
    for movie_id, movie_info in movie_data.items():
        for field in ["title", "categories"]:
            if not movie_info.get(field):  # Check if the field is missing or empty
                missing_fields_count[field] += 1

    # Print the results
    print("Missing fields analysis:")
    for field, count in missing_fields_count.items():
        print(f"{field}: {count} missing")


# Main function
def main() -> None:
    movie_ids, titles_dict, genres_dict = get_movie_data(movies_file)
    movie_to_imdb: Dict[str, str] = get_movie_to_imdb_mapping(links_file)
    filtered_mapping: Dict[str, str] = filter_mapping(movie_ids, movie_to_imdb)
    metadata: Dict[str, Dict[str, Any]] = scrape_imdb_metadata(filtered_mapping)
    save_metadata(metadata, titles_dict, genres_dict, output_file)
    print(f"Metadata saved to {output_file}")
    analyze_missing_fields(output_file)


if __name__ == "__main__":
    # main()
    # analyze_missing_fields(output_file)
    with open(output_file, "r") as f:
        metadata = json.load(f)
    movie_ids, titles_dict, genres_dict = get_movie_data(movies_file)
    save_metadata(metadata, movie_ids, titles_dict, genres_dict, output_file)

    # pickle.lo
