import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Paths to input and output files
input_file = "data/ml-1m/movie_metadata.json"
output_file = "data/ml-1m/movie_embeddings.json"

# Load the Hugging Face model and tokenizer
model_name = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to create embeddings
def create_embeddings(metadata_file: str, output_file: str) -> None:
    # Load movie metadata
    with open(metadata_file, "r") as f:
        movie_data = json.load(f)

    embeddings = {}

    # Generate embeddings for each movie
    for movie_id, movie_info in tqdm(movie_data.items(), desc="Generating embeddings"):
        # Combine metadata fields into a single input text
        title = movie_info.get("title", "")
        categories = ", ".join(movie_info.get("categories", []))
        description = movie_info.get("description", "")
        input_text = f"Title: {title}. Categories: {categories}. Description: {description}"

        # Tokenize and create embeddings
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling

        # Store the embedding
        embeddings[movie_id] = embedding

    # Save embeddings to a JSON file
    with open(output_file, "w") as f:
        json.dump(embeddings, f, indent=4)

    print(f"Embeddings saved to {output_file}")


# Main function
if __name__ == "__main__":
    create_embeddings(input_file, output_file)
