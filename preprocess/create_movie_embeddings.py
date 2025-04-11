import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Paths to input and output files
input_file = "data/ml-1m/movie_metadata.json"
output_file = "data/ml-1m/movie_embeddings.p"

# Load the Hugging Face model and tokenizer
model_name = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]



# Function to create embeddings
def create_embeddings(metadata_file: str, output_file: str) -> None:
    # Load movie metadata
    with open(metadata_file, "r") as f:
        movie_data = json.load(f)

    embeddings = {}

    # Generate embeddings for each movie
    for movie_id, movie_info in tqdm(movie_data.items(), total=len(movie_data)):
        # Combine metadata fields into a single input text
        title = movie_info.get("title", "")
        categories = ", ".join(movie_info.get("categories", []))
        description = movie_info.get("description", "")
        input_text = f"Title: {title}. Categories: {categories}. Description: {description}"

        # Tokenize and create embeddings
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = average_pool(outputs.last_hidden_state, inputs['attention_mask']).squeeze().tolist() # Mean pooling

        # Store the embedding
        embeddings[movie_id] = embedding

    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Embeddings saved to {output_file}")


# Main function
if __name__ == "__main__":
    create_embeddings(input_file, output_file)
