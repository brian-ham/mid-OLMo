import os
import requests

# Dataset dictionary
datasets = {
    "v3-small-c4_en-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/c4_en/val/part-0-00000.npy"
    ],
    "v3-small-dolma_books-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_books/val/part-0-00000.npy"
    ],
    "v3-small-dolma_common-crawl-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_common-crawl/val/part-0-00000.npy"
    ],
    "v3-small-dolma_reddit-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_reddit/val/part-0-00000.npy"
    ],
    "v3-small-dolma_wiki-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/dolma_wiki/val/part-0-00000.npy"
    ],
    "v3-small-wikitext_103-validation": [
        "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/wikitext_103/val/part-0-00000.npy"
    ]
}

# Target directory for saving the datasets
target_directory = "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/eval_data/"

# Ensure the directory exists
os.makedirs(target_directory, exist_ok=True)

# Function to download a file
def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {url} -> {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Loop through the datasets and download each file
for dataset_name, urls in datasets.items():
    for url in urls:
        # Construct a save path with a meaningful filename
        filename = f"{dataset_name}.npy"
        save_path = os.path.join(target_directory, filename)
        # Download the file
        download_file(url, save_path)

print("All downloads completed.")