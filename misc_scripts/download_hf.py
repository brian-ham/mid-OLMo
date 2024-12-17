from huggingface_hub import snapshot_download
import os

def download_hf_repo(repo_id, local_dir, revision=None):
    """
    Download the files from a Hugging Face Hub repository to a local directory.

    Args:
        repo_id (str): The ID of the repository on Hugging Face Hub.
        local_dir (str): The local directory to save the downloaded files.
        revision (str): The branch, tag, or commit ID to download (default is the main branch).
    """
    print(f"Downloading repository {repo_id} to {local_dir}...")
    # Download the snapshot of the repository
    snapshot_download(repo_id=repo_id, cache_dir=local_dir, revision=revision)
    print(f"Repository {repo_id} successfully downloaded to {local_dir}.")


if __name__ == "__main__":
    # Replace with the actual repository ID and target local path
    repo_id = "allenai/OLMo-1B-0724-hf"  # Example repo ID
    local_dir = "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/dolma"  # Local directory for downloaded data
    revision = "step1009000-tokens2115B"  # Optional: Specify a branch, tag, or commit hash to download

    # Download the repository
    download_hf_repo(repo_id, local_dir, revision)
