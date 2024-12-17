import numpy as np
import os
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import requests
from pathlib import Path


def token_count(file_path):
    """
    Count the number of tokens in a memory-mapped .npy file.

    Args:
        file_path (str): Path to the .npy file.

    Returns:
        int: Number of tokens in the file.
    """
    tokens = np.load(file_path, mmap_mode='r')
    num_tokens = tokens.shape[0]  # Assuming tokens are stored in a 1D array
    return num_tokens


def inspect_npy_file(file_path, show_sample=30):
    """
    Inspect the items in a .npy file.

    Args:
        file_path (str): Path to the .npy file.
        show_sample (int): Number of items to sample and display.
    """
    try:
        # Load the .npy file
        data = np.load(file_path, mmap_mode='r', allow_pickle=True)
        
        print(f"File: {file_path}")
        print(f"Data type: {data.dtype}")
        print(f"Shape: {data.shape}")
        print(f"Number of items: {data.size}")

        # Display a sample of items
        print("\nSample of items:")
        print(data[:show_sample])

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data[:show_sample]


def combine_until_limit(folder_path, token_limit=1_000_000_000):
    """
    Combines .npy files until the total number of tokens is approximately token_limit...
    The output file is named data-1B.npy and saved in the same folder as the input files.

    Args:
        folder_path (str): Path to the folder containing .npy files.
        token_limit (int): Target number of tokens to combine (default is 1 billion).
    """
    output_file = os.path.join(folder_path, "data-1B.npy")
    if os.path.exists(output_file):
        print("data-1B.npy already exists in the folder. Exiting...")
        return

    memmap_files = [file for file in os.listdir(folder_path) if file.endswith(".npy")]
    memmap_files.sort()  # Ensure consistent ordering of files
    print(f"Found {len(memmap_files)} memmap files. Target token limit: {token_limit}")

    combined_data = []
    total_tokens = 0

    for file in memmap_files:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path, mmap_mode='r')
        num_tokens = data.shape[0]

        if total_tokens + num_tokens <= token_limit:
            combined_data.append(data)
            total_tokens += num_tokens
            print(f"Added {file}: {num_tokens} tokens (Total: {total_tokens})")
        else:
            # Add only the portion of the current file needed to hit the limit
            remaining_tokens = token_limit - total_tokens
            if remaining_tokens > 0:
                combined_data.append(data[:remaining_tokens])
                total_tokens += remaining_tokens
                print(f"Partially added {file}: {remaining_tokens} tokens (Total: {total_tokens})")
            break

    # Combine and save the final data
    combined_data = np.concatenate(combined_data)
    np.save(output_file, combined_data)
    print(f"Combined data written to {output_file} with {total_tokens} tokens.")


def combine_for_eval(folder_path, token_limit=100_000):
    """
    Combines .npy files until the total number of tokens is approximately token_limit.
    Minimizes overlap with files used in the previous function by reversing the file order.
    The output file is named data-eval.npy and saved in the same folder as the input files.

    Args:
        folder_path (str): Path to the folder containing .npy files.
        token_limit (int): Target number of tokens to combine (default is 1 billion).
    """
    output_file = os.path.join(folder_path, "data-eval.npy")
    if os.path.exists(output_file):
        print("eval-data.npy already exists in the folder. Exiting...")
        return

    # Get and reverse the order of memmap files
    memmap_files = [file for file in os.listdir(folder_path) if file.endswith(".npy")]
    memmap_files.sort(reverse=True)  # Reverse the order to minimize overlap
    print(f"Found {len(memmap_files)} memmap files. Target token limit: {token_limit}")
    # exclude any files called data-combined.npy or data-1B.npy or data-eval.npy
    memmap_files = [file for file in memmap_files if file not in ["data-combined.npy", "data-1B.npy", "data-eval.npy"]]
    print(f"Excluding data-combined.npy, data-1B.npy, data-eval.npy")
    combined_data = []
    total_tokens = 0

    for file in memmap_files:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path, mmap_mode="r")
        num_tokens = data.shape[0]

        if total_tokens + num_tokens <= token_limit:
            combined_data.append(data)
            total_tokens += num_tokens
            print(f"Added {file}: {num_tokens} tokens (Total: {total_tokens})")
        else:
            # Add only the portion of the current file needed to hit the limit
            remaining_tokens = token_limit - total_tokens
            if remaining_tokens > 0:
                combined_data.append(data[:remaining_tokens])
                total_tokens += remaining_tokens
                print(f"Partially added {file}: {remaining_tokens} tokens (Total: {total_tokens})")
            break

    # Combine and save the final data
    combined_data = np.concatenate(combined_data)
    np.save(output_file, combined_data)
    print(f"Eval data written to {output_file} with {total_tokens} tokens.")


def combine_all_files_exclude_last(folder_path):
    """
    Combines all .npy files in a given folder into a single .npy file,
    excluding the last file so that it can be used for evaluation.
    The output file is named data-combined.npy and saved in the same folder.

    Args:
        folder_path (str): Path to the folder containing .npy files.
    """
    output_file = os.path.join(folder_path, "data-combined.npy")
    if os.path.exists(output_file):
        # Overwrite the existing file
        os.remove(output_file)
        print("data-combined.npy already exists in the folder, overwriting...")

    # Find all .npy files in the folder
    memmap_files = [file for file in os.listdir(folder_path) if file.endswith(".npy")]
    memmap_files.sort()  # Ensure consistent ordering of files
    print(f"Found {len(memmap_files)} memmap files.")

    if len(memmap_files) <= 1:
        print("Not enough files to combine (at least 2 required to reserve one for evaluation). Exiting...")
        return

    # Exclude the last file
    files_to_combine = memmap_files[:-1]  # Exclude the last file
    print(f"Excluding the last file for evaluation: {memmap_files[-1]}")

    combined_data = []
    total_tokens = 0

    # Loop through and load data
    for file in files_to_combine:
        file_path = os.path.join(folder_path, file)
        try:
            data = np.load(file_path, mmap_mode='r')  # Memory-mapped load
            num_tokens = data.shape[0]
            combined_data.append(data)
            total_tokens += num_tokens
            print(f"Added {file}: {num_tokens} tokens (Total: {total_tokens})")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    # Combine all data into one array
    print("Combining data...")
    combined_data = np.concatenate(combined_data)
    
    # Save the combined data
    np.save(output_file, combined_data)
    print(f"Combined data written to {output_file} with {total_tokens} tokens.")


def download_memmap(url, save_path):
    """
    Download a .npy file from a public URL and save it to a specified path.

    Args:
        url (str): Public URL of the .npy file.
        save_path (str): Path to save the downloaded file.

    Returns:
        str: Path to the saved file.

    Raises:
        Exception: If the download fails.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    try:
        print(f"Downloading: {url}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad responses (e.g., 404, 500)
            total_size = int(response.headers.get("Content-Length", 0))
            with open(save_path, "wb") as file:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):  # Download in 8KB chunks
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    print(f"\rDownloaded {downloaded_size / 1024**2:.2f} MB / {total_size / 1024**2:.2f} MB", end="")
        print(f"\nDownload complete: {save_path}")
        return str(save_path)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise


def decode_tokens(tokenizer, tokens):
    """
    Decode a list of tokens using a tokenizer.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        tokens (List[int]): List of token IDs.

    Returns:
        str: Decoded text.
    """
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text

if __name__ == "__main__":

    ######## FOR COMBINING FILES
    folder_paths = ["/n/netscratch/sham_lab/Everyone/bham/mid_olmo/web_samples_v2/memmap/prompts",
                    "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/web_samples_v2/memmap/texts",
                    "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/web_samples_v1/memmap/prompts",
                    "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/web_samples_v1/memmap/texts"]
    # for folder_path in folder_paths:
    #     total_count = 0

    #     # Count tokens in all files
    #     for file in os.listdir(folder_path):
    #         file_path = os.path.join(folder_path, file)
    #         num_tokens = token_count(file_path)
    #         total_count += num_tokens
    #         print("Number of tokens in file", file, ":", num_tokens)

    #     print(
    #         f"Number of tokens in directory {folder_path.split('/')[-2]}/{folder_path.split('/')[-1]}: {total_count}, "
    #         f"avg: {np.round(total_count / len(os.listdir(folder_path)))}"
    #     )

    #     # Combine all files in folder_path and save to data-combined.npy
    #     combine_all_files_exclude_last(folder_path)



    ######## FOR COMBINING UP TO LIMIT

    # # Combine files to create data-1B.npy
    # for folder_path in folder_paths:
    #     combine_until_limit(folder_path, token_limit=1_000_000_000)


    # ####### FOR CREATING EVAL SET
    # for folder_path in folder_paths:
    #     combine_for_eval(folder_path, token_limit=100_000)


    ############# FOR INSPECTING A FILE
    # save_path = "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/web_samples_v1/memmap/texts/data-1B.npy"

    # tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B")
    
    # examples = inspect_npy_file(save_path, show_sample=3000)
    # decoded_text = decode_tokens(tokenizer, examples)
    # print("\nDecoded text sample:")
    # print(decoded_text)


    #### FOR COUNTING TOKENS
    npy_path = "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/web_samples_v1/memmap/prompts/data-combined.npy"
    num_tokens = token_count(npy_path)
    print(f"Number of tokens in {npy_path}: {num_tokens}")