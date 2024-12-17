import os
from pathlib import Path

import numpy as np
from datasets import Dataset
from tokenizers import Tokenizer
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


def remove_prompt(text: str) -> str:
    """
    Filters the input text from the first instance of a quotation mark (") 
    to the last instance of "Write a" (case-sensitive).

    Args:
        text (str): The input text.

    Returns:
        str: The text between the first quotation mark and the last occurrence 
             of "Write a", or an empty string if not found.
    """
    first_quote = text.find('"')
    last_write_a = text.rfind('".\n\nWrite a')

    if first_quote != -1 and last_write_a != -1 and last_write_a > first_quote:
        return text[first_quote + 1 : last_write_a]
    print("ERROR: Quotation mark or 'Write a' not found in text.")
    return ""



def convert_arrow_to_memmap(arrow_file: str, memmap_dir: str, tokenizer, debug=False) -> None:
    """
    Converts a single .arrow file to two .npy memory-mapped files:
        - One for the original text data

    Args:
        arrow_file (str): Path to the input .arrow file.
        memmap_dir (str): Directory to save the output .npy file.
        tokenizer_path (str): Path to the tokenizer JSON file.
    """
    print("\n------------------------------------")
    print("Converting", arrow_file)

    # Load the .arrow file as a Hugging Face Dataset
    dataset = Dataset.from_file(arrow_file)

    if debug:
        print(f"Dataset Prompt example: {dataset['prompt'][0]}")
        print(f"Dataset Text example: {dataset['text'][0]}")
        print("Dataset Prompt Tokenized example: ", tokenizer.encode(remove_prompt(dataset["prompt"][0])))
        print("Dataset Text Tokenized example: ", tokenizer.encode(dataset["text"][0]))
    prompts_tokenized = [
            tokenizer.encode(remove_prompt(data)) for data in dataset["prompt"]
        ]
    texts_tokenized = [
            tokenizer.encode(data) for data in dataset["text"]
        ]
    
    print(f"File: {arrow_file.rstrip('/').split('/')[-1]}, tokenization complete.")

    # Keep track of: prompts token length, text token length
    prompts_len = sum([len(ids) for ids in prompts_tokenized])
    texts_len = sum([len(ids) for ids in texts_tokenized])

    if debug:
        print(f"Example Prompt before Truncation: {prompts_tokenized[0]}, Length: {len(prompts_tokenized[0])}")
        print(f"Example Text before Truncation: {texts_tokenized[0]}, Length: {len(texts_tokenized[0])}\n")


    # Truncate each entry in texts_tokenized to length of correpsonding prompt in prompts_tokenized
    for i in range(len(texts_tokenized)):
        token_length = min(len(prompts_tokenized[i]), len(texts_tokenized[i]))
        prompts_tokenized[i] = prompts_tokenized[i][:token_length]
        texts_tokenized[i] = texts_tokenized[i][:token_length]

    if debug:
        print(f"Example Prompt after Truncation: {prompts_tokenized[0]},\n Length: {len(prompts_tokenized[0])}")
        print(f"Example Text after Truncation: {texts_tokenized[0]},\n Length: {len(texts_tokenized[0])}\n")
    print(f"File: {arrow_file.rstrip('/').split('/')[-1]}, truncation complete.")

    # Flatten and convert tokenized data into a single NumPy array
    prompts_flattened = np.concatenate([np.array(ids, dtype=np.uint16) for ids in prompts_tokenized])
    texts_flattened = np.concatenate([np.array(ids, dtype=np.uint16) for ids in texts_tokenized])

    # Prepare output file path
    prompts_file = Path(memmap_dir) / f"prompts/{Path(arrow_file).stem}.npy"
    texts_file = Path(memmap_dir) / f"texts/{Path(arrow_file).stem}.npy"

    print(f"**** {arrow_file.rstrip('/').split('/')[-1]}: Prompts Length: {len(prompts_flattened)}, Texts Length: {len(texts_flattened)}")

    # Save as a memory-mapped .npy file
    np.save(prompts_file, prompts_flattened)
    np.save(texts_file, texts_flattened)
    print(f"Converted {arrow_file} to {prompts_file}, {texts_file}.")
    print(f"File: {arrow_file.rstrip('/').split('/')[-1]} Prompts Len: {prompts_len}, Texts Len: {texts_len}")
    print("------------------------------------\n")


def memmap_write(hf_data_split):
    # Input directory containing .arrow files
    base_path = "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/"
    debug = False
    arrow_dir = os.path.join(base_path, hf_data_split, "data/train")

    # Output directory for memory-mapped .npy files
    memmap_dir = os.path.join(base_path, hf_data_split, "memmap")
    os.makedirs(memmap_dir, exist_ok=True)
    os.makedirs(memmap_dir + "/prompts", exist_ok=True)
    os.makedirs(memmap_dir + "/texts", exist_ok=True)

    # Path to the tokenizer
    tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B")

    if debug:
        arrow_file = os.path.join(arrow_dir, "data-00001-of-00118.arrow")
        print("Starting")
        convert_arrow_to_memmap(arrow_file, memmap_dir, tokenizer)
    # List all .arrow files in the input directory
    else:
        arrow_files = [str(file) for file in Path(arrow_dir).glob("*.arrow")]
        # Process files concurrently using ThreadPoolExecutor
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(convert_arrow_to_memmap, arrow_file, memmap_dir, tokenizer): arrow_file
                for arrow_file in arrow_files
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file {futures[future]}: {e}")

    print("All files have been converted.")


if __name__ == "__main__":
    memmap_write("web_samples_v2")