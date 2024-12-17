import multiprocessing
multiprocessing.set_start_method('spawn')  # Set at the very top

import argparse
import json
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import torch
import os

# Load the model and tokenizer once (shared across workers)
olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-Instruct")
tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B-Instruct")

user_prompt = """For the following text, rephrase it in a clear way that contains the same information in the original text.

Text: {text}"""
args = {
    "max_new_tokens": 500,
    "top_k": 50,
    "top_p": 0.95
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process JSON files')
    parser.add_argument('--input', '-i', required=True)  # Input file
    parser.add_argument('--output', '-o', required=True)  # Output file
    parser.add_argument('--sample_size', '-s', type=int)  # Number of samples to process (optional)
    return parser.parse_args()

def generate_rephrasing(text, model, tokenizer, device):
    """Generate a rephrased version of the input text."""
    chat = [
        {"role": "user", "content": user_prompt.format(text=text)}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    response = model.generate(input_ids=inputs, **args)
    rephrased_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    return rephrased_text.split("<|assistant|>")[-1].strip()

def worker(queue, output_file, lock, model, tokenizer, device):
    """Worker process function to process data chunks."""
    model.to(device)  # Move model to the assigned GPU
    while not queue.empty():
        try:
            # Dynamically fetch data from the queue
            data_dict = queue.get_nowait()
            rephrased_text = generate_rephrasing(data_dict["text"], model, tokenizer, device)
            data_dict["rephrased_text"] = rephrased_text

            # Write results to the output file
            with lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(data_dict) + '\n')
        except Exception as e:
            print(f"Error processing text: {e}")

def main():
    print("Starting...")
    args = parse_arguments()

    # Load input data
    with open(args.input, 'r') as f:
        if args.sample_size:
            data = [json.loads(next(f)) for _ in range(args.sample_size)]
        else:
            data = [json.loads(line) for line in f]

    # Shared multiprocessing queue for dynamic task allocation
    queue = multiprocessing.Queue()
    for item in data:
        queue.put(item)

    lock = multiprocessing.Lock()  # For synchronized file writes

    # Determine the number of GPUs available (via Slurm)
    num_processes = int(os.environ.get("SLURM_GPUS_PER_TASK", 1))
    print(f"Number of processes (GPUs): {num_processes}")

    # Start worker processes
    processes = []
    for i in range(num_processes):
        # Pin each process to a specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        device = torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
        p = multiprocessing.Process(
            target=worker,
            args=(queue, args.output, lock, olmo, tokenizer, device)
        )
        p.start()
        processes.append(p)

    # Ensure all processes finish
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
