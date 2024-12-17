import pyarrow as pa

def inspect_arrow_stream_file(file_path):
    """
    Inspect the structure and contents of an Arrow stream file.

    Args:
        file_path (str): Path to the Arrow stream file.

    Returns:
        None
    """
    try:
        # Open the Arrow stream file in binary mode
        with open(file_path, "rb") as f:
            reader = pa.RecordBatchStreamReader(f)
            
            # Print the schema (column names and types)
            print("Schema:")
            print(reader.schema)
            
            # Read all data into a table
            table = reader.read_all()
            
            # Print number of rows and columns
            print("\nNumber of Rows:", table.num_rows)
            print("Number of Columns:", len(table.schema))
            
            # Show metadata (if any)
            if reader.schema.metadata:
                print("\nMetadata:")
                for key, value in reader.schema.metadata.items():
                    print(f"{key.decode('utf-8')}: {value.decode('utf-8')}")
            
            # Preview the first 5 rows
            print("\nPreview (First 5 Rows):")
            print(table.to_pandas().head())
            # print the prompt column in full for the first 10 rows, as a for loop becasue it gets truncated if just using .head()
            # print("\nPrompt column (First 10 Rows):")
            # for i in range(10):
            #     print(table.to_pandas()['prompt'][i])
            #     print("\n\n==========================================\n\n")
            # for the first prompt row, print out each character ona newline
            # actually make a listo f chars and print the list
            print("\nPrompt column (First Row):")
            prompt = table.to_pandas()['prompt'][200]
            prompt_chars = list(prompt)
            print(prompt_chars)
    except Exception as e:
        print(f"Failed to inspect Arrow stream file: {e}")

# Example usage
arrow_stream_file_path = "/n/netscratch/sham_lab/Everyone/bham/mid_olmo/web_samples_v1/data/train/data-00001-of-00139.arrow"
inspect_arrow_stream_file(arrow_stream_file_path)
