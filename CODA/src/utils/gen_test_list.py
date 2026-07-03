import os

def generate_unique_source_paths(input_txt_files, output_txt_file):
    unique_paths = set()

    print(f"Starting to process {len(input_txt_files)} input files...")

    for txt_file in input_txt_files:
        if not os.path.exists(txt_file):
            print(f"Warning: file does not exist -> {txt_file}")
            continue
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: 
                    continue
                
                # Conversion logic:
                # 1. Replace directory: /ori-embedding/ -> /ori-src/
                # 2. Replace extension: .json -> .c
                # Note: this assumes the path structure is fixed, so a simple replace is sufficient.
                
                # Check whether the line matches the expected format to avoid processing invalid lines.
                if '/ori-embedding/' in line and '/BigVul/' in line and line.endswith('.json'):
                    src_path = line.replace('/ori-embedding/', '/all-src/')
                    src_path = src_path[:-5] + '.c' # Remove .json (5 characters) and append .c.
                    
                    unique_paths.add(src_path)
                elif '/ori-embedding/' in line and line.endswith('.json'):
                    src_path = line.replace('/ori-embedding/', '/src/')
                    src_path = src_path[:-5] + '.c' # Remove .json (5 characters) and append .c.
                    
                    unique_paths.add(src_path)
                else:
                    # If the format does not match, choose to ignore it or print a warning.
                    # print(f"Skipping line with unmatched format: {line}")
                    pass

    print(f"Collection complete. Found {len(unique_paths)} unique source-code paths.")

    # Sort before writing for easier review.
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for path in sorted(list(unique_paths)):
            f.write(path + '\n')
    
    print(f"Results saved to -> {output_txt_file}")

# --- Configuration section ---
if __name__ == "__main__":
    ## Test-set file list
    input_files = [
        "{HOME_PATH}/VulDS/BigVul/ori-embedding/bigvul_test.txt",
        "{HOME_PATH}/VulDS/BigVul/ori-embedding/cwe119_test.txt",
        "{HOME_PATH}/VulDS/Devign/ori-embedding/devign_test.txt",
        "{HOME_PATH}/VulDS/Reveal/ori-embedding/reveal_test.txt",
    ]
    
    ## Output file path
    output_file = "{HOME_PATH}/vul_robustness/DIP/data/raw/all_unique_source_paths.txt"
    
    generate_unique_source_paths(input_files, output_file)