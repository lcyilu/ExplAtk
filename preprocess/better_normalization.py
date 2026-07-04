import json
import os
import re
from tqdm import tqdm
from tree_sitter import Language, Parser
from common.utils.keywords import __key_words__, __macros__, __special_ids__, __builtin__funcs__, __other__keywords__

dirpath = '{project_root}/my-languages.so' ##

# Load the language library
C_LANGUAGE = Language(dirpath, 'c')
CPP_LANGUAGE = Language(dirpath, 'cpp')

def rename_identifiers_in_code(code: bytes, language='cpp') -> str:
    """
    Rename identifiers in the given C/C++ code and return the modified code.

    Parameters:
    - code: The source code as a byte string.
    - language: 'c' for C code, 'cpp' for C++ code.

    Returns:
    - The modified code with renamed identifiers.
    """
    # Initialize the parser
    if language == 'c':
        parser = Parser()
        parser.set_language(C_LANGUAGE)
    elif language == 'cpp': 
        parser = Parser()
        parser.set_language(CPP_LANGUAGE)
    else:
        raise ValueError("Unsupported language. Use 'c' or 'cpp'.")
    
    # Parse the code
    tree = parser.parse(code)
    root_node = tree.root_node

    # Dictionary to store old and new names
    renaming_map = {}
    var_counter = 1
    func_counter = 1

    identifiers_info = []

    # Compile the regular expression to match identifier and all *_identifier
    identifier_candidates = [
        'identifier',
        'field_identifier'
    ]

    func_id_parents = [
        'function_declarator', 
        'call_expression',
        'field_initializer'
    ]

    def rename_identifiers(node):
        nonlocal var_counter, func_counter
        for child in node.children:
            if child.type in identifier_candidates:
                # Get the old name
                old_name = code[child.start_byte:child.end_byte].decode('utf-8')
                
                # Determine the context of the identifier
                if child.parent.type in func_id_parents:
                    # It's a function name
                    if len({old_name}.difference(__builtin__funcs__)) != 0 and len({old_name}.difference(__other__keywords__)) != 0:
                        if old_name not in renaming_map:
                            new_name = f'FUNC{func_counter}'
                            func_counter += 1
                            renaming_map[old_name] = new_name
                            identifiers_info.append({
                                'name': old_name,
                                'type': 'function',
                                'start_point': child.start_point,
                                'end_point': child.end_point
                            })
                        else:
                            identifiers_info.append({
                                'name': old_name,
                                'type': 'function',
                                'start_point': child.start_point,
                                'end_point': child.end_point
                            })
                else:
                    # It's a variable name
                    if len({old_name}.difference(__key_words__)) != 0 and len({old_name}.difference(__special_ids__)) != 0 and len({old_name}.difference(__macros__)) != 0 and len({old_name}.difference(__other__keywords__)) != 0:
                        if old_name not in renaming_map:
                            new_name = f'VAR{var_counter}'
                            var_counter += 1
                            renaming_map[old_name] = new_name
                            identifiers_info.append({
                                'name': old_name,
                                'type': 'variable',
                                'start_point': child.start_point,
                                'end_point': child.end_point
                            })
                        else:
                            identifiers_info.append({
                                'name': old_name,
                                'type': 'variable',
                                'start_point': child.start_point,
                                'end_point': child.end_point
                            })

            # Recursively rename in child nodes
            rename_identifiers(child)

    # Start renaming from the root node
    rename_identifiers(root_node)

    # Replace old names with new names in the code
    # Sort by position and replace from back to front (to avoid position offset issues)
    identifiers_info.sort(key=lambda x: (x['start_point'][0], x['start_point'][1]), reverse=True)
    
    # Perform the replacement
    code_lines = code.decode('utf-8').split('\n')
    
    for identifier in identifiers_info:
        if identifier['name'] in renaming_map:
            new_name = renaming_map[identifier['name']]
            start_row, start_col = identifier['start_point']
            end_row, end_col = identifier['end_point']
            
            # Replace the text at the specified position
            if start_row < len(code_lines):
                line = code_lines[start_row]
                code_lines[start_row] = line[:start_col] + new_name + line[end_col:]
    
    normalized_code = '\n'.join(code_lines)
    return normalized_code

# def remove_comments(code: str) -> str:
#     """
#     Remove comments from the given C/C++ code.

#     Parameters:
#     - code: The source code as a byte string.  
#     """ 
#     linefeed='\n'
#     annotations = re.findall('(?<!:)\\/\\/.*|\\/\\*(?:\\s|.)*?\\*\\/', code)
#     #print(annotations)
#     for annotation in annotations:
#         lf_num = annotation.count('\n')
#         if lf_num == 0:
#             code = code.replace(annotation,'')
#             continue
#         code = code.replace(annotation,lf_num*linefeed)

#     return code

def remove_comments(code: str) -> str:
    """
    Remove comments from the given C/C++ code efficiently.
    """
    # Optimized regex:
    # 1. (?<!:)\/\/.*     Matches // line comments, and not preceded by : (to avoid matching http://)
    # 2. \/\*[\s\S]*?\*\/ Matches /* ... */ block comments. [\s\S] is an efficient way to match any character (including newlines)
    pattern = r'(?<!:)\/\/.*|\/\*[\s\S]*?\*\/'
    
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'): # Double-check that it is a comment (although the regex already guarantees it)
            # If it is a block comment (/* ... */), we keep the newline characters inside it
            # so that the line numbers of the code remain unchanged, which helps subsequent analysis.
            if s.startswith('/*'):
                return '\n' * s.count('\n')
            # If it is a line comment (// ...), replace it directly with an empty string
            else:
                return ''
        return s

    # Use re.sub to scan and replace in one pass, which is orders of magnitude more efficient than findall + a replace loop
    return re.sub(pattern, replacer, code)

def filter_files(file_list, valid_stems):
    """
    Filter file_list (list of full paths) based on valid_stems (list of file names)
    """
    # 1. Convert the whitelist to a set, improving lookup speed from O(n) to O(1)
    valid_set = set(valid_stems)
    
    # 2. Filter using a list comprehension
    # os.path.basename(f): get the file name "abc.json"
    # os.path.splitext(...)[0]: get "abc" (stem)
    filtered_list = [
        f for f in file_list 
        if os.path.splitext(os.path.basename(f))[0] in valid_set
    ]
    return filtered_list

def normalize_one_file(filepath, language='cpp'):
    with open(filepath, "r") as file:
        code = file.read()
    code_no_comments = remove_comments(code).encode('utf-8')
    normalized_code = rename_identifiers_in_code(code_no_comments, language)
    return normalized_code

def normalize(dirpath, ds_list_path, output_dir=None, language='cpp'):
    from collections import defaultdict
    if isinstance(ds_list_path, str):
        with open(ds_list_path,'r') as f: ##
            all_stems = json.load(f)
    else:
        merged_data = defaultdict(set)
        all_stems = {}
        for dl in ds_list_path:
            with open(dl, 'r') as f:
                data = json.load(f)
            # Iterate over every key of the current file
            for key, value_list in data.items():
                # Make sure value_list is a list, to prevent errors caused by malformed data
                if isinstance(value_list, list):
                    # The update method adds each element of the list into the set and deduplicates automatically
                    merged_data[key].update(value_list)
                else:
                    # If it is not a list (e.g. just a single string), this line handles it for compatibility
                    merged_data[key].add(value_list)
        
        for key, value_set in merged_data.items():
            all_stems[key] = list(value_set)

    
    for subname in os.listdir(dirpath):
        sub_path = os.path.join(dirpath, subname)
        if subname in ["samples", "vul_patch"]:
            continue
        if os.path.isdir(sub_path):
            output_subdir = os.path.join(output_dir, subname)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            src_list = [file for file in os.listdir(sub_path) if file.endswith('.c') or file.endswith('.cpp') or file.endswith('.h') or file.endswith('.hpp')]
            stem_key = str(subname) + '_files'
            src_list = filter_files(src_list, all_stems[stem_key])
            for src in tqdm(src_list, desc=f"Normalizing code files in {sub_path}"):
                try:
                    filepath = os.path.join(sub_path, src)
                    normalized_code = normalize_one_file(filepath, language)
                    if output_dir:
                        output_path = os.path.join(output_subdir, src)
                        with open(output_path, "w") as out_file:
                            out_file.write(normalized_code)
                    else:
                        print(f"Processing file:\n {src}")
                        print(f"Normalized code:\n")
                        print(normalized_code)
                except Exception as e:
                    print(f"Failed to normalize {src}!")

def test():
    # Example for testing
    code = b'''
    int main() {
        // test for comments removal
        // test for comments removal
        struct Person {
            string name; // test for comments removal
            int age;
            float height; // test for comments removal

            Person(string n, int a, float h) : name(n), age(a), height(h) {}
        };
        
        Person person1("Charlie", 28, 5.9);

        // test for comments removal
        cout << "Person: " << person1.name << ", " << person1.age << " years old, " << person1.height << " ft" << endl;

        return 0;
    }
    '''

    # code = b'''
    # bool grubfs_free(GrubFS *gf){
    #     if (gf){
    #         if (gf->file && gf->file->device)
    #             free(gf->file->device->disk);
    #         free(gf->file);
    #         free(gf);
    #     }
    #     return false;
    # }
    # '''

    code_no_comments = remove_comments(code.decode('utf-8')).encode('utf-8')
    print("Code without comments:\n", code_no_comments.decode('utf-8'))

    normalized_code = rename_identifiers_in_code(code_no_comments, language='cpp')
    print("Normalized Code:\n", normalized_code)

    # filepath = '{HOME_PATH}/VulDS/BigVul/all-src/vul/1_CVE-2016-9601_ghostscript_CWE-119_e698d5c11d27212aa1098bc5b1673a3378563092_1.c'
    # with open(filepath, "r") as file:
    #     code = file.read()
    # print("Original Code:\n", code)
    # normalized_code = normalize_one_file(filepath,'cpp')
    # print(normalized_code)

if __name__ == "__main__":
    # test()
    dirpath = "{HOME_PATH}/VulDS/BigVul/all-src"
    # ds_list_path = "{HOME_PATH}/VulDS/BigVul/bigvul_ds.json"
    ds_list_path = ["{HOME_PATH}/VulDS/BigVul/bigvul_ds.json",
                    "{HOME_PATH}/VulDS/BigVul/cwe119_ds.json"]
    output_dir="{HOME_PATH}/VulDS/BigVul/normal-src"
    normalize(dirpath=dirpath,ds_list_path=ds_list_path, output_dir=output_dir)
    # src_dir = '{HOME_PATH}/VulDS/BigVul/all-src/'
    # out_dir = '{HOME_PATH}/VulDS/BigVul/normal-src/'
    # normalize(src_dir, language='cpp', output_dir=out_dir)