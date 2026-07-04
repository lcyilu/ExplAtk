import os
DS_PATH = '{HOME_PATH}/VulDS/'
DS_LIST = ['BigVul', 'Reveal', 'Devign']
CODE_TRANS_TXL_SH = '{project_root}/preprocess/CLONEGEN/CodeTransformationTest/RM'

def trans_vul_filter(ds_name):
    ds_dir = os.path.join(DS_PATH, ds_name)
    if ds_name == 'BigVul':
        src_dir = os.path.join(ds_dir, 'all-src')
    else:
        src_dir = os.path.join(ds_dir, 'src')
    vul_dir = os.path.join(src_dir, 'vul/')
    vul_files = [os.path.abspath(os.path.join(vul_dir, f)) for f in os.listdir(vul_dir) if f.endswith(('.c','.cpp'))]
    with open(os.path.join(CODE_TRANS_TXL_SH, f"failed_files_{ds_name.lower()}.txt")) as fp:
        path_list = fp.readlines()
        for file_path in path_list:
            file_path = file_path.strip()
            if file_path in vul_files:
                vul_files.remove(file_path)
    print(f"After filtering out samples with syntax errors, {len(vul_files)} samples remain")
    with open(os.path.join(src_dir, 'vul_filtered.txt'), 'w') as f:
        for file_path in vul_files:
            f.write(file_path + '\n')

def trans_novul_filter(ds_name):
    ds_dir = os.path.join(DS_PATH, ds_name)
    if ds_name == 'BigVul':
        src_dir = os.path.join(ds_dir, 'all-src')
    else:
        src_dir = os.path.join(ds_dir, 'src')
    novul_dir = os.path.join(src_dir, 'novul/')
    novul_files = [os.path.abspath(os.path.join(novul_dir, f)) for f in os.listdir(novul_dir) if f.endswith(('.c','.cpp'))]
    with open(os.path.join(CODE_TRANS_TXL_SH, f"failed_files_{ds_name.lower()}_novul.txt")) as fp:
        path_list = fp.readlines()
        for file_path in path_list:
            file_path = file_path.strip()
            if file_path in novul_files:
                novul_files.remove(file_path)
    print(f"After filtering out samples with syntax errors, {len(novul_files)} samples remain")
    with open(os.path.join(src_dir, 'novul_filtered.txt'), 'w') as f:
        for file_path in novul_files:
            f.write(file_path + '\n')

def enhance_data(ds_name):
    ds_dir = os.path.join(DS_PATH, ds_name)
    if ds_name == 'BigVul':
        src_dir = os.path.join(ds_dir, 'all-src')
    else:
        src_dir = os.path.join(ds_dir, 'src')
    trans_vul_dir = os.path.join(src_dir, 'trans_vul/')
    filtered_files = os.path.join(src_dir, 'vul_filtered.txt')
    if not os.path.exists(trans_vul_dir):
        os.mkdir(trans_vul_dir)

    pwd = os.getcwd()

    os.chdir(CODE_TRANS_TXL_SH)
    os.environ['input'] = str(filtered_files)
    os.environ['out'] = str(trans_vul_dir)
    os.system(f'./mutation.sh $input $out 1')

    os.chdir(pwd)

def enhance_novul_data(ds_name):
    ds_dir = os.path.join(DS_PATH, ds_name)
    if ds_name == 'BigVul':
        src_dir = os.path.join(ds_dir, 'all-src')
    else:
        src_dir = os.path.join(ds_dir, 'src')
    trans_vul_dir = os.path.join(src_dir, 'trans_novul/')
    filtered_files = os.path.join(src_dir, 'novul_filtered.txt')
    if not os.path.exists(trans_vul_dir):
        os.mkdir(trans_vul_dir)

    pwd = os.getcwd()

    os.chdir(CODE_TRANS_TXL_SH)
    os.environ['input'] = str(filtered_files)
    os.environ['out'] = str(trans_vul_dir)
    os.system(f'./mutation.sh $input $out 1')

    os.chdir(pwd)
    

def main():
    # for ds_name in DS_LIST:
    #     # Run syntax checking for the corresponding dataset: "./batch_txl.sh {vul_dir} {txl_opt}"
    #     # After generating the list of syntax-error samples: CODE_TRANS_TXL_SH + "failed_files_{ds_name}.txt"
    #     trans_vul_filter(ds_name)
    #     # Generate augmented samples
    #     enhance_data(ds_name)

    trans_novul_filter('Devign')
    enhance_novul_data('Devign')

if __name__ == "__main__":
    main()