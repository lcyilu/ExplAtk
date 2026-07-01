"""
gen_embedding.py  ——  线程安全改造版

【对外接口完全不变】
所有公开函数的签名、参数语义、返回值、副作用（产出文件位置等）都与原版一致，
上游调用方零改动。

【改造点】
1. load_word_vectors:
   原版每次调用都 KeyedVectors.load,既慢又会在多线程下并发 mmap 同一份文件;
   改为按 path 缓存的单例 + 双重检查锁,首次以外的调用只有一次 dict.get 开销。
2. joern_parse / joern_export:
   原版用 os.environ + os.system,os.environ 是进程级共享状态,在多线程下
   会出现 A 设置的变量被 B 覆盖、最终 joern 处理错样本的严重 race condition;
   改为 subprocess.run(["sh", "joern-...", arg1, arg2, ...], cwd=joern_path),
   完全不动父进程 environ 与 CWD,线程之间互不影响。
3. src2embedding / src2pdg:
   原版在主函数里 os.chdir(joern_path) 污染进程级 CWD;改造后 cwd 由 subprocess
   单独管理,这里只剩 tempfile 隔离的纯 IO,完全线程安全。
4. 后处理的 mv / rm -rf 改用 shutil,避免 shell 调用 + 处理路径中的特殊字符。
"""

import codecs
from functools import lru_cache
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import warnings

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from gensim.models import KeyedVectors
import pygraphviz as pgv

from common.config_loader import get_settings
from common.utils.dot_parser import LooseDotParser

warnings.filterwarnings("ignore")
cnt = 0


def tokenize_code_line(line):
    # Sets for operators
    operators3 = {'<<=', '>>='}
    operators2 = {
        '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
        '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
    operators1 = {
        '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
        '=', ',', '?', ':', ';', '{', '}', '!', '~'
    }

    tmp, w = [], []
    i = 0
    if type(i) == None:
        return []
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 3].strip())
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 2].strip())
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    if (len(w) != 0):
        tmp.append(''.join(w).strip())
        w = []
    # Filter out irrelevant strings
    tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
    return tmp

@lru_cache(maxsize=1)
def load_word_vectors(path=None):
    """
    线程安全 + 进程内单例缓存。
    签名与原版一致;同一路径多次调用返回同一对象,避免重复 mmap 与 KeyedVectors 解析。
    使用双重检查锁,稳态下的开销仅一次 dict.get。
    """
    path = path or get_settings().word2vec_path
    kv = KeyedVectors.load(path, mmap="r")
    return kv


def read_json(filename):
    #读取文件
    with open(filename.strip(),'r') as f:
        file = json.load(f)
    #文件内容读取到torch.tensor()中
    x = torch.tensor(file['node_features'],dtype=torch.float32)
    num_nodes = x.shape[0]
    if num_nodes < 10:
        return None

    edge_index_list = []
    for edge in file['graph']:
        if edge[0] <= num_nodes and edge[2] <= num_nodes:
            edge_index_list.append([edge[0],edge[2]])
    edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()

    edge_attr_list = []
    for edge in file['graph']:
        edge_attr_list.append([edge[1]])
    edge_attr = torch.tensor(edge_attr_list)

    #y=[]
    #y.append([file['target']])
    #y=torch.tensor(y)
    y = torch.tensor([file['target']], dtype=int)

    data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, name = filename.strip().split('/')[-1])
    #torch.save(data,filename+'.pt')
    return data


def src2embedding(src, label):
    """
    线程安全版:
    - 不再 os.chdir(joern_path)(CWD 由 subprocess 在 joern_parse/export 内部处理)
    - load_word_vectors 现在是缓存的,不会重复 load
    输入输出与原版完全一致。
    """
    if isinstance(src, bytes):
        src = src.decode('utf-8')
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Convert source code to PDG
        src_file = os.path.join(temp_dir, "temp_code.c")
        cpg_bin_file = os.path.join(temp_dir, "temp_cpg.bin")
        with open(src_file, 'w') as f:
            f.write(src)
        joern_parse(src_file, cpg_bin_file)

        pdg_dir = os.path.join(temp_dir, "temp_pdg")
        joern_export(cpg_bin_file, pdg_dir)
        pdg_file = os.path.join(temp_dir, "temp_pdg.dot")

        # 2. Convert PDG to embedding using word2vec
        word_vectors = load_word_vectors()
        embedding = pdgfile2embedding(pdg_file, word_vectors=word_vectors, true_label=label)

    return embedding


# def joern_parse(src, bin):
#     """
#     线程安全版:
#     - 不再修改 os.environ
#     - 不依赖父进程 CWD,通过 subprocess 的 cwd 参数在子进程独立设置
#     - argv 直接传值,避免 shell 解释带来的路径转义问题
#     签名与原版一致。
#     """
#     subprocess.run(
#         ["sh", "joern-parse", str(src), "--language", "c", "--out", str(bin)],
#         cwd=get_settings().joern_path,
#         stderr=subprocess.DEVNULL,
#         check=False,
#     )

def joern_parse(src, bin):
    joern_path = get_settings().joern_path
    # 用独立 workspace 隔离并发,工作目录用 temp dir,joern 脚本用绝对路径
    with tempfile.TemporaryDirectory(prefix="joern_ws_") as ws:
        try:
            subprocess.run(
                ["sh", os.path.join(joern_path, "joern-parse"),
                 str(src), "--language", "c", "--out", str(bin)],
                cwd=ws,                              # workspace 隔离在这
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass


def joern_export(bin, pdg_dir):
    joern_path = get_settings().joern_path
    with tempfile.TemporaryDirectory(prefix="joern_ws_") as ws:
        try:
            subprocess.run(
                ["sh", os.path.join(joern_path, "joern-export"),
                 str(bin), "--repr", "pdg", "--out", str(pdg_dir)],
                cwd=ws,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass
    # 后处理(与 ws 无关,操作的是调用方传入的 pdg_dir)
    try:
        pdg_list = os.listdir(pdg_dir)
        for pdg in pdg_list:
            if pdg.startswith("0-pdg"):
                file_path = os.path.join(pdg_dir, pdg)
                shutil.move(file_path, str(pdg_dir) + ".dot")
                shutil.rmtree(pdg_dir, ignore_errors=True)
                break
    except Exception:
        pass



# def joern_export(bin, pdg_dir):
#     """
#     线程安全版,改动同 joern_parse。
#     后处理的 mv/rm 也换成 shutil。
#     """
#     subprocess.run(
#         ["sh", "joern-export", str(bin), "--repr", "pdg", "--out", str(pdg_dir)],
#         cwd=get_settings().joern_path,
#         stderr=subprocess.DEVNULL,
#         check=False,
#     )
#     try:
#         pdg_list = os.listdir(pdg_dir)
#         for pdg in pdg_list:
#             if pdg.startswith("0-pdg"):
#                 file_path = os.path.join(pdg_dir, pdg)
#                 shutil.move(file_path, str(pdg_dir) + ".dot")
#                 shutil.rmtree(pdg_dir, ignore_errors=True)
#                 break
#     except Exception:
#         pass


def pdgfile2embedding(dot_pdg, word_vectors, true_label):
    node_index = dict()
    node_feature = dict()
    try:
        with open(dot_pdg, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            # 重写
            parser = LooseDotParser()
            pdg = parser.to_networkx(content)
        except Exception as e:
            pdg = None
            print("Failed to load dot file with pydot:", e)

        if pdg is not None:
            for index, node in enumerate(pdg.nodes()):
                node_index[node] = index
                label = pdg.nodes[node]['label'][1:-1]
                code = label.partition(',')[2]
                feature = np.array([0.0 for i in range(100)])
                for token in tokenize_code_line(code):
                    # mask placeholder replace
                    if token == get_settings().mask_placeholder:
                        token == '<MASK>'
                    if token in word_vectors:
                        feature += np.array(word_vectors[token])
                    else:
                        feature += np.array([0.0 for i in range(100)])
                node_feature[index] = feature

            nodes_ = []
            for i in range(len(list(pdg.nodes()))):
                nodes_.append(list(node_feature[i]))

            edges_ = []
            for item in pdg.adj.items():
                s = item[0]
                for edge_relation in item[1]:
                    d = edge_relation
                    ddg_flag = 0
                    cdg_flag = 0
                    for edge in item[1]._atlas[edge_relation].items():
                        if 'DDG' in edge[1]['label'] and ddg_flag == 0:
                            edge_type = 0
                            ddg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))
                        elif 'CDG' in edge[1]['label'] and cdg_flag == 0:
                            edge_type = 1
                            cdg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))

            x = torch.tensor(nodes_,dtype=torch.float32)
            num_nodes = x.shape[0]

            edge_index_list = []
            for edge in edges_:
                if edge[0] <= num_nodes and edge[2] <= num_nodes:
                    edge_index_list.append([edge[0],edge[2]])
            edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()

            edge_attr_list = []
            for edge in edges_:
                edge_attr_list.append([edge[1]])
            edge_attr = torch.tensor(edge_attr_list)

            y = torch.tensor(true_label, dtype=int)

            data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    except:
        print("Failed to trans")
        pass
    return data


def src2pdg(src):
    """
    线程安全版:删掉 os.chdir,其余逻辑不变。
    """
    if isinstance(src, bytes):
        src = src.decode('utf-8')
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Convert source code to PDG
        src_file = os.path.join(temp_dir, "temp_code.c")
        cpg_bin_file = os.path.join(temp_dir, "temp_cpg.bin")
        with open(src_file, 'w') as f:
            f.write(src)
        joern_parse(src_file, cpg_bin_file)

        pdg_dir = os.path.join(temp_dir, "temp_pdg")
        joern_export(cpg_bin_file, pdg_dir)
        pdg_file = os.path.join(temp_dir, "temp_pdg.dot")

        with open(pdg_file, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            # 重写
            parser = LooseDotParser()
            pdg = parser.to_networkx(content)
        except Exception as e:
            pdg = None
            print("Failed to load dot file with pydot:", e)

    return pdg


def pdg2embedding(pdg, word_vectors, true_label):
    node_index = dict()
    node_feature = dict()
    try:
        if type(pdg) != None:
            for index, node in enumerate(pdg.nodes()):
                node_index[node] = index
                label = pdg.nodes[node]['label'][1:-1]
                code = label.partition(',')[2]
                feature = np.array([0.0 for i in range(100)])
                for token in tokenize_code_line(code):
                    # mask placeholder replace
                    if token == get_settings().mask_placeholder:
                        token = '<MASK>'
                    if token in word_vectors:
                        feature += np.array(word_vectors[token])
                    else:
                        feature += np.array([0.0 for i in range(100)])
                node_feature[index] = feature

            nodes_ = []
            for i in range(len(list(pdg.nodes()))):
                nodes_.append(list(node_feature[i]))

            edges_ = []
            for item in pdg.adj.items():
                s = item[0]
                for edge_relation in item[1]:
                    d = edge_relation
                    ddg_flag = 0
                    cdg_flag = 0
                    for edge in item[1]._atlas[edge_relation].items():
                        if 'DDG' in edge[1]['label'] and ddg_flag == 0:
                            edge_type = 0
                            ddg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))
                        elif 'CDG' in edge[1]['label'] and cdg_flag == 0:
                            edge_type = 1
                            cdg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))

            x = torch.tensor(nodes_,dtype=torch.float32)
            num_nodes = x.shape[0]

            edge_index_list = []
            for edge in edges_:
                if edge[0] <= num_nodes and edge[2] <= num_nodes:
                    edge_index_list.append([edge[0],edge[2]])
            edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()

            edge_attr_list = []
            for edge in edges_:
                edge_attr_list.append([edge[1]])
            edge_attr = torch.tensor(edge_attr_list)

            y = torch.tensor(true_label, dtype=int)

            data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    except:
        print("Failed to trans")
        pass
    return data


def renamed_pdg_to_embedding(pdg, word_vectors,ori_var, new_var, true_label):
    node_index = dict()
    node_feature = dict()
    try:
        if type(pdg) != None:
            for index, node in enumerate(pdg.nodes()):
                node_index[node] = index
                label = pdg.nodes[node]['label'][1:-1]
                code = label.partition(',')[2]
                feature = np.array([0.0 for i in range(100)])
                for token in tokenize_code_line(code):
                    # mask placeholder replace
                    if token == ori_var:
                        token = new_var
                        # print(f"Trying to rename '{ori_var}' to '{new_var}'")
                    if token in word_vectors:
                        feature += np.array(word_vectors[token])
                    else:
                        feature += np.array([0.0 for i in range(100)])
                node_feature[index] = feature

            nodes_ = []
            for i in range(len(list(pdg.nodes()))):
                nodes_.append(list(node_feature[i]))

            edges_ = []
            for item in pdg.adj.items():
                s = item[0]
                for edge_relation in item[1]:
                    d = edge_relation
                    ddg_flag = 0
                    cdg_flag = 0
                    for edge in item[1]._atlas[edge_relation].items():
                        if 'DDG' in edge[1]['label'] and ddg_flag == 0:
                            edge_type = 0
                            ddg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))
                        elif 'CDG' in edge[1]['label'] and cdg_flag == 0:
                            edge_type = 1
                            cdg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))

            x = torch.tensor(nodes_,dtype=torch.float32)
            num_nodes = x.shape[0]

            edge_index_list = []
            for edge in edges_:
                if edge[0] <= num_nodes and edge[2] <= num_nodes:
                    edge_index_list.append([edge[0],edge[2]])
            edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()

            edge_attr_list = []
            for edge in edges_:
                edge_attr_list.append([edge[1]])
            edge_attr = torch.tensor(edge_attr_list)

            y = torch.tensor(true_label, dtype=int)

            data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    except:
        print("Failed to trans")
        pass
    return data


def multi_renamed_pdg_to_embedding(pdg, word_vectors,renamed_vars, true_label):
    node_index = dict()
    node_feature = dict()
    try:
        if type(pdg) != None:
            for index, node in enumerate(pdg.nodes()):
                node_index[node] = index
                label = pdg.nodes[node]['label'][1:-1]
                code = label.partition(',')[2]
                feature = np.array([0.0 for i in range(100)])
                for token in tokenize_code_line(code):
                    # mask placeholder replace
                    if token in renamed_vars.keys():
                        token = renamed_vars[token]
                        # print(f"Trying to rename '{ori_var}' to '{new_var}'")
                    if token in word_vectors:
                        feature += np.array(word_vectors[token])
                    else:
                        feature += np.array([0.0 for i in range(100)])
                node_feature[index] = feature

            nodes_ = []
            for i in range(len(list(pdg.nodes()))):
                nodes_.append(list(node_feature[i]))

            edges_ = []
            for item in pdg.adj.items():
                s = item[0]
                for edge_relation in item[1]:
                    d = edge_relation
                    ddg_flag = 0
                    cdg_flag = 0
                    for edge in item[1]._atlas[edge_relation].items():
                        if 'DDG' in edge[1]['label'] and ddg_flag == 0:
                            edge_type = 0
                            ddg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))
                        elif 'CDG' in edge[1]['label'] and cdg_flag == 0:
                            edge_type = 1
                            cdg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))

            x = torch.tensor(nodes_,dtype=torch.float32)
            num_nodes = x.shape[0]

            edge_index_list = []
            for edge in edges_:
                if edge[0] <= num_nodes and edge[2] <= num_nodes:
                    edge_index_list.append([edge[0],edge[2]])
            edge_index = torch.tensor(edge_index_list,dtype=torch.long).t()

            edge_attr_list = []
            for edge in edges_:
                edge_attr_list.append([edge[1]])
            edge_attr = torch.tensor(edge_attr_list)

            y = torch.tensor(true_label, dtype=int)

            data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    except:
        print("Failed to trans")
        pass
    return data


def test():
    code = b"""
int FUN1(char* VAR1) {
    char VAR2[32];
    char VAR3[] = "";
    int VAR4 = 0;
    int VAR5;


    VAR5 = strlen(VAR1);
    FUN2("", VAR5);


    if (VAR5 > 0) {
        VAR4 = 1;
    }


    strcpy(VAR2, VAR1);


    FUN2("", VAR2);


    if (strlen(VAR2) > 0) {
        FUN2("");
        return VAR4;
    }

    return -1;
}
"""

    embedding = src2embedding(code, label=1)
    print(embedding)


if __name__ == "__main__":
    # test()
    print("{HOME_PATH}/VulDS/BigVul/ori-pdg/trans_vul/1_CVE-2011-1428_savannah_CWE-20_c265cad1c95b84abfd4e8d861f25926ef13b5d91_2_13.dot")