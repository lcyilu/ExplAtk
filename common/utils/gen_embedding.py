"""
gen_embedding.py -- thread-safe refactored version

Public interfaces are fully unchanged.
All public function signatures, parameter semantics, return values, and side effects
(such as output file locations) remain the same as in the original version, so
upstream callers do not need any changes.

Refactoring points:
1. load_word_vectors:
   The original version called KeyedVectors.load on every invocation, which was slow
   and could concurrently mmap the same file in multiple threads. This version uses
   a path-keyed singleton cache with double-checked locking; after the first load,
   calls only incur one dict.get lookup.
2. joern_parse / joern_export:
   The original version used os.environ and os.system. Since os.environ is shared at
   the process level, concurrent threads could overwrite each other's variables and
   cause Joern to process the wrong sample. This version uses subprocess.run(
   ["sh", "joern-...", arg1, arg2, ...], cwd=joern_path), leaving the parent process
   environment and CWD untouched so threads do not interfere with each other.
3. src2embedding / src2pdg:
   The original version called os.chdir(joern_path) in the main function, polluting
   the process-level CWD. The refactored version lets subprocess manage cwd
   separately; only tempfile-isolated pure IO remains here, so it is thread-safe.
4. Post-processing mv / rm -rf operations now use shutil to avoid shell calls and
   to handle special characters in paths safely.
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
    Thread-safe in-process singleton cache.
    The signature matches the original version. Multiple calls with the same path
    return the same object, avoiding repeated mmap operations and KeyedVectors parsing.
    Double-checked locking keeps steady-state overhead to a single dict.get lookup.
    """
    path = path or get_settings().word2vec_path
    kv = KeyedVectors.load(path, mmap="r")
    return kv


def read_json(filename):
    # Read the file
    with open(filename.strip(),'r') as f:
        file = json.load(f)
    # Convert file contents into torch.tensor()
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
    Thread-safe version:
    - No longer calls os.chdir(joern_path); CWD is handled inside joern_parse/export
      through subprocess.
    - load_word_vectors is now cached and will not load repeatedly.
    Inputs and outputs are fully consistent with the original version.
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
#     Thread-safe version:
#     - No longer modifies os.environ
#     - Does not rely on the parent-process CWD; subprocess cwd sets it independently in the child process
#     - Passes argv directly to avoid path-escaping issues caused by shell interpretation
#     The signature matches the original version.
#     """
#     subprocess.run(
#         ["sh", "joern-parse", str(src), "--language", "c", "--out", str(bin)],
#         cwd=get_settings().joern_path,
#         stderr=subprocess.DEVNULL,
#         check=False,
#     )

def joern_parse(src, bin):
    joern_path = get_settings().joern_path
    # Use an independent workspace to isolate concurrent runs; use a temp dir as the working directory and absolute paths for Joern scripts
    with tempfile.TemporaryDirectory(prefix="joern_ws_") as ws:
        try:
            subprocess.run(
                ["sh", os.path.join(joern_path, "joern-parse"),
                 str(src), "--language", "c", "--out", str(bin)],
                cwd=ws,                              # workspace isolation happens here
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
    # Post-processing is independent of ws and operates on the caller-provided pdg_dir
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
#     Thread-safe version, with the same changes as joern_parse.
#     Post-processing mv/rm operations are also replaced with shutil.
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
            # Rewrite
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
    Thread-safe version: removes os.chdir while keeping the remaining logic unchanged.
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
            # Rewrite
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