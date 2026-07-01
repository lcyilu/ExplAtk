"""
解释结果到源码的映射（基于 Joern CPG-bin）
=============================================

数据来源：
  1. cpg-bin 文件 → 通过 Joern 导出所有节点 JSON（含 lineNumber, code 等）
  2. PDG dot 文件 → 节点遍历顺序（与 pdg2embedding 一致）+ 边类型（DDG/CDG）
  3. CocaExplainer 输出 → node_importance, edge_importance

输出：
  丰富的攻击目标信息，同时包含关键节点和关键边，
  区分边类型（DDG/CDG），支持不同的攻击策略。
"""

import json
import os
import re
import tempfile
import subprocess
from typing import List
import numpy as np
import networkx as nx
from src.utils.dot_parser import LooseDotParser


def export_nodes_from_cpg_bin(
    cpg_bin_path,
    joern_path='{HOME_PATH}/joerns/joern-src/joern-1.1.172',
    save_json_path=None,
    timeout=120,
):
    """
    调用 Joern 从 cpg-bin 文件导出所有节点信息，返回 Python 列表。

    写入临时 JSON 文件 → 读取为 Python 对象 → 删除临时文件。

    Args:
        cpg_bin_path:   .bin 文件路径
        joern_path:     Joern 安装目录
        save_json_path: 可选，若提供则保存一份到该路径（方便后续复用，不再调 Joern）
        timeout:        Joern 执行超时秒数

    Returns:
        list[dict]: 解析后的节点列表
    """
    # 临时文件路径
    tmp_json = os.path.join(tempfile.gettempdir(), f'joern_cpg_{os.getpid()}.json')

    original_dir = os.getcwd()
    try:
        os.chdir(joern_path)

        import_cmd = f'importCpg("{cpg_bin_path}")\r'
        export_cmd = f'cpg.all.toJsonPretty |> "{tmp_json}"\r'
        cmd = import_cmd + export_cmd

        proc = subprocess.Popen(
            ["./joern"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding='utf-8',
        )
        stdout, stderr = proc.communicate(cmd, timeout=timeout)

        if not os.path.exists(tmp_json):
            print(f"[错误] Joern 未生成输出文件")
            print(f"  stdout: {stdout[:500]}")
            print(f"  stderr: {stderr[:500]}")
            return []

        # 读取临时文件
        with open(tmp_json, 'r', encoding='utf-8') as f:
            nodes = json.load(f)

        if not isinstance(nodes, list):
            print(f"[错误] JSON 内容不是列表")
            return []

        # 可选：保存一份供后续复用
        if save_json_path:
            os.makedirs(os.path.dirname(save_json_path) or '.', exist_ok=True)
            with open(save_json_path, 'w', encoding='utf-8') as f:
                json.dump(nodes, f, ensure_ascii=False, indent=2)

        return nodes

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"[错误] Joern 执行超时（{timeout}s）")
        return []
    except Exception as e:
        print(f"[错误] 导出失败: {e}")
        return []
    finally:
        os.chdir(original_dir)
        # 清理临时文件
        if os.path.exists(tmp_json):
            try:
                os.unlink(tmp_json)
            except OSError:
                pass


def load_cpg_nodes_json(json_path):
    """加载已存在的 Joern 导出 JSON 文件。"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        return nodes if isinstance(nodes, list) else []
    except Exception as e:
        print(f"[错误] 加载 JSON 失败: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
# 核心映射函数
# ══════════════════════════════════════════════════════════════════════

def map_explanation_to_source(
    explain_result,
    dot_path,
    cpg_bin_path=None,
    cpg_json_path=None,
    source_path=None,
    joern_path='{HOME_PATH}/joerns/joern-src/joern-1.1.172',
    top_k=5,
    threshold=None,
) :
    """
    将解释器输出映射回源码，提供节点和边的完整攻击信息。

    cpg_json_path 和 cpg_bin_path 二选一：
      - 如果 cpg_json_path 已存在，直接读取
      - 否则从 cpg_bin_path 调用 Joern 导出

    Args:
        explain_result:  CocaExplainer.explain() 的返回值
        dot_path:        PDG dot 文件路径
        cpg_bin_path:    Joern cpg-bin 文件路径（与 cpg_json_path 二选一）
        cpg_json_path:   已导出的 cpg.all JSON 路径（与 cpg_bin_path 二选一）
        source_path:     原始 C 源码文件路径（用于读取完整源码行）
        joern_path:      Joern 安装目录
        top_k:           返回前 k 个关键节点/边
        threshold:       可选，按重要性阈值过滤（优先于 top_k）

    Returns:
        ExplanationMapping 对象（见下方类定义）
    """
    node_importance = explain_result['node_importance']
    edge_importance = explain_result['edge_importance']
    num_nodes = len(node_importance)
    num_edges = len(edge_importance)

    # ── Step 0: 读取源码文件 ──────────────────────────────────
    source_lines = {}
    if source_path and os.path.exists(source_path):
        source_lines = _read_source_file(source_path)

    # ── Step 1: 获取 CPG 节点信息 ─────────────────────────────
    if cpg_json_path and os.path.exists(cpg_json_path):
        cpg_nodes = load_cpg_nodes_json(cpg_json_path)
    elif cpg_bin_path:
        cpg_nodes = export_nodes_from_cpg_bin(
            cpg_bin_path,
            joern_path=joern_path,
            save_json_path=cpg_json_path,  # 可选：顺便保存一份供复用
        )
    else:
        raise ValueError("必须提供 cpg_bin_path 或 cpg_json_path 之一")

    # 构建 CPG节点ID → 节点详情 的映射
    cpg_id2info = {}
    for node in cpg_nodes:
        nid = node.get('id')
        if nid is not None:
            cpg_id2info[str(nid)] = {
                'lineNumber': node.get('lineNumber'),
                'code': node.get('code', ''),
                'label': node.get('_label', ''),       # AST节点类型
                'name': node.get('name', ''),
                'typeFullName': node.get('typeFullName', ''),
                'order': node.get('order'),
            }

    # ── Step 2: 解析 dot 文件，建立遍历顺序和边信息 ───────────
    pdg = _load_dot_file(dot_path)
    if pdg is None:
        return ExplanationMapping.empty(num_nodes, num_edges)

    dot_nodes = list(pdg.nodes())
    if len(dot_nodes) != num_nodes:
        print(f"[警告] 节点数不一致: dot={len(dot_nodes)}, 图嵌入={num_nodes}")

    # ── Step 3: 构建 图嵌入索引 → 源码信息 ────────────────────
    index2info = {}
    for index, dot_node in enumerate(dot_nodes):
        if index >= num_nodes:
            break

        dot_node_id = str(dot_node)
        cpg_info = cpg_id2info.get(dot_node_id, {})

        # 从 dot label 中提取代码作为备选
        dot_label = pdg.nodes[dot_node].get('label', '')
        dot_code = _extract_code_from_dot_label(dot_label)

        index2info[index] = {
            'line_no': cpg_info.get('lineNumber'),
            'code': cpg_info.get('code') or dot_code,
            'ast_type': cpg_info.get('label', ''),
            'name': cpg_info.get('name', ''),
            'type_name': cpg_info.get('typeFullName', ''),
            'dot_node_id': dot_node_id,
        }

    # ── Step 4: 构建边信息（保留类型）──────────────────────────
    edge_details = []
    for item in pdg.adj.items():
        s = item[0]
        for d in item[1]:
            ddg_added = False
            cdg_added = False
            for edge_key, edge_data in item[1][d].items():
                elabel = edge_data.get('label', '')
                if 'DDG' in elabel and not ddg_added:
                    edge_details.append(_make_edge_detail(
                        s, d, 'DDG', elabel, dot_nodes, index2info
                    ))
                    ddg_added = True
                elif 'CDG' in elabel and not cdg_added:
                    edge_details.append(_make_edge_detail(
                        s, d, 'CDG', elabel, dot_nodes, index2info
                    ))
                    cdg_added = True

    # ── Step 5: 按行号聚合节点重要性 ──────────────────────────
    line2nodes = {}  # line_no → 聚合信息
    for idx in range(num_nodes):
        info = index2info.get(idx, {})
        line_no = info.get('line_no')
        if line_no is None:
            continue
        line_no = int(line_no)
        score = float(node_importance[idx])

        # 代码优先级：源码文件 > CPG节点code > dot label
        code = source_lines.get(line_no) or info.get('code', '')

        if line_no not in line2nodes:
            line2nodes[line_no] = {
                'line_no': line_no,
                'code': code,
                'cpg_code': info.get('code', ''),
                'importance': score,
                'node_indices': [idx],
                'ast_types': [info.get('ast_type', '')],
            }
        else:
            line2nodes[line_no]['node_indices'].append(idx)
            line2nodes[line_no]['ast_types'].append(info.get('ast_type', ''))
            if score > line2nodes[line_no]['importance']:
                line2nodes[line_no]['importance'] = score

    # ── Step 6: 给边打分并附加源码信息 ────────────────────────
    # 用源码文件的内容覆盖边端点的代码（如果有源码文件）
    scored_edges = []
    for i, ed in enumerate(edge_details):
        if i < num_edges:
            ed['importance'] = float(edge_importance[i])
        else:
            ed['importance'] = 0.0

        # 源码文件覆盖
        if source_lines:
            if ed.get('src_line') and ed['src_line'] in source_lines:
                ed['src_code'] = source_lines[ed['src_line']]
            if ed.get('dst_line') and ed['dst_line'] in source_lines:
                ed['dst_code'] = source_lines[ed['dst_line']]

        scored_edges.append(ed)

    # ── Step 7: 筛选 top-k ────────────────────────────────────
    if threshold is not None:
        top_lines = [v for v in line2nodes.values() if v['importance'] > threshold]
        top_edges = [e for e in scored_edges if e['importance'] > threshold]
    else:
        top_lines = sorted(line2nodes.values(), key=lambda x: x['importance'], reverse=True)[:top_k]
        top_edges = sorted(scored_edges, key=lambda x: x['importance'], reverse=True)[:top_k]

    # ── Step 8: 统计 ──────────────────────────────────────────
    num_mapped = sum(1 for v in index2info.values() if v.get('line_no') is not None)

    return ExplanationMapping(
        vulnerable_nodes=sorted(top_lines, key=lambda x: x['importance'], reverse=True),
        vulnerable_edges=top_edges,
        all_lines=line2nodes,
        all_edges=scored_edges,
        index2info=index2info,
        num_mapped=num_mapped,
        num_total=num_nodes,
    )


# ══════════════════════════════════════════════════════════════════════
# 映射结果封装
# ══════════════════════════════════════════════════════════════════════

class ExplanationMapping:
    """封装映射结果，提供多种视角的访问接口。"""

    def __init__(self, vulnerable_nodes, vulnerable_edges, all_lines,
                 all_edges, index2info, num_mapped, num_total):
        self.vulnerable_nodes = vulnerable_nodes
        self.vulnerable_edges = vulnerable_edges
        self.all_lines = all_lines
        self.all_edges = all_edges
        self.index2info = index2info
        self.num_mapped = num_mapped
        self.num_total = num_total

    @classmethod
    def empty(cls, num_nodes, num_edges):
        return cls([], [], {}, [], {}, 0, num_nodes)

    # ─────────────────────────────────────────────────────────
    # 攻击目标获取接口
    # ─────────────────────────────────────────────────────────

    def get_token_attack_targets(self, top_k=5):
        """
        获取适合做 token 级替换（标识符重命名）的攻击目标。
        优先选择包含标识符的高重要性节点。

        Returns:
            list[dict]: [{'line_no', 'code', 'importance'}, ...]
        """
        return [
            {'line_no': n['line_no'], 'code': n['code'], 'importance': n['importance']}
            for n in self.vulnerable_nodes[:top_k]
        ]

    def get_control_flow_attack_targets(self, top_k=5):
        """
        获取适合做控制流变换（for↔while, if-else交换等）的攻击目标。
        筛选 CDG 边中重要性最高的边及其端点语句。

        Returns:
            list[dict]: [{
                'src_line', 'src_code',
                'dst_line', 'dst_code',
                'importance', 'edge_label'
            }, ...]
        """
        cdg_edges = [e for e in self.vulnerable_edges if e['edge_type'] == 'CDG']
        cdg_edges.sort(key=lambda x: x['importance'], reverse=True)
        return cdg_edges[:top_k]

    def get_data_flow_attack_targets(self, top_k=5):
        """
        获取适合做数据流变换（引入临时变量、操作数交换等）的攻击目标。
        筛选 DDG 边中重要性最高的边及其端点语句。

        Returns:
            list[dict]: [{
                'src_line', 'src_code',
                'dst_line', 'dst_code',
                'importance', 'edge_label',
                'dep_variable',   # DDG 依赖的变量名
            }, ...]
        """
        ddg_edges = [e for e in self.vulnerable_edges if e['edge_type'] == 'DDG']
        ddg_edges.sort(key=lambda x: x['importance'], reverse=True)
        return ddg_edges[:top_k]

    def get_all_attack_targets(self, top_k=5):
        """
        获取综合攻击目标，按策略分组。

        Returns:
            dict: {
                'token_targets':        [...],  # 适合标识符替换
                'control_flow_targets': [...],  # 适合控制流变换
                'data_flow_targets':    [...],  # 适合数据流变换
            }
        """
        return {
            'token_targets': self.get_token_attack_targets(top_k),
            'control_flow_targets': self.get_control_flow_attack_targets(top_k),
            'data_flow_targets': self.get_data_flow_attack_targets(top_k),
        }
    
    def get_all_lines_ranked(self):
        """
        返回所有行按重要性降序排列，不做 top-k 截断。
        供结构变换阶段使用全局排序。
        """
        return sorted(
            self.all_lines.values(),
            key=lambda x: x['importance'],
            reverse=True,
        )

    def get_all_edges_ranked(self):
        """
        返回所有边按重要性降序排列，不做 top-k 截断。
        供结构变换阶段使用全局排序。
        """
        return sorted(
            self.all_edges,
            key=lambda x: x.get('importance', 0.0),
            reverse=True,
        )

    def get_identifier_importance(self, source_code, lang='c'):
        """
        基于所有节点的重要性分数，为源码中每个标识符计算优先级权重。

        策略：
          - 每行取其所有节点中的 max(importance) 作为行分数（已在 all_lines 中完成）
          - 每个标识符取其出现的所有行中的 max(行分数) 作为标识符权重
          - 按权重降序排列
        """
        import re
        from common.utils.parser import extract_identifiers_from_one_src

        code_bytes = source_code.encode('utf-8') if isinstance(source_code, str) else source_code
        raw_ids = extract_identifiers_from_one_src(code_bytes, lang=lang)
        unique_ids = list(set(raw_ids))

        if not unique_ids:
            return []

        lines_list = source_code.split('\n') if isinstance(source_code, str) else source_code.decode('utf-8').split('\n')

        result = []
        for ident in unique_ids:
            appeared_lines = []
            pattern = re.compile(r'\b' + re.escape(ident) + r'\b')
            for line_idx, line_content in enumerate(lines_list):
                if pattern.search(line_content):
                    appeared_lines.append(line_idx + 1)

            max_imp = 0.0
            for ln in appeared_lines:
                if ln in self.all_lines:
                    max_imp = max(max_imp, self.all_lines[ln]['importance'])

            result.append({
                'name': ident,
                'importance': max_imp,
                'lines': appeared_lines,
            })

        result.sort(key=lambda x: x['importance'], reverse=True)
        return result


    # ─────────────────────────────────────────────────────────
    # 展示接口
    # ─────────────────────────────────────────────────────────

    def print_summary(self):
        """打印完整的映射结果摘要。"""
        print("=" * 75)
        print("  Coca Explanation → Source Code Mapping")
        print(f"  映射质量: {self.num_mapped}/{self.num_total} 节点"
              f" ({self.num_mapped/max(self.num_total,1)*100:.1f}%)")
        print("=" * 75)

        # ── 关键节点 ──
        print("\n📌 关键节点 (按重要性排序):")
        print("-" * 75)
        for i, node in enumerate(self.vulnerable_nodes):
            bar = '█' * int(node['importance'] * 25)
            print(f"  #{i+1}  Line {node['line_no']:<5d} "
                  f"[{node['importance']:.4f}] {bar}")
            if node['code']:
                print(f"       Code: {node['code'][:85]}")
            print(f"       Nodes: {node['node_indices']}  "
                  f"AST: {list(set(node['ast_types']))[:3]}")
            print()

        # ── 关键边（CDG）──
        cdg_edges = [e for e in self.vulnerable_edges if e['edge_type'] == 'CDG']
        if cdg_edges:
            print("🔀 关键控制依赖边 (CDG):")
            print("-" * 75)
            for i, edge in enumerate(cdg_edges):
                print(f"  #{i+1}  Line {edge['src_line']} → Line {edge['dst_line']}  "
                      f"[{edge['importance']:.4f}]")
                print(f"       Src: {edge['src_code'][:70]}")
                print(f"       Dst: {edge['dst_code'][:70]}")
                print()

        # ── 关键边（DDG）──
        ddg_edges = [e for e in self.vulnerable_edges if e['edge_type'] == 'DDG']
        if ddg_edges:
            print("📊 关键数据依赖边 (DDG):")
            print("-" * 75)
            for i, edge in enumerate(ddg_edges):
                dep_var = edge.get('dep_variable', '')
                var_info = f"  var={dep_var}" if dep_var else ""
                print(f"  #{i+1}  Line {edge['src_line']} → Line {edge['dst_line']}  "
                      f"[{edge['importance']:.4f}]{var_info}")
                print(f"       Src: {edge['src_code'][:70]}")
                print(f"       Dst: {edge['dst_code'][:70]}")
                print()

        print("=" * 75)

    def print_attack_plan(self):
        """打印按攻击策略分组的攻击计划。"""
        targets = self.get_all_attack_targets()

        print("=" * 75)
        print("  攻击计划")
        print("=" * 75)

        print("\n🏷️  Token 级攻击（标识符重命名）:")
        if targets['token_targets']:
            for t in targets['token_targets']:
                print(f"    Line {t['line_no']}: {t['code'][:70]}  "
                      f"(imp={t['importance']:.3f})")
        else:
            print("    无可用目标")

        print(f"\n🔀 控制流攻击（for↔while, if-else交换等）:")
        if targets['control_flow_targets']:
            for e in targets['control_flow_targets']:
                print(f"    Line {e['src_line']} → {e['dst_line']}  "
                      f"(imp={e['importance']:.3f})")
                print(f"      {e['src_code'][:60]}")
                print(f"      → {e['dst_code'][:60]}")
        else:
            print("    无 CDG 边目标")

        print(f"\n📊 数据流攻击（引入临时变量、操作数交换等）:")
        if targets['data_flow_targets']:
            for e in targets['data_flow_targets']:
                dep = e.get('dep_variable', '')
                print(f"    Line {e['src_line']} → {e['dst_line']}  "
                      f"(imp={e['importance']:.3f}) dep_var={dep}")
                print(f"      {e['src_code'][:60]}")
                print(f"      → {e['dst_code'][:60]}")
        else:
            print("    无 DDG 边目标")

        print("=" * 75)

    def to_dict(self):
        """序列化为 dict（可保存为 JSON）。"""
        return {
            'vulnerable_nodes': self.vulnerable_nodes,
            'vulnerable_edges': self.vulnerable_edges,
            'num_mapped': self.num_mapped,
            'num_total': self.num_total,
        }

    def save(self, path):
        """保存映射结果到 JSON 文件。"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════
# 内部辅助函数
# ══════════════════════════════════════════════════════════════════════

def _make_edge_detail(src_dot_node, dst_dot_node, edge_type, edge_label,
                      dot_nodes, index2info):
    """构造一条边的详细信息。"""
    src_idx = dot_nodes.index(src_dot_node) if src_dot_node in dot_nodes else None
    dst_idx = dot_nodes.index(dst_dot_node) if dst_dot_node in dot_nodes else None

    src_info = index2info.get(src_idx, {}) if src_idx is not None else {}
    dst_info = index2info.get(dst_idx, {}) if dst_idx is not None else {}

    # 从 DDG 边的 label 中提取依赖变量名
    # 典型格式: "DDG: varName" 或 "DDG:varName"
    dep_variable = ''
    if edge_type == 'DDG' and ':' in edge_label:
        parts = edge_label.split(':')
        if len(parts) >= 2:
            dep_variable = parts[-1].strip().strip('"')

    return {
        'src_index': src_idx,
        'dst_index': dst_idx,
        'src_line': src_info.get('line_no'),
        'dst_line': dst_info.get('line_no'),
        'src_code': src_info.get('code', ''),
        'dst_code': dst_info.get('code', ''),
        'edge_type': edge_type,
        'edge_label': edge_label.strip('"'),
        'dep_variable': dep_variable,
        'importance': 0.0,  # 后续填充
    }


def _load_dot_file(dot_path):
    """使用 LooseDotParser 安全加载 dot 文件。"""
    if not os.path.exists(dot_path):
        print(f"[错误] dot 文件不存在: {dot_path}")
        return None

    try:
        with open(dot_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用新的解析器
        parser = LooseDotParser()
        pdg = parser.to_networkx(content)
        
        return pdg

    except Exception as e:
        print(f"[错误] 解析 dot 文件失败: {e}")
        return None


def _read_source_file(source_path):
    """
    读取 C 源码文件，返回 {行号: 代码内容} 的映射。
    行号从 1 开始。
    """
    source_lines = {}
    try:
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, start=1):
                source_lines[i] = line.rstrip('\n\r')
    except Exception as e:
        print(f"[警告] 读取源码文件失败 {source_path}: {e}")
    return source_lines


def _extract_code_from_dot_label(label_str):
    """从 dot 节点 label 中提取代码片段。"""
    if not label_str:
        return ''
    label_str = label_str.strip('"')
    if label_str.startswith('(') and label_str.endswith(')'):
        label_str = label_str[1:-1]
    code = label_str.partition(',')[2].strip()
    return code


# ══════════════════════════════════════════════════════════════════════
# 便捷入口
# ══════════════════════════════════════════════════════════════════════

def explain_and_map(
    explainer,
    data,
    predicted_label,
    dot_path,
    cpg_bin_path=None,
    cpg_json_path=None,
    source_path=None,
    joern_path='{HOME_PATH}/joerns/joern-src/joern-1.1.172',
    top_k=5,
):
    """
    一步到位：运行解释器 + 映射回源码。

    Args:
        explainer:       CocaExplainer 实例
        data:            PyG Data 对象
        predicted_label: 模型预测标签
        dot_path:        PDG dot 文件路径
        cpg_bin_path:    Joern cpg-bin 路径（可选）
        cpg_json_path:   已导出的 JSON 路径（可选）
        source_path:     原始 C 源码文件路径（可选）
        joern_path:      Joern 安装路径
        top_k:           返回 top-k

    Returns:
        ExplanationMapping 对象
    """
    # 运行解释器
    result = explainer.explain(data, predicted_label)

    # 映射回源码
    mapping = map_explanation_to_source(
        explain_result=result,
        dot_path=dot_path,
        cpg_bin_path=cpg_bin_path,
        cpg_json_path=cpg_json_path,
        source_path=source_path,
        joern_path=joern_path,
        top_k=top_k,
    )

    return mapping


# ══════════════════════════════════════════════════════════════════════
# 使用示例
# ══════════════════════════════════════════════════════════════════════
def demo_mapping():
    from coca_explainer import CocaExplainer
    from src.model.wrapper import ModelWrapper
    from src.utils.gen_embedding import read_json
    from src.utils.gen_embedding import src2embedding
    
    src_code = """
static guint32 Short (
    proto_tree * kByteMark ,
    tvbuff_t * Get16uBuf ,
    gint bytes ,
    gint length ,
    const guint lookChar
) {
    guint32 value ;
    gboolean length_error ;

    switch ( length ) {
    case 1 :
        value = tvb_get_guint8 ( Get16uBuf , bytes ) ;
        break ;

    case 2 :
        value = ( lookChar & ENC_LITTLE_ENDIAN )
            ? tvb_get_letohs ( Get16uBuf , bytes )
            : tvb_get_ntohs ( Get16uBuf , bytes ) ;
        break ;

    case 3 :
        value = ( lookChar & ENC_LITTLE_ENDIAN )
            ? tvb_get_letoh24 ( Get16uBuf , bytes )
            : tvb_get_ntoh24 ( Get16uBuf , bytes ) ;
        break ;

    case 4 :
        value = ( lookChar & ENC_LITTLE_ENDIAN )
            ? tvb_get_letohl ( Get16uBuf , bytes )
            : tvb_get_ntohl ( Get16uBuf , bytes ) ;
        break ;

    default :
        if ( length < 1 ) {
            length_error = TRUE ;
            value = 0 ;
        }
        else {
            length_error = FALSE ;
            value = ( lookChar & ENC_LITTLE_ENDIAN )
                ? tvb_get_letohl ( Get16uBuf , bytes )
                : tvb_get_ntohl ( Get16uBuf , bytes ) ;
        }

        report_type_length_mismatch (
            kByteMark ,
            "an unsigned integer" ,
            length ,
            length_error
        ) ;
        break ;
    }

    return value ;
}
"""

    # 1. 运行解释器
    wrapper = ModelWrapper('reveal', '{HOME_PATH}/vul_explain/23_explain_eval_ISSTA/trained_model/ori-ds/reveal/reveal-cwe119/mod_94.59_92.5_96.77_93.61.ckpt')
    pred_label,conf,margin = wrapper.predict_label_and_true_conf_margin(src2embedding(src_code,0),0)
    print(pred_label)
    print(conf)
    print(margin)
    # explainer = CocaExplainer(model=wrapper.model, device='cuda')
    # data = read_json('{HOME_PATH}/VulDS/BigVul/ori-embedding/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.json')               # 一个被检测为 vulnerable 的样本
    # pred_label = wrapper.predict_label(data)  # 通常为 1
    # result = explainer.explain(data, pred_label)

    # # 2. 映射回源码（完整信息）
    # mapping = map_explanation_to_source(
    #     explain_result=result,
    #     dot_path='{HOME_PATH}/VulDS/BigVul/ori-pdg/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.dot',
    #     cpg_bin_path='{HOME_PATH}/VulDS/BigVul/all-cpg-bin/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.bin',
    #     source_path='{HOME_PATH}/VulDS/BigVul/all-src/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.c',     # 可选
    # )

    # 获取分策略攻击目标
    # mapping.print_summary()
    # with open('{HOME_PATH}/VulDS/BigVul/all-src/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.c', 'r', encoding='utf-8', errors='ignore') as f:
    #     source_code = f.read()
    # res = mapping.get_identifier_importance(source_code=source_code)
    # print(res)

if __name__ == '__main__':
    demo_mapping()

