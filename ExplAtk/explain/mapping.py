"""
Mapping explanation results back to source code (based on Joern CPG-bin)
=============================================

Data sources:
  1. cpg-bin file -> export all node JSON through Joern (including lineNumber, code, etc.)
  2. PDG dot file -> node traversal order (consistent with pdg2embedding) + edge types (DDG/CDG)
  3. CocaExplainer output -> node_importance, edge_importance

Output:
  Rich attack-target information, including both key nodes and key edges,
  with edge types (DDG/CDG) distinguished to support different attack strategies.
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
    Call Joern to export all node information from a cpg-bin file and return a Python list.

    Write to a temporary JSON file -> read it as a Python object -> delete the temporary file.

    Args:
        cpg_bin_path:   .bin file path
        joern_path:     Joern installation directory
        save_json_path: optional; if provided, save a copy to this path for later reuse without calling Joern again
        timeout:        Joern execution timeout in seconds

    Returns:
        list[dict]: parsed node list
    """
    # Temporary file path
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
            print(f"[Error] Joern did not generate an output file")
            print(f"  stdout: {stdout[:500]}")
            print(f"  stderr: {stderr[:500]}")
            return []

        # Read the temporary file
        with open(tmp_json, 'r', encoding='utf-8') as f:
            nodes = json.load(f)

        if not isinstance(nodes, list):
            print(f"[Error] JSON content is not a list")
            return []

        # Optional: save a copy for later reuse
        if save_json_path:
            os.makedirs(os.path.dirname(save_json_path) or '.', exist_ok=True)
            with open(save_json_path, 'w', encoding='utf-8') as f:
                json.dump(nodes, f, ensure_ascii=False, indent=2)

        return nodes

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"[Error] Joern execution timed out ({timeout}s)")
        return []
    except Exception as e:
        print(f"[Error] Export failed: {e}")
        return []
    finally:
        os.chdir(original_dir)
        # Clean up the temporary file
        if os.path.exists(tmp_json):
            try:
                os.unlink(tmp_json)
            except OSError:
                pass


def load_cpg_nodes_json(json_path):
    """Load an existing JSON file exported by Joern."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        return nodes if isinstance(nodes, list) else []
    except Exception as e:
        print(f"[Error] Failed to load JSON: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
# Core mapping function
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
    Map explainer output back to source code and provide complete attack information for nodes and edges.

    Choose either cpg_json_path or cpg_bin_path:
      - If cpg_json_path already exists, read it directly
      - Otherwise, call Joern to export from cpg_bin_path

    Args:
        explain_result:  return value of CocaExplainer.explain()
        dot_path:        PDG dot file path
        cpg_bin_path:    Joern cpg-bin file path (choose either this or cpg_json_path)
        cpg_json_path:   exported cpg.all JSON path (choose either this or cpg_bin_path)
        source_path:     original C source file path (used to read complete source lines)
        joern_path:      Joern installation directory
        top_k:           return the top-k key nodes/edges
        threshold:       optional; filter by importance threshold (takes precedence over top_k)

    Returns:
        ExplanationMapping object (see the class definition below)
    """
    node_importance = explain_result['node_importance']
    edge_importance = explain_result['edge_importance']
    num_nodes = len(node_importance)
    num_edges = len(edge_importance)

    # ── Step 0: Read the source file ──────────────────────────────────
    source_lines = {}
    if source_path and os.path.exists(source_path):
        source_lines = _read_source_file(source_path)

    # ── Step 1: Get CPG node information ─────────────────────────────
    if cpg_json_path and os.path.exists(cpg_json_path):
        cpg_nodes = load_cpg_nodes_json(cpg_json_path)
    elif cpg_bin_path:
        cpg_nodes = export_nodes_from_cpg_bin(
            cpg_bin_path,
            joern_path=joern_path,
            save_json_path=cpg_json_path,  # Optional: also save a copy for reuse
        )
    else:
        raise ValueError("Either cpg_bin_path or cpg_json_path must be provided")

    # Build the mapping from CPG node ID -> node details
    cpg_id2info = {}
    for node in cpg_nodes:
        nid = node.get('id')
        if nid is not None:
            cpg_id2info[str(nid)] = {
                'lineNumber': node.get('lineNumber'),
                'code': node.get('code', ''),
                'label': node.get('_label', ''),       # AST node type
                'name': node.get('name', ''),
                'typeFullName': node.get('typeFullName', ''),
                'order': node.get('order'),
            }

    # ── Step 2: Parse the dot file and build traversal order and edge information ───────────
    pdg = _load_dot_file(dot_path)
    if pdg is None:
        return ExplanationMapping.empty(num_nodes, num_edges)

    dot_nodes = list(pdg.nodes())
    if len(dot_nodes) != num_nodes:
        print(f"[Warning] Node count mismatch: dot={len(dot_nodes)}, graph_embedding={num_nodes}")

    # ── Step 3: Build graph embedding index -> source information ────────────────────
    index2info = {}
    for index, dot_node in enumerate(dot_nodes):
        if index >= num_nodes:
            break

        dot_node_id = str(dot_node)
        cpg_info = cpg_id2info.get(dot_node_id, {})

        # Extract code from the dot label as a fallback
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

    # ── Step 4: Build edge information (preserve types)──────────────────────────
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

    # ── Step 5: Aggregate node importance by line number ──────────────────────────
    line2nodes = {}  # line_no -> aggregated information
    for idx in range(num_nodes):
        info = index2info.get(idx, {})
        line_no = info.get('line_no')
        if line_no is None:
            continue
        line_no = int(line_no)
        score = float(node_importance[idx])

        # Code priority: source file > CPG node code > dot label
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

    # ── Step 6: Score edges and attach source information ────────────────────────
    # Override endpoint code with content from the source file if available
    scored_edges = []
    for i, ed in enumerate(edge_details):
        if i < num_edges:
            ed['importance'] = float(edge_importance[i])
        else:
            ed['importance'] = 0.0

        # Source-file override
        if source_lines:
            if ed.get('src_line') and ed['src_line'] in source_lines:
                ed['src_code'] = source_lines[ed['src_line']]
            if ed.get('dst_line') and ed['dst_line'] in source_lines:
                ed['dst_code'] = source_lines[ed['dst_line']]

        scored_edges.append(ed)

    # ── Step 7: Select top-k ────────────────────────────────────
    if threshold is not None:
        top_lines = [v for v in line2nodes.values() if v['importance'] > threshold]
        top_edges = [e for e in scored_edges if e['importance'] > threshold]
    else:
        top_lines = sorted(line2nodes.values(), key=lambda x: x['importance'], reverse=True)[:top_k]
        top_edges = sorted(scored_edges, key=lambda x: x['importance'], reverse=True)[:top_k]

    # ── Step 8: Statistics ──────────────────────────────────────────
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
# Mapping-result wrapper
# ══════════════════════════════════════════════════════════════════════

class ExplanationMapping:
    """Wrap mapping results and provide accessors from multiple perspectives."""

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
    # Attack target accessors
    # ─────────────────────────────────────────────────────────

    def get_token_attack_targets(self, top_k=5):
        """
        Get attack targets suitable for token-level replacement (identifier renaming).
        Prefer high-importance nodes that contain identifiers.

        Returns:
            list[dict]: [{'line_no', 'code', 'importance'}, ...]
        """
        return [
            {'line_no': n['line_no'], 'code': n['code'], 'importance': n['importance']}
            for n in self.vulnerable_nodes[:top_k]
        ]

    def get_control_flow_attack_targets(self, top_k=5):
        """
        Get attack targets suitable for control-flow transformations (for <-> while, if-else swapping, etc.).
        Select the highest-importance CDG edges and their endpoint statements.

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
        Get attack targets suitable for data-flow transformations (introducing temporary variables, operand swapping, etc.).
        Select the highest-importance DDG edges and their endpoint statements.

        Returns:
            list[dict]: [{
                'src_line', 'src_code',
                'dst_line', 'dst_code',
                'importance', 'edge_label',
                'dep_variable',   # variable name depended on by the DDG edge
            }, ...]
        """
        ddg_edges = [e for e in self.vulnerable_edges if e['edge_type'] == 'DDG']
        ddg_edges.sort(key=lambda x: x['importance'], reverse=True)
        return ddg_edges[:top_k]

    def get_all_attack_targets(self, top_k=5):
        """
        Get comprehensive attack targets grouped by strategy.

        Returns:
            dict: {
                'token_targets':        [...],  # suitable for identifier replacement
                'control_flow_targets': [...],  # suitable for control-flow transformation
                'data_flow_targets':    [...],  # suitable for data-flow transformation
            }
        """
        return {
            'token_targets': self.get_token_attack_targets(top_k),
            'control_flow_targets': self.get_control_flow_attack_targets(top_k),
            'data_flow_targets': self.get_data_flow_attack_targets(top_k),
        }
    
    def get_all_lines_ranked(self):
        """
        Return all lines sorted by importance in descending order, without top-k truncation.
        Used by the structure-transformation stage for global ranking.
        """
        return sorted(
            self.all_lines.values(),
            key=lambda x: x['importance'],
            reverse=True,
        )

    def get_all_edges_ranked(self):
        """
        Return all edges sorted by importance in descending order, without top-k truncation.
        Used by the structure-transformation stage for global ranking.
        """
        return sorted(
            self.all_edges,
            key=lambda x: x.get('importance', 0.0),
            reverse=True,
        )

    def get_identifier_importance(self, source_code, lang='c'):
        """
        Compute a priority weight for each identifier in the source code based on all node importance scores.

        Strategy:
          - For each line, use the max(importance) among all its nodes as the line score (already done in all_lines)
          - For each identifier, use the maximum line score across all lines where it appears as its identifier weight
          - Sort by weight in descending order
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
    # Display interface
    # ─────────────────────────────────────────────────────────

    def print_summary(self):
        """Print a complete summary of the mapping results."""
        print("=" * 75)
        print("  Coca Explanation → Source Code Mapping")
        print(f"  Mapping quality: {self.num_mapped}/{self.num_total} nodes"
              f" ({self.num_mapped/max(self.num_total,1)*100:.1f}%)")
        print("=" * 75)

        # ── Key nodes ──
        print("\n📌 Key nodes (sorted by importance):")
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

        # ── Key edges (CDG) ──
        cdg_edges = [e for e in self.vulnerable_edges if e['edge_type'] == 'CDG']
        if cdg_edges:
            print("🔀 Key control-dependence edges (CDG):")
            print("-" * 75)
            for i, edge in enumerate(cdg_edges):
                print(f"  #{i+1}  Line {edge['src_line']} → Line {edge['dst_line']}  "
                      f"[{edge['importance']:.4f}]")
                print(f"       Src: {edge['src_code'][:70]}")
                print(f"       Dst: {edge['dst_code'][:70]}")
                print()

        # ── Key edges (DDG) ──
        ddg_edges = [e for e in self.vulnerable_edges if e['edge_type'] == 'DDG']
        if ddg_edges:
            print("📊 Key data-dependence edges (DDG):")
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
        """Print the attack plan grouped by attack strategy."""
        targets = self.get_all_attack_targets()

        print("=" * 75)
        print("  Attack Plan")
        print("=" * 75)

        print("\n🏷️  Token-level attack (identifier renaming):")
        if targets['token_targets']:
            for t in targets['token_targets']:
                print(f"    Line {t['line_no']}: {t['code'][:70]}  "
                      f"(imp={t['importance']:.3f})")
        else:
            print("    No available targets")

        print(f"\n🔀 Control-flow attack (for <-> while, if-else swapping, etc.):")
        if targets['control_flow_targets']:
            for e in targets['control_flow_targets']:
                print(f"    Line {e['src_line']} → {e['dst_line']}  "
                      f"(imp={e['importance']:.3f})")
                print(f"      {e['src_code'][:60]}")
                print(f"      → {e['dst_code'][:60]}")
        else:
            print("    No CDG edge targets")

        print(f"\n📊 Data-flow attack (introducing temporary variables, operand swapping, etc.):")
        if targets['data_flow_targets']:
            for e in targets['data_flow_targets']:
                dep = e.get('dep_variable', '')
                print(f"    Line {e['src_line']} → {e['dst_line']}  "
                      f"(imp={e['importance']:.3f}) dep_var={dep}")
                print(f"      {e['src_code'][:60]}")
                print(f"      → {e['dst_code'][:60]}")
        else:
            print("    No DDG edge targets")

        print("=" * 75)

    def to_dict(self):
        """Serialize to a dict that can be saved as JSON."""
        return {
            'vulnerable_nodes': self.vulnerable_nodes,
            'vulnerable_edges': self.vulnerable_edges,
            'num_mapped': self.num_mapped,
            'num_total': self.num_total,
        }

    def save(self, path):
        """Save mapping results to a JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════
# Internal helper functions
# ══════════════════════════════════════════════════════════════════════

def _make_edge_detail(src_dot_node, dst_dot_node, edge_type, edge_label,
                      dot_nodes, index2info):
    """Construct detailed information for one edge."""
    src_idx = dot_nodes.index(src_dot_node) if src_dot_node in dot_nodes else None
    dst_idx = dot_nodes.index(dst_dot_node) if dst_dot_node in dot_nodes else None

    src_info = index2info.get(src_idx, {}) if src_idx is not None else {}
    dst_info = index2info.get(dst_idx, {}) if dst_idx is not None else {}

    # Extract the dependent variable name from the DDG edge label
    # Typical formats: "DDG: varName" or "DDG:varName"
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
        'importance': 0.0,  # filled later
    }


def _load_dot_file(dot_path):
    """Safely load a dot file with LooseDotParser."""
    if not os.path.exists(dot_path):
        print(f"[Error] dot file does not exist: {dot_path}")
        return None

    try:
        with open(dot_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Use the new parser
        parser = LooseDotParser()
        pdg = parser.to_networkx(content)
        
        return pdg

    except Exception as e:
        print(f"[Error] Failed to parse dot file: {e}")
        return None


def _read_source_file(source_path):
    """
    Read a C source file and return a mapping of {line number: code content}.
    Line numbers start from 1.
    """
    source_lines = {}
    try:
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, start=1):
                source_lines[i] = line.rstrip('\n\r')
    except Exception as e:
        print(f"[Warning] Failed to read source file {source_path}: {e}")
    return source_lines


def _extract_code_from_dot_label(label_str):
    """Extract a code snippet from a dot node label."""
    if not label_str:
        return ''
    label_str = label_str.strip('"')
    if label_str.startswith('(') and label_str.endswith(')'):
        label_str = label_str[1:-1]
    code = label_str.partition(',')[2].strip()
    return code


# ══════════════════════════════════════════════════════════════════════
# Convenience entry point
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
    Run the explainer and map the results back to source code in one step.

    Args:
        explainer:       CocaExplainer instance
        data:            PyG Data object
        predicted_label: model-predicted label
        dot_path:        PDG dot file path
        cpg_bin_path:    Joern cpg-bin path (optional)
        cpg_json_path:   exported JSON path (optional)
        source_path:     original C source file path (optional)
        joern_path:      Joern installation path
        top_k:           return top-k

    Returns:
        ExplanationMapping object
    """
    # Run the explainer
    result = explainer.explain(data, predicted_label)

    # Map back to source code
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
# Usage example
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

    # 1. Run the explainer
    wrapper = ModelWrapper('reveal', '{MODEL_SAVE_PATH}/trained_model/ori-ds/reveal/reveal-cwe119/mod_94.59_92.5_96.77_93.61.ckpt')
    pred_label,conf,margin = wrapper.predict_label_and_true_conf_margin(src2embedding(src_code,0),0)
    print(pred_label)
    print(conf)
    print(margin)
    # explainer = CocaExplainer(model=wrapper.model, device='cuda')
    # data = read_json('{HOME_PATH}/VulDS/BigVul/ori-embedding/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.json')               # a sample detected as vulnerable
    # pred_label = wrapper.predict_label(data)  # usually 1
    # result = explainer.explain(data, pred_label)

    # # 2. Map back to source code (full information)
    # mapping = map_explanation_to_source(
    #     explain_result=result,
    #     dot_path='{HOME_PATH}/VulDS/BigVul/ori-pdg/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.dot',
    #     cpg_bin_path='{HOME_PATH}/VulDS/BigVul/all-cpg-bin/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.bin',
    #     source_path='{HOME_PATH}/VulDS/BigVul/all-src/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.c',     # optional
    # )

    # Get strategy-grouped attack targets
    # mapping.print_summary()
    # with open('{HOME_PATH}/VulDS/BigVul/all-src/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.c', 'r', encoding='utf-8', errors='ignore') as f:
    #     source_code = f.read()
    # res = mapping.get_identifier_importance(source_code=source_code)
    # print(res)

if __name__ == '__main__':
    demo_mapping()

