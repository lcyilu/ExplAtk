import re

import networkx as nx


class LooseDotParser:
    def __init__(self):
        self.node_pattern = re.compile(
            r'(?s)^\s*"(\d+)"\s*\[\s*label\s*=\s*"(\(([^,]+),\s*(.*?)\))"\s*\]\s*$',
            re.MULTILINE,
        )
        self.edge_pattern = re.compile(
            r'(?s)^\s*"(\d+)"\s*->\s*"(\d+)"\s*\[\s*label\s*=\s*"(.*?)"\s*\]\s*$',
            re.MULTILINE,
        )

    def parse(self, dot_content: str):
        nodes = {}
        edges = []
        for match in self.node_pattern.finditer(dot_content):
            node_id = match.group(1)
            raw_label = match.group(2)
            clean_label = raw_label.replace('\\"', '"').replace('\\\\', '\\')
            nodes[node_id] = clean_label
        for match in self.edge_pattern.finditer(dot_content):
            src_id = match.group(1)
            dst_id = match.group(2)
            raw_label = match.group(3)
            clean_label = raw_label.replace('\\"', '"') if raw_label else ""
            edges.append((src_id, dst_id, clean_label))
        return nodes, edges

    def to_networkx(self, dot_content: str) -> nx.MultiDiGraph:
        nodes, edges = self.parse(dot_content)
        g = nx.MultiDiGraph()
        for nid, label in nodes.items():
            g.add_node(nid, label=label)
        for src, dst, label in edges:
            g.add_edge(src, dst, label=label)
        return g
