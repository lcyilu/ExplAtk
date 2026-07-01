from __future__ import annotations

from typing import Union

from common.utils.parser import initialize_language_parser, src2tree


def _ensure_bytes(src: Union[str, bytes]) -> bytes:
    if isinstance(src, str):
        return src.encode("utf-8")
    return src


def rename_identifier(src_bytes, old_name, new_name, lang="cpp"):
    src_bytes = _ensure_bytes(src_bytes)
    parser = initialize_language_parser(lang)
    root_node = src2tree(parser, src_bytes)
    identifiers_info = []

    def find_old_id(node):
        for child in node.children:
            if child.type in ["identifier", "field_identifier"]:
                id_name = src_bytes[child.start_byte : child.end_byte].decode("utf-8")
                if id_name == old_name:
                    identifiers_info.append(
                        {"start_point": child.start_point, "end_point": child.end_point}
                    )
            find_old_id(child)

    find_old_id(root_node)
    identifiers_info.sort(
        key=lambda x: (x["start_point"][0], x["start_point"][1]), reverse=True
    )
    code_lines = src_bytes.decode("utf-8").split("\n")
    for identifier in identifiers_info:
        start_row, start_col = identifier["start_point"]
        end_row, end_col = identifier["end_point"]
        if start_row < len(code_lines):
            line = code_lines[start_row]
            code_lines[start_row] = line[:start_col] + new_name + line[end_col:]
    return "\n".join(code_lines)


def rename_identifiers(src_bytes, renamed_id_dict, lang="cpp"):
    src_bytes = _ensure_bytes(src_bytes)
    parser = initialize_language_parser(lang)
    root_node = src2tree(parser, src_bytes)
    identifiers_info = []

    def find_old_ids(node):
        for child in node.children:
            if child.type in ["identifier", "field_identifier"]:
                id_name = src_bytes[child.start_byte : child.end_byte].decode("utf-8")
                if id_name in renamed_id_dict.keys():
                    identifiers_info.append(
                        {
                            "name": id_name,
                            "start_point": child.start_point,
                            "end_point": child.end_point,
                        }
                    )
            find_old_ids(child)

    find_old_ids(root_node)
    identifiers_info.sort(
        key=lambda x: (x["start_point"][0], x["start_point"][1]), reverse=True
    )
    code_lines = src_bytes.decode("utf-8").split("\n")
    for identifier in identifiers_info:
        start_row, start_col = identifier["start_point"]
        end_row, end_col = identifier["end_point"]
        if start_row < len(code_lines):
            line = code_lines[start_row]
            code_lines[start_row] = (
                line[:start_col] + renamed_id_dict[identifier["name"]] + line[end_col:]
            )
    return "\n".join(code_lines)
