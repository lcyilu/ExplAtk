from __future__ import annotations

from tree_sitter import Language, Parser

from common.config_loader import get_settings
from common.utils.keywords import (
    __builtin__funcs__,
    __key_words__,
    __macros__,
    __other__keywords__,
    __special_ids__,
)

list_all_keywords = [
    __key_words__,
    __macros__,
    __special_ids__,
    __builtin__funcs__,
    __other__keywords__,
]
__all__keywords__ = frozenset().union(*list_all_keywords)


def initialize_language_parser(lang: str | None = None):
    s = get_settings()
    lang = lang or s.default_language
    language = Language(s.language_so_path, lang)
    parser = Parser()
    parser.set_language(language)
    return parser


def src2tree(parser, src_bytes):
    tree = parser.parse(src_bytes)
    return tree.root_node


def extract_identifiers_from_one_src(src_bytes, lang: str | None = None):
    parser = initialize_language_parser(lang)
    root_node = src2tree(parser, src_bytes)
    identifiers = []
    identifier_candidates = ["identifier", "field_identifier"]

    def extract_identifiers(node):
        for child in node.children:
            if child.type in identifier_candidates:
                id_name = src_bytes[child.start_byte : child.end_byte].decode("utf-8")
                if is_not_keyword(id_name):
                    identifiers.append(id_name)
            extract_identifiers(child)

    extract_identifiers(root_node)
    return identifiers


def is_not_keyword(identifier: str) -> bool:
    return len({identifier}.difference(__all__keywords__)) != 0


def get_all_tokens(node, source_bytes):
    tokens = []
    if node.type in ["comment", "string_literal", "char_literal", "preproc_include"]:
        return []
    if len(node.children) == 0:
        token_text = source_bytes[node.start_byte : node.end_byte].decode("utf8")
        return [(token_text, node.start_byte, node.end_byte, node.type)]
    for child in node.children:
        tokens.extend(get_all_tokens(child, source_bytes))
    return tokens


def get_all_tokens_from_one_src(source_bytes, lang):
    parser = initialize_language_parser(lang)
    root_node = src2tree(parser, source_bytes)
    return get_all_tokens(root_node, source_bytes)
