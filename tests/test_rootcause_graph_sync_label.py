"""Root Cause 验证测试: novel 数据集 FalkorDB Edges 为 0

Root Cause: graph_sync.py 中 entity_type 只做了 space→underscore 替换，
未清理括号/冒号/斜杠等 Cypher label 非法字符。novel 数据集中存在
type='WORK OF LITERATURE (IMPLIED ENTITY TYPE: CULTURAL PRACTICE)' 的实体，
转换后含 '(' 导致 Cypher 语法错误，异常未在循环内捕获，整个 sync 中断。
"""
import re
import pytest


def sanitize_label_original(entity_type: str) -> str:
    """graph_sync.py 原始实现"""
    return entity_type.replace(" ", "_")


def sanitize_label_fixed(entity_type: str) -> str:
    """修复后：只保留字母、数字、下划线"""
    return re.sub(r"[^A-Za-z0-9_]", "_", entity_type.replace(" ", "_"))


PROBLEMATIC_TYPES = [
    "WORK OF LITERATURE (IMPLIED ENTITY TYPE: CULTURAL PRACTICE)",
    "ANIMAL/CREATURE",
    "SYMBOL/CULTURAL ICON",
    "ABSTRACTION / CULTURAL PRACTICE",
    "DISEASE/CULTURAL PRACTICE",
    "VESSEL/TRANSPORT",
    "ORGANIZATION/TECHNOLOGY",
    "ANIMAL/REGIONAL SYMBOL",
]


class TestRootCause:
    def test_reproduce_issue(self):
        """原始 sanitize 产生含非法字符的 Cypher label"""
        for t in PROBLEMATIC_TYPES:
            label = sanitize_label_original(t)
            assert re.search(r"[():/]", label), f"Expected illegal chars in '{label}'"

    def test_fix_resolves_issue(self):
        """修复后所有 label 只含合法字符"""
        for t in PROBLEMATIC_TYPES:
            label = sanitize_label_fixed(t)
            assert re.match(r"^[A-Za-z0-9_]+$", label), f"Illegal chars in '{label}'"

    def test_normal_types_unaffected(self):
        """正常类型（如 medical 数据集）修复前后结果一致"""
        normal = ["DISEASE", "BODY_PART", "PERSON", "LOCATION"]
        for t in normal:
            assert sanitize_label_original(t) == sanitize_label_fixed(t)
