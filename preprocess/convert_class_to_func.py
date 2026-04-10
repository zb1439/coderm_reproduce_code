#!/usr/bin/env python3
"""Convert class Solution format to standalone function format + wildcard imports header.

Reads a func.jsonl with class Solution methods, outputs a new func.jsonl
with standalone functions matching coderm's expected format.
"""
import ast
import json
import re
import sys
import textwrap
from pathlib import Path

IMPORT_HEADER = """\
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6*10**5)
"""


def convert_solution(code: str) -> str:
    """Convert a class Solution method to standalone function."""
    code = code.strip()
    
    # If it's already a standalone function (no class Solution), just add header
    if "class Solution" not in code:
        return IMPORT_HEADER + "\n" + code
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Can't parse, return as-is with header
        return IMPORT_HEADER + "\n" + code
    
    # Find the Solution class
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            # Collect all methods
            functions = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Remove 'self' from args
                    if item.args.args and item.args.args[0].arg == "self":
                        item.args.args.pop(0)
                    functions.append(item)
            
            if not functions:
                return IMPORT_HEADER + "\n" + code
            
            # Replace self.method() calls with method() calls
            for func in functions:
                _remove_self_refs(func)
            
            # Build module with standalone functions + any non-class top-level code
            new_body = []
            for node2 in tree.body:
                if isinstance(node2, ast.ClassDef) and node2.name == "Solution":
                    new_body.extend(functions)
                else:
                    new_body.append(node2)
            
            module = ast.Module(body=new_body, type_ignores=[])
            ast.fix_missing_locations(module)
            result = ast.unparse(module)
            return IMPORT_HEADER + "\n" + result
    
    # No Solution class found, return with header
    return IMPORT_HEADER + "\n" + code


def _remove_self_refs(node):
    """Replace self.xxx with xxx in AST."""
    for child in ast.walk(node):
        # self.method(...) -> method(...)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if isinstance(child.func.value, ast.Name) and child.func.value.id == "self":
                child.func = ast.Name(id=child.func.attr, ctx=ast.Load())
        # self.var -> var (for ast.Attribute nodes that aren't calls)
        if isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name) and child.value.id == "self":
                # We need to replace this node in its parent
                pass  # handled by the parent replacement below
    
    # Second pass: replace self.xxx attribute access in assignments etc
    for child in ast.walk(node):
        for field, value in ast.iter_fields(child):
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self":
                setattr(child, field, ast.Name(id=value.attr, ctx=value.ctx))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id == "self":
                        value[i] = ast.Name(id=item.attr, ctx=item.ctx)


def convert_file(input_path: str, output_path: str):
    with open(input_path) as f:
        rows = [json.loads(l) for l in f]
    
    converted = 0
    skipped = 0
    output_rows = []
    
    for row in rows:
        new_solutions = []
        for sol in row["solutions"]:
            try:
                new_sol = convert_solution(sol)
                new_solutions.append(new_sol)
                converted += 1
            except Exception as e:
                skipped += 1
        output_rows.append({"task_id": row["task_id"], "solutions": new_solutions})
    
    with open(output_path, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"Converted: {converted}, Skipped: {skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_file(args.input, args.output)
