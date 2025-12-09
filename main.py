from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union
from itertools import product

from z3 import Bool, And, Or, Not, Solver, sat


@dataclass
class Var:
    name: str


@dataclass
class NotNode:
    child: "Node"


@dataclass
class BinNode:
    op: str
    left: "Node"
    right: "Node"


Node = Union[Var, NotNode, BinNode]


def tokenize(expr: str) -> List[str]:
    specials = set("()!&|")
    tokens: List[str] = []
    cur = ""
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in specials:
            if cur:
                tokens.append(cur)
                cur = ""
            tokens.append(ch)
            i += 1
        elif ch.isspace():
            if cur:
                tokens.append(cur)
                cur = ""
            i += 1
        else:
            if i + 1 < len(expr) and expr[i:i + 2] == "->":
                if cur:
                    tokens.append(cur)
                    cur = ""
                tokens.append("->")
                i += 2
            elif i + 2 < len(expr) and expr[i:i + 3] == "<->":
                if cur:
                    tokens.append(cur)
                    cur = ""
                tokens.append("<->")
                i += 3
            else:
                cur += ch
                i += 1
    if cur:
        tokens.append(cur)
    return tokens


def parse(expr: str) -> Node:
    tokens = tokenize(expr)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def take(expected: str) -> None:
        nonlocal pos
        if pos < len(tokens) and tokens[pos] == expected:
            pos += 1
        else:
            raise SyntaxError(f"expected {expected}, got {peek()}")

    def parse_expr() -> Node:
        return parse_equiv()

    def parse_equiv() -> Node:
        nonlocal pos
        node = parse_impl()
        while peek() == "<->":
            pos += 1
            rhs = parse_impl()
            node = BinNode("<->", node, rhs)
        return node

    def parse_impl() -> Node:
        nonlocal pos
        node = parse_or()
        while peek() == "->":
            pos += 1
            rhs = parse_or()
            node = BinNode("->", node, rhs)
        return node

    def parse_or() -> Node:
        nonlocal pos
        node = parse_and()
        while peek() == "|":
            pos += 1
            rhs = parse_and()
            node = BinNode("|", node, rhs)
        return node

    def parse_and() -> Node:
        nonlocal pos
        node = parse_unary()
        while peek() == "&":
            pos += 1
            rhs = parse_unary()
            node = BinNode("&", node, rhs)
        return node

    def parse_unary() -> Node:
        nonlocal pos
        if peek() == "!":
            pos += 1
            child = parse_unary()
            return NotNode(child)
        return parse_atom()

    def parse_atom() -> Node:
        nonlocal pos
        tok = peek()
        if tok is None:
            raise SyntaxError("unexpected end of input")
        if tok == "(":
            pos += 1
            node = parse_expr()
            take(")")
            return node
        pos += 1
        return Var(tok)

    node = parse_expr()
    if pos != len(tokens):
        raise SyntaxError("extra tokens at end")
    return node


def ast_to_python(node: Node) -> str:
    if isinstance(node, Var):
        return node.name
    if isinstance(node, NotNode):
        return f"(not {ast_to_python(node.child)})"
    if isinstance(node, BinNode):
        left = ast_to_python(node.left)
        right = ast_to_python(node.right)
        if node.op == "&":
            return f"({left} and {right})"
        if node.op == "|":
            return f"({left} or {right})"
        if node.op == "->":
            return f"((not {left}) or {right})"
        if node.op == "<->":
            return f"(({left} and {right}) or ((not {left}) and (not {right})))"
        raise ValueError(f"unknown op {node.op}")
    raise TypeError("unknown AST node")


def ast_to_z3(node: Node, ctx_vars: Dict[str, Bool]):
    if isinstance(node, Var):
        if node.name not in ctx_vars:
            ctx_vars[node.name] = Bool(node.name)
        return ctx_vars[node.name]
    if isinstance(node, NotNode):
        return Not(ast_to_z3(node.child, ctx_vars))
    if isinstance(node, BinNode):
        l = ast_to_z3(node.left, ctx_vars)
        r = ast_to_z3(node.right, ctx_vars)
        if node.op == "&":
            return And(l, r)
        if node.op == "|":
            return Or(l, r)
        if node.op == "->":
            return Or(Not(l), r)
        if node.op == "<->":
            return Or(And(l, r), And(Not(l), Not(r)))
        raise ValueError(f"unknown op {node.op}")
    raise TypeError("unknown AST node")


def make_circuit_lambda(expr: str) -> Callable[[Dict[str, bool]], bool]:
    ast = parse(expr)
    expr_py = ast_to_python(ast)
    code = compile(expr_py, "", "eval")

    def circuit(assign: Dict[str, bool]) -> bool:
        return bool(eval(code, {"__builtins__": {}}, assign))

    return circuit


def extract_vars(expr: str) -> List[str]:
    vars_ = set()
    token = ""
    for ch in expr:
        if ch.isalnum() or ch == "_":
            token += ch
        else:
            if token.startswith("x") and token[1:].isdigit():
                vars_.add(token)
            token = ""
    if token and token.startswith("x") and token[1:].isdigit():
        vars_.add(token)
    return sorted(vars_, key=lambda s: int(s[1:]))


def check_equiv_bruteforce(f1_expr: str, f2_expr: str) -> Tuple[bool, float]:
    vars1 = extract_vars(f1_expr)
    vars2 = extract_vars(f2_expr)
    all_vars = sorted(set(vars1) | set(vars2), key=lambda s: int(s[1:]))

    f1 = make_circuit_lambda(f1_expr)
    f2 = make_circuit_lambda(f2_expr)

    start = time.perf_counter()

    if not all_vars:
        env: Dict[str, bool] = {}
        eq = (f1(env) == f2(env))
        dt = (time.perf_counter() - start) * 1000.0
        return eq, dt

    n = len(all_vars)
    if n > 25:
        raise RuntimeError("Too many inputs for brute force")

    for bits in product([False, True], repeat=n):
        env = {name: bit for name, bit in zip(all_vars, bits)}
        if f1(env) != f2(env):
            dt = (time.perf_counter() - start) * 1000.0
            return False, dt

    dt = (time.perf_counter() - start) * 1000.0
    return True, dt


def build_z3_circuit(expr: str, ctx_vars: Dict[str, Bool]):
    ast = parse(expr)
    return ast_to_z3(ast, ctx_vars)


def check_equiv_z3(f1_expr: str, f2_expr: str) -> Tuple[bool, float]:
    vars1 = extract_vars(f1_expr)
    vars2 = extract_vars(f2_expr)
    all_vars = sorted(set(vars1) | set(vars2), key=lambda s: int(s[1:]))

    ctx_vars: Dict[str, Bool] = {name: Bool(name) for name in all_vars}

    f1 = build_z3_circuit(f1_expr, ctx_vars)
    f2 = build_z3_circuit(f2_expr, ctx_vars)

    miter = Or(And(f1, Not(f2)), And(Not(f1), f2))

    s = Solver()
    s.add(miter)

    start = time.perf_counter()
    res = s.check()
    dt = (time.perf_counter() - start) * 1000.0

    return (res != sat), dt


def random_ast(depth: int, var_names: List[str]) -> Node:
    if depth <= 0:
        return Var(random.choice(var_names))
    if depth > 0 and random.random() < 0.25:
        return Var(random.choice(var_names))
    if random.random() < 0.3:
        return NotNode(random_ast(depth - 1, var_names))
    op = random.choice(["&", "|", "->", "<->"])
    left = random_ast(depth - 1, var_names)
    right = random_ast(depth - 1, var_names)
    return BinNode(op, left, right)


def ast_to_expr(node: Node) -> str:
    if isinstance(node, Var):
        return node.name
    if isinstance(node, NotNode):
        return "!(" + ast_to_expr(node.child) + ")"
    if isinstance(node, BinNode):
        return "(" + ast_to_expr(node.left) + " " + node.op + " " + ast_to_expr(node.right) + ")"
    raise TypeError("unknown AST node")


def ast_collect_vars(node: Node, out: set) -> None:
    if isinstance(node, Var):
        out.add(node.name)
    elif isinstance(node, NotNode):
        ast_collect_vars(node.child, out)
    elif isinstance(node, BinNode):
        ast_collect_vars(node.left, out)
        ast_collect_vars(node.right, out)


def random_formula(num_vars: int, depth: int, min_used_vars: int) -> str:
    vars_n = [f"x{i}" for i in range(num_vars)]
    min_used = max(1, min(min_used_vars, num_vars))
    best_ast = None
    best_cnt = 0
    for _ in range(50):
        ast = random_ast(depth, vars_n)
        used: set = set()
        ast_collect_vars(ast, used)
        cnt = len(used)
        if cnt >= min_used:
            return ast_to_expr(ast)
        if cnt > best_cnt:
            best_cnt = cnt
            best_ast = ast
    if best_ast is None:
        best_ast = random_ast(depth, vars_n)
    return ast_to_expr(best_ast)


def demo():
    tests = [
        ("x0", "x0", True),
        ("x0", "!x0", False),
        ("!x0", "!x0", True),
        ("x0 & x1", "x1 & x0", True),
        ("x0 | x1", "x1 | x0", True),
        ("(x0 & x1) & x2", "x0 & (x1 & x2)", True),
        ("(x0 | x1) | x2", "x0 | (x1 | x2)", True),
        ("x0 & (x1 | x2)", "(x0 & x1) | (x0 & x2)", True),
        ("x0 | (x1 & x2)", "(x0 | x1) & (x0 | x2)", True),
        ("!(x0 & x1)", "!x0 | !x1", True),
        ("!(x0 | x1)", "!x0 & !x1", True),
        ("x0 -> x1", "!x0 | x1", True),
        ("(x0 & x1) -> x0", "!(x0 & x1) | x0", True),
        ("x0 <-> x1", "(x0 & x1) | (!x0 & !x1)", True),
        ("(x0 <-> x1) & x2", "((x0 & x1) | (!x0 & !x1)) & x2", True),
        ("x0 & x1", "x0 | x1", False),
        ("x0 | !x0", "x1 | !x1", True),
        ("(x0 & !x0)", "(x1 & !x1)", True),
        ("(x0 & x1) | (x0 & !x1)", "x0", True),
        ("(x0 | x1) & (!x0 | x1)", "x1", True),
        ("(x0 & x1) | (!x0 & x1)", "x1", True),
        ("(x0 & x1) | (x0 & x2)", "x0 & (x1 | x2)", True),
        ("(x0 | x1) & (!x0 | !x1)", "(x0 & !x1) | (!x0 & x1)", True),
    ]

    for _ in range(5):
        f = random_formula(8, 4, min_used_vars=4)
        tests.append((f, f, True))

    for _ in range(5):
        f = random_formula(12, 6, min_used_vars=6)
        g = "!(" + f + ")"
        tests.append((f, g, False))

    for n in [10, 15, 20]:
        vars_n = [f"x{i}" for i in range(n)]
        f1 = " & ".join(vars_n)
        f2 = " & ".join(reversed(vars_n))
        tests.append((f1, f2, True))

    passed = 0
    failed = 0
    total_sat_time = 0.0
    total_brute_time = 0.0

    for i, (f1, f2, exp) in enumerate(tests, start=1):
        print(f"\n[Test {i}]")
        print(" F1:", f1)
        print(" F2:", f2)
        eq_brute, t_brute = check_equiv_bruteforce(f1, f2)
        eq_z3, t_z3 = check_equiv_z3(f1, f2)
        total_brute_time += t_brute
        total_sat_time += t_z3
        same = (eq_brute == eq_z3)
        if same:
            passed += 1
        else:
            failed += 1
        print(" expected:", "equiv" if exp else "diff")
        print(" brute: ", "equiv" if eq_brute else "diff",
              f"({t_brute:.3f} ms)")
        print(" Z3:   ", "equiv" if eq_z3 else "diff",
              f"({t_z3:.3f} ms)")

    total = passed + failed
    rate = (passed * 100.0 / total) if total else 0.0

    print("\nPassed:", passed, "Failed:", failed)
    print(f"Rate: {rate:.1f}%")
    print(f"Total SAT Time: {total_sat_time:.3f} ms")
    print(f"Total Brute Time: {total_brute_time:.3f} ms")


if __name__ == "__main__":
    demo()
