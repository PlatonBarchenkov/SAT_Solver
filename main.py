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
    if n > 21:
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


CNFClause = List[int]
CNF = List[CNFClause]


def normalize_ast(node: Node) -> Node:
    if isinstance(node, Var):
        return node
    if isinstance(node, NotNode):
        return NotNode(normalize_ast(node.child))
    if isinstance(node, BinNode):
        left = normalize_ast(node.left)
        right = normalize_ast(node.right)
        if node.op == "->":
            return BinNode("|", NotNode(left), right)
        if node.op == "<->":
            a_impl_b = BinNode("|", NotNode(left), right)
            b_impl_a = BinNode("|", NotNode(right), left)
            return BinNode("&", normalize_ast(a_impl_b), normalize_ast(b_impl_a))
        if node.op in ("&", "|"):
            return BinNode(node.op, left, right)
        raise ValueError(f"unknown op in normalize_ast: {node.op}")
    raise TypeError("unknown AST node in normalize_ast")


def tseitin_cnf(node: Node) -> Tuple[CNF, Dict[str, int], int]:
    node = normalize_ast(node)
    cnf: CNF = []
    var_map: Dict[str, int] = {}
    next_var = 1

    def encode(n: Node) -> int:
        nonlocal next_var
        if isinstance(n, Var):
            if n.name not in var_map:
                var_map[n.name] = next_var
                next_var += 1
            return var_map[n.name]
        if isinstance(n, NotNode):
            p = encode(n.child)
            v = next_var
            next_var += 1
            cnf.append([-v, -p])
            cnf.append([v, p])
            return v
        if isinstance(n, BinNode):
            if n.op not in ("&", "|"):
                raise ValueError("tseitin_cnf expects only &, |, ! after normalize")
            a = encode(n.left)
            b = encode(n.right)
            v = next_var
            next_var += 1
            if n.op == "&":
                cnf.append([-v, a])
                cnf.append([-v, b])
                cnf.append([v, -a, -b])
            else:
                cnf.append([v, -a])
                cnf.append([v, -b])
                cnf.append([-v, a, b])
            return v
        raise TypeError("unknown AST node in tseitin_cnf")

    top = encode(node)
    cnf.append([top])
    return cnf, var_map, next_var - 1


def _simplify_clause(clause: CNFClause, assignment: Dict[int, bool]) -> Tuple[str, CNFClause]:
    new_clause: CNFClause = []
    for lit in clause:
        var = abs(lit)
        val = assignment.get(var)
        if val is None:
            new_clause.append(lit)
        else:
            if lit > 0:
                if val is True:
                    return "sat", []
            else:
                if val is False:
                    return "sat", []
    if not new_clause:
        return "conflict", []
    return "clause", new_clause


def unit_propagate(cnf: CNF, assignment: Dict[int, bool]) -> Tuple[CNF, Dict[int, bool], bool]:
    cnf = [c[:] for c in cnf]
    changed = True
    while changed:
        changed = False
        unit_literals: List[int] = []
        new_cnf: CNF = []
        for clause in cnf:
            status, simplified = _simplify_clause(clause, assignment)
            if status == "sat":
                continue
            if status == "conflict":
                return [], assignment, True
            if len(simplified) == 1:
                unit_literals.append(simplified[0])
            new_cnf.append(simplified)
        for lit in unit_literals:
            var = abs(lit)
            val = lit > 0
            cur = assignment.get(var)
            if cur is None:
                assignment[var] = val
                changed = True
            elif cur != val:
                return [], assignment, True
        cnf = new_cnf
    return cnf, assignment, False


def pure_literal_elimination(cnf: CNF, assignment: Dict[int, bool]) -> Tuple[CNF, Dict[int, bool]]:
    pos: Dict[int, bool] = {}
    neg: Dict[int, bool] = {}
    for clause in cnf:
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                continue
            if lit > 0:
                pos[var] = True
            else:
                neg[var] = True
    pure_assignments: Dict[int, bool] = {}
    for var in set(list(pos.keys()) + list(neg.keys())):
        if var in assignment:
            continue
        has_pos = var in pos
        has_neg = var in neg
        if has_pos and not has_neg:
            pure_assignments[var] = True
        elif has_neg and not has_pos:
            pure_assignments[var] = False
    if not pure_assignments:
        return cnf, assignment
    assignment.update(pure_assignments)
    new_cnf: CNF = []
    for clause in cnf:
        satisfied = False
        for lit in clause:
            var = abs(lit)
            if var in pure_assignments:
                val = pure_assignments[var]
                if (lit > 0 and val is True) or (lit < 0 and val is False):
                    satisfied = True
                    break
        if not satisfied:
            new_cnf.append(clause)
    return new_cnf, assignment


def preprocess_cnf(cnf: CNF) -> Tuple[CNF, Dict[int, bool], bool]:
    assignment: Dict[int, bool] = {}
    cnf, assignment, conflict = unit_propagate(cnf, assignment)
    if conflict:
        return [], assignment, True
    changed = True
    while changed:
        changed = False
        cnf_before = cnf
        assignment_before = dict(assignment)
        cnf, assignment = pure_literal_elimination(cnf, assignment)
        cnf, assignment, conflict = unit_propagate(cnf, assignment)
        if conflict:
            return [], assignment, True
        if cnf != cnf_before or assignment != assignment_before:
            changed = True
    return cnf, assignment, False


def solve_cnf_with_z3(cnf: CNF, assignment: Dict[int, bool]):
    vars_in_cnf = {abs(lit) for clause in cnf for lit in clause}
    vars_in_assign = set(assignment.keys())
    all_vars = sorted(vars_in_cnf | vars_in_assign)
    z3_vars = {idx: Bool(f"v{idx}") for idx in all_vars}
    s = Solver()
    for var, val in assignment.items():
        lit = z3_vars[var]
        s.add(lit if val else Not(lit))
    for clause in cnf:
        if not clause:
            s.add(False)
            break
        z3_lits = []
        for lit in clause:
            var = abs(lit)
            z = z3_vars[var]
            z3_lits.append(z if lit > 0 else Not(z))
        s.add(Or(*z3_lits))
    return s.check()


def check_equiv_z3_preprocessed(f1_expr: str, f2_expr: str) -> Tuple[bool, float]:
    ast1 = parse(f1_expr)
    ast2 = parse(f2_expr)
    miter_ast = BinNode(
        "|",
        BinNode("&", ast1, NotNode(ast2)),
        BinNode("&", NotNode(ast1), ast2),
    )
    cnf, var_map, max_var = tseitin_cnf(miter_ast)
    start = time.perf_counter()
    cnf_simpl, assignment, conflict = preprocess_cnf(cnf)
    if conflict:
        dt = (time.perf_counter() - start) * 1000.0
        return True, dt
    res = solve_cnf_with_z3(cnf_simpl, assignment)
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
    total_sat_pre_time = 0.0
    total_brute_time = 0.0
    for i, (f1, f2, exp) in enumerate(tests, start=1):
        print(f"\n[Test {i}]")
        print(" F1:", f1)
        print(" F2:", f2)
        eq_brute, t_brute = check_equiv_bruteforce(f1, f2)
        eq_z3, t_z3 = check_equiv_z3(f1, f2)
        eq_z3p, t_z3p = check_equiv_z3_preprocessed(f1, f2)
        total_brute_time += t_brute
        total_sat_time += t_z3
        total_sat_pre_time += t_z3p
        same = (eq_brute == eq_z3 == eq_z3p)
        if same:
            passed += 1
        else:
            failed += 1
        print(" expected:", "equiv" if exp else "diff")
        print(" brute: ", "equiv" if eq_brute else "diff", f"({t_brute:.3f} ms)")
        print(" Z3:    ", "equiv" if eq_z3 else "diff", f"({t_z3:.3f} ms)")
        print(" Z3+pre:", "equiv" if eq_z3p else "diff", f"({t_z3p:.3f} ms)")
    total = passed + failed
    rate = (passed * 100.0 / total) if total else 0.0
    print("\nPassed:", passed, "Failed:", failed)
    print(f"Rate: {rate:.1f}%")
    print(f"Total SAT Time: {total_sat_time:.3f} ms")
    print(f"Total SAT+pre Time: {total_sat_pre_time:.3f} ms")
    print(f"Total Brute Time: {total_brute_time:.3f} ms")


if __name__ == "__main__":
    demo()
