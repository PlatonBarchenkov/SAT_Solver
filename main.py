from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union, Optional, Set
from itertools import product
from z3 import Bool, And, Or, Not, Solver, sat


@dataclass
class Var:
    name: str


@dataclass
class NotNode:
    child: 'Node'


@dataclass
class BinNode:
    op: str
    left: 'Node'
    right: 'Node'


Node = Union[Var, NotNode, BinNode]


def tokenize(expr: str) -> List[str]:
    specials = set('()!&|')
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
            if i + 2 <= len(expr) and expr[i:i + 2] == '->':
                if cur:
                    tokens.append(cur)
                    cur = ""
                tokens.append('->')
                i += 2
            elif i + 3 <= len(expr) and expr[i:i + 3] == '<->':
                if cur:
                    tokens.append(cur)
                    cur = ""
                tokens.append('<->')
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
            raise SyntaxError(f"expected '{expected}', got {peek()}")

    def parse_expr() -> Node:
        return parse_equiv()

    def parse_equiv() -> Node:
        nonlocal pos
        node = parse_impl()
        while peek() == '<->':
            pos += 1
            rhs = parse_impl()
            node = BinNode('<->', node, rhs)
        return node

    def parse_impl() -> Node:
        nonlocal pos
        node = parse_or()
        while peek() == '->':
            pos += 1
            rhs = parse_or()
            node = BinNode('->', node, rhs)
        return node

    def parse_or() -> Node:
        nonlocal pos
        node = parse_and()
        while peek() == '|':
            pos += 1
            rhs = parse_and()
            node = BinNode('|', node, rhs)
        return node

    def parse_and() -> Node:
        nonlocal pos
        node = parse_unary()
        while peek() == '&':
            pos += 1
            rhs = parse_unary()
            node = BinNode('&', node, rhs)
        return node

    def parse_unary() -> Node:
        nonlocal pos
        if peek() == '!':
            pos += 1
            child = parse_unary()
            return NotNode(child)
        return parse_atom()

    def parse_atom() -> Node:
        nonlocal pos
        tok = peek()
        if tok is None:
            raise SyntaxError("unexpected end of input")
        if tok == '(':
            pos += 1
            node = parse_expr()
            take(')')
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
        if node.op == '&':
            return f"({left} and {right})"
        if node.op == '|':
            return f"({left} or {right})"
        if node.op == '->':
            return f"(not {left} or {right})"
        if node.op == '<->':
            return f"(({left} and {right}) or (not {left} and not {right}))"
    raise ValueError(f"unknown op: {node.op}")
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
        if node.op == '&':
            return And(l, r)
        if node.op == '|':
            return Or(l, r)
        if node.op == '->':
            return Or(Not(l), r)
        if node.op == '<->':
            return Or(And(l, r), And(Not(l), Not(r)))
    raise ValueError(f"unknown op: {node.op}")
    raise TypeError("unknown AST node")


def make_circuit_lambda(expr: str) -> Callable[[Dict[str, bool]], bool]:
    ast = parse(expr)
    expr_py = ast_to_python(ast)
    code = compile(expr_py, "<string>", "eval")

    def circuit(assign: Dict[str, bool]) -> bool:
        return bool(eval(code, {"__builtins__": {}}, assign))

    return circuit


def extract_vars(expr: str) -> List[str]:
    vars = set()
    token = ""
    for ch in (expr + " "):
        if ch.isalnum() or ch == '_':
            token += ch
        else:
            if token and token.startswith('x') and token[1:].isdigit():
                vars.add(token)
            token = ""
    if token and token.startswith('x') and token[1:].isdigit():
        vars.add(token)
    return sorted(vars, key=lambda s: int(s[1:]))


@dataclass(frozen=True)
class Literal:
    variable: str
    negation: bool

    def __repr__(self):
        return ('¬' if self.negation else '') + self.variable

    def neg(self) -> 'Literal':
        return Literal(self.variable, not self.negation)


@dataclass
class Clause:
    literals: List[Literal]

    def __repr__(self):
        return '∨'.join(map(str, self.literals))

    def __iter__(self):
        return iter(self.literals)

    def __len__(self):
        return len(self.literals)


@dataclass
class Formula:
    clauses: List[Clause]
    __variables: Set[str]

    def __init__(self, clauses: List[Clause]):
        self.clauses = []
        self.__variables = set()
        for clause in clauses:
            self.clauses.append(Clause(list(set(clause.literals))))
            for lit in clause.literals:
                self.__variables.add(lit.variable)

    def variables(self) -> Set[str]:
        return self.__variables

    def __repr__(self):
        return ' ∧ '.join(f'({clause})' for clause in self.clauses)

    def __iter__(self):
        return iter(self.clauses)

    def __len__(self):
        return len(self.clauses)


def ast_to_cnf(node: Node) -> Formula:
    clauses: List[Clause] = []
    aux_counter = [0]

    def new_aux() -> str:
        aux_counter[0] += 1
        return f"_aux{aux_counter[0]}"

    def process_node(n: Node) -> str:
        if isinstance(n, Var):
            return n.name

        if isinstance(n, NotNode):
            child_var = process_node(n.child)
            aux = new_aux()
            clauses.append(Clause([Literal(aux, False), Literal(child_var, False)]))
            clauses.append(Clause([Literal(aux, True), Literal(child_var, True)]))
            return aux

        if isinstance(n, BinNode):
            left_var = process_node(n.left)
            right_var = process_node(n.right)
            aux = new_aux()

            if n.op == '&':
                clauses.append(Clause([Literal(aux, True), Literal(left_var, False)]))
                clauses.append(Clause([Literal(aux, True), Literal(right_var, False)]))
                clauses.append(Clause([Literal(aux, False), Literal(left_var, True), Literal(right_var, True)]))

            elif n.op == '|':
                clauses.append(Clause([Literal(aux, True), Literal(left_var, False), Literal(right_var, False)]))
                clauses.append(Clause([Literal(aux, False), Literal(left_var, True)]))
                clauses.append(Clause([Literal(aux, False), Literal(right_var, True)]))

            elif n.op == '->':
                clauses.append(Clause([Literal(aux, True), Literal(left_var, True), Literal(right_var, False)]))
                clauses.append(Clause([Literal(aux, False), Literal(left_var, False)]))
                clauses.append(Clause([Literal(aux, False), Literal(right_var, True)]))

            elif n.op == '<->':
                clauses.append(Clause([Literal(aux, True), Literal(left_var, True), Literal(right_var, False)]))
                clauses.append(Clause([Literal(aux, True), Literal(left_var, False), Literal(right_var, True)]))
                clauses.append(Clause([Literal(aux, False), Literal(left_var, False), Literal(right_var, False)]))
                clauses.append(Clause([Literal(aux, False), Literal(left_var, True), Literal(right_var, True)]))

            return aux

        raise TypeError("Unknown node type")

    root_var = process_node(node)
    clauses.append(Clause([Literal(root_var, False)]))

    return Formula(clauses)


@dataclass
class DPLLAssignment:
    value: bool
    decision_level: int


class DPLLAssignments(dict):
    def __init__(self):
        super().__init__()
        self.dl = 0

    def value(self, literal: Literal) -> bool:
        if literal.negation:
            return not self[literal.variable].value
        else:
            return self[literal.variable].value

    def assign(self, variable: str, value: bool, dl: int):
        self[variable] = DPLLAssignment(value, dl)


def dpll_unit_propagation(formula: Formula, assignments: DPLLAssignments) -> Tuple[bool, Optional[Clause]]:
    changed = True
    while changed:
        changed = False
        for clause in formula:
            values = []
            unassigned_lit = None

            for lit in clause:
                if lit.variable not in assignments:
                    if unassigned_lit is None:
                        unassigned_lit = lit
                    values.append(None)
                else:
                    val = assignments.value(lit)
                    values.append(val)
                    if val:
                        break
            else:
                if True not in values:
                    if values.count(None) == 0:
                        return True, clause
                    elif values.count(None) == 1:
                        var = unassigned_lit.variable
                        val = not unassigned_lit.negation
                        assignments.assign(var, val, assignments.dl)
                        changed = True

    return False, None


def dpll_solve_cnf(formula: Formula) -> Optional[DPLLAssignments]:
    assignments = DPLLAssignments()

    conflict, _ = dpll_unit_propagation(formula, assignments)
    if conflict:
        return None

    decision_stack = []

    while True:
        if len(assignments) == len(formula.variables()):
            return assignments

        unassigned = [v for v in formula.variables() if v not in assignments]
        if not unassigned:
            return assignments

        var = unassigned[0]

        assignments.dl += 1
        assignments.assign(var, True, assignments.dl)
        decision_stack.append((var, True, assignments.dl, dict(assignments)))

        conflict, _ = dpll_unit_propagation(formula, assignments)

        if not conflict:
            continue

        while decision_stack:
            last_var, last_val, last_dl, saved_state = decision_stack.pop()

            to_remove = [v for v in assignments if assignments[v].decision_level >= last_dl]
            for v in to_remove:
                del assignments[v]

            if last_val:
                assignments.dl = last_dl
                assignments.assign(last_var, False, last_dl)
                decision_stack.append((last_var, False, last_dl, None))

                conflict, _ = dpll_unit_propagation(formula, assignments)
                if not conflict:
                    break
        else:
            return None


@dataclass
class CDCLAssignment:
    value: bool
    antecedent: Optional[Clause]
    dl: int


class CDCLAssignments(dict):
    def __init__(self):
        super().__init__()
        self.dl = 0

    def value(self, literal: Literal) -> bool:
        if literal.negation:
            return not self[literal.variable].value
        else:
            return self[literal.variable].value

    def assign(self, variable: str, value: bool, antecedent: Optional[Clause]):
        self[variable] = CDCLAssignment(value, antecedent, self.dl)


def cdcl_clause_status(clause: Clause, assignments: CDCLAssignments) -> str:
    values = []
    for literal in clause:
        if literal.variable not in assignments:
            values.append(None)
        else:
            values.append(assignments.value(literal))

    if True in values:
        return 'satisfied'
    elif values.count(False) == len(values):
        return 'unsatisfied'
    elif values.count(False) == len(values) - 1:
        return 'unit'
    else:
        return 'unresolved'


def cdcl_unit_propagation(formula: Formula, assignments: CDCLAssignments) -> Tuple[str, Optional[Clause]]:
    finish = False
    while not finish:
        finish = True
        for clause in formula:
            status = cdcl_clause_status(clause, assignments)

            if status in ('unresolved', 'satisfied'):
                continue
            elif status == 'unit':
                literal = next(lit for lit in clause if lit.variable not in assignments)
                var = literal.variable
                val = not literal.negation
                assignments.assign(var, val, antecedent=clause)
                finish = False
            else:
                return ('conflict', clause)

    return ('unresolved', None)


def cdcl_resolve(a: Clause, b: Clause, x: str) -> Clause:
    result = set(a.literals + b.literals) - {Literal(x, True), Literal(x, False)}
    return Clause(list(result))


def cdcl_conflict_analysis(clause: Clause, assignments: CDCLAssignments) -> Tuple[int, Clause]:
    if assignments.dl == 0:
        return (-1, None)

    literals = [lit for lit in clause if lit.variable in assignments and assignments[lit.variable].dl == assignments.dl]

    while len(literals) != 1:
        implied = [lit for lit in literals if assignments[lit.variable].antecedent is not None]
        if not implied:
            break

        literal = implied[0]
        antecedent = assignments[literal.variable].antecedent
        clause = cdcl_resolve(clause, antecedent, literal.variable)

        literals = [lit for lit in clause if
                    lit.variable in assignments and assignments[lit.variable].dl == assignments.dl]

    decision_levels = sorted(set(assignments[lit.variable].dl for lit in clause if lit.variable in assignments))

    if len(decision_levels) <= 1:
        return 0, clause
    else:
        return decision_levels[-2], clause


def cdcl_backtrack(assignments: CDCLAssignments, b: int):
    to_remove = [var for var, assn in assignments.items() if assn.dl > b]
    for var in to_remove:
        assignments.pop(var)


def cdcl_solve_cnf(formula: Formula) -> Optional[CDCLAssignments]:
    assignments = CDCLAssignments()

    reason, clause = cdcl_unit_propagation(formula, assignments)
    if reason == 'conflict':
        return None

    while len(assignments) < len(formula.variables()):
        unassigned = [v for v in formula.variables() if v not in assignments]
        if not unassigned:
            break

        var = unassigned[0]
        val = True

        assignments.dl += 1
        assignments.assign(var, val, antecedent=None)

        while True:
            reason, clause = cdcl_unit_propagation(formula, assignments)

            if reason != 'conflict':
                break

            b, learnt_clause = cdcl_conflict_analysis(clause, assignments)

            if b < 0:
                return None

            formula.clauses.append(learnt_clause)
            cdcl_backtrack(assignments, b)
            assignments.dl = b

    return assignments


def check_equiv_bruteforce(f1_expr: str, f2_expr: str) -> Tuple[Optional[bool], float]:
    vars1 = extract_vars(f1_expr)
    vars2 = extract_vars(f2_expr)
    all_vars = sorted(set(vars1 + vars2), key=lambda s: int(s[1:]))

    n = len(all_vars)
    if n > 25:
        return None, 0.0

    f1 = make_circuit_lambda(f1_expr)
    f2 = make_circuit_lambda(f2_expr)

    start = time.perf_counter()

    if not all_vars:
        env: Dict[str, bool] = {}
        eq = (f1(env) == f2(env))
        dt = (time.perf_counter() - start) * 1000.0
        return eq, dt

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
    all_vars = sorted(set(vars1 + vars2), key=lambda s: int(s[1:]))

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


def check_equiv_dpll(f1_expr: str, f2_expr: str) -> Tuple[bool, float]:
    try:
        vars1 = extract_vars(f1_expr)
        vars2 = extract_vars(f2_expr)
        all_vars = sorted(set(vars1 + vars2), key=lambda s: int(s[1:]))

        ast1 = parse(f1_expr)
        ast2 = parse(f2_expr)

        miter_ast = BinNode('|',
                            BinNode('&', ast1, NotNode(ast2)),
                            BinNode('&', NotNode(ast1), ast2))

        cnf = ast_to_cnf(miter_ast)

        start = time.perf_counter()
        result = dpll_solve_cnf(cnf)
        dt = (time.perf_counter() - start) * 1000.0

        return (result is None), dt
    except Exception:
        return False, 0.0


def check_equiv_cdcl(f1_expr: str, f2_expr: str) -> Tuple[bool, float]:
    try:
        vars1 = extract_vars(f1_expr)
        vars2 = extract_vars(f2_expr)
        all_vars = sorted(set(vars1 + vars2), key=lambda s: int(s[1:]))

        ast1 = parse(f1_expr)
        ast2 = parse(f2_expr)

        miter_ast = BinNode('|',
                            BinNode('&', ast1, NotNode(ast2)),
                            BinNode('&', NotNode(ast1), ast2))

        cnf = ast_to_cnf(miter_ast)

        start = time.perf_counter()
        result = cdcl_solve_cnf(cnf)
        dt = (time.perf_counter() - start) * 1000.0

        return (result is None), dt
    except Exception:
        return False, 0.0


def random_ast(depth: int, var_names: List[str]) -> Node:
    if depth == 0:
        return Var(random.choice(var_names))

    if depth > 0 and random.random() < 0.25:
        return Var(random.choice(var_names))

    if random.random() < 0.3:
        return NotNode(random_ast(depth - 1, var_names))

    op = random.choice(['&', '|', '->', '<->'])
    left = random_ast(depth - 1, var_names)
    right = random_ast(depth - 1, var_names)
    return BinNode(op, left, right)


def ast_to_expr(node: Node) -> str:
    if isinstance(node, Var):
        return node.name
    if isinstance(node, NotNode):
        return f"!{ast_to_expr(node.child)}"
    if isinstance(node, BinNode):
        return f"({ast_to_expr(node.left)} {node.op} {ast_to_expr(node.right)})"
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
        used = set()
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
        ("x0 & x1 & x2", "x0 & x1 & x2", True),
        ("x0 | x1 | x2", "x0 | x1 | x2", True),
        ("x0 & x1 & x2", "(x0 & x1) & x2", True),
        ("x0 | x1 | x2", "(x0 | x1) | x2", True),
        ("!x0 & x1", "!x0 & !x1", True),
        ("!x0 | x1", "!x0 | !x1", True),
        ("x0 -> x1", "!x0 | x1", True),
        ("(x0 & x1) -> x0", "!x0 | x1 | x0", True),
        ("x0 <-> x1", "(x0 & x1) | (!x0 & !x1)", True),
        ("x0 <-> (x1 & x2)", "(x0 & x1 & !x0 & !x1) | x2", True),
        ("x0 & x1", "x0 | x1", False),
        ("x0 & !x0", "x1 & !x1", True),
        ("x0 | !x0", "x1 | !x1", True),
        ("(x0 & x1) | (x0 & !x1)", "x0", True),
        ("(x0 & x1) | (!x0 & x1)", "x1", True),
        ("(x0 | x1) & (!x0 | x1)", "x1", True),
        ("(x0 & x1) | (x0 & x2)", "(x0 & x1) | x2", True),
        ("(x0 & x1) | (!x0 & !x1)", "(x0 & !x1) | (!x0 & x1)", True),
    ]

    for _ in range(5):
        f = random_formula(8, 4, min_used_vars=4)
        tests.append((f, f, True))

    for _ in range(5):
        f = random_formula(12, 6, min_used_vars=6)
        g = "!(" + f + ")"
        tests.append((f, g, False))

    vars_n = [f"x{i}" for i in range(20)]
    f1 = " & ".join(vars_n)
    f2 = " & ".join(reversed(vars_n))
    tests.append((f1, f2, True))

    for n in [30, 60, 100]:
        vars_n = [f"x{i}" for i in range(n)]
        f1 = " & ".join(vars_n)
        f2 = " & ".join(reversed(vars_n))
        tests.append((f1, f2, True))

    for _ in range(3):
        f = random_formula(30, 4, min_used_vars=26)
        tests.append((f, f, True))

    passed = 0
    failed = 0
    total_sat_time = 0.0
    total_brute_time = 0.0
    total_dpll_time = 0.0
    total_cdcl_time = 0.0

    print(f"Running {len(tests)} tests...")

    for i, (f1, f2, exp) in enumerate(tests, start=1):
        vars1 = extract_vars(f1)
        n_vars = len(vars1)

        print(f"\n=== Test {i} ===")
        if n_vars <= 15:
            print(f"F1: {f1}")
            print(f"F2: {f2}")
        else:
            print(f"Large Formula: {n_vars} variables")

        eq_brute, t_brute = check_equiv_bruteforce(f1, f2)
        eq_z3, t_z3 = check_equiv_z3(f1, f2)
        eq_dpll, t_dpll = check_equiv_dpll(f1, f2)
        eq_cdcl, t_cdcl = check_equiv_cdcl(f1, f2)

        if eq_brute is not None:
            total_brute_time += t_brute

        total_sat_time += t_z3
        total_dpll_time += t_dpll
        total_cdcl_time += t_cdcl

        if eq_brute is not None:
            same = (eq_brute == eq_z3 == eq_dpll == eq_cdcl)
        else:
            same = (eq_z3 == eq_dpll == eq_cdcl)

        if same:
            passed += 1
        else:
            failed += 1
            print(f"!!! FAILED TEST {i} !!!")

        print(f"expected: {'equiv' if exp else 'diff'}")

        if eq_brute is not None:
            print(f"brute  : {'equiv' if eq_brute else 'diff'} ({t_brute:.3f} ms)")
        else:
            print(f"brute  : SKIPPED (>25 vars)")

        print(f"Z3     : {'equiv' if eq_z3 else 'diff'} ({t_z3:.3f} ms)")
        print(f"DPLL   : {'equiv' if eq_dpll else 'diff'} ({t_dpll:.3f} ms)")
        print(f"CDCL   : {'equiv' if eq_cdcl else 'diff'} ({t_cdcl:.3f} ms)")

    total = passed + failed
    rate = (passed * 100.0 / total) if total else 0.0
    print(f"\n{'=' * 50}")
    print(f"Passed: {passed}/{total}, Failed: {failed}")
    print(f"Rate: {rate:.1f}%")
    print(f"Total Brute Time: {total_brute_time:.3f} ms")
    print(f"Total Z3 Time: {total_sat_time:.3f} ms")
    print(f"Total DPLL Time: {total_dpll_time:.3f} ms")
    print(f"Total CDCL Time: {total_cdcl_time:.3f} ms")


if __name__ == "__main__":
    demo()
