from __future__ import annotations
from typing import Tuple, Optional
from main import parse, check_equiv_bruteforce, check_equiv_z3, check_equiv_dpll, check_equiv_cdcl

def read_formula(index: int) -> str:
    while True:
        s = input(f"[{index}] >> ").strip()
        if not s: return ""
        try:
            parse(s)
            return s
        except SyntaxError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")

def run_once() -> Optional[bool]:
    f1 = read_formula(1)
    if not f1: return None
    f2 = read_formula(2)
    if not f2: return None

    print("-" * 20)
    print(f"F1: {f1}")
    print(f"F2: {f2}")

    eq_brute, t_brute = check_equiv_bruteforce(f1, f2)
    eq_z3, t_z3 = check_equiv_z3(f1, f2)
    eq_dpll, t_dpll = check_equiv_dpll(f1, f2)
    eq_cdcl, t_cdcl = check_equiv_cdcl(f1, f2)

    print(f"Brute : {'equiv' if eq_brute else 'diff'} ({t_brute:.3f} ms)")
    print(f"Z3    : {'equiv' if eq_z3 else 'diff'} ({t_z3:.3f} ms)")
    print(f"DPLL  : {'equiv' if eq_dpll else 'diff'} ({t_dpll:.3f} ms)")
    print(f"CDCL  : {'equiv' if eq_cdcl else 'diff'} ({t_cdcl:.3f} ms)")

    return True

def main():
    print("Allowed ops: !, &, |, ->, <->.")
    print("Examples: x0, x1, x2, ...")
    print("Enter empty line to exit.")
    while True:
        res = run_once()
        if res is None:
            break
        print("=" * 20)

if __name__ == "__main__":
    main()
