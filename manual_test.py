from __future__ import annotations

from typing import Tuple

from main import parse, check_equiv_bruteforce, check_equiv_z3, check_equiv_z3_preprocessed


def read_formula(index: int) -> str:
    while True:
        s = input(f"Введите формулу {index} (или пустую строку для выхода): ").strip()
        if not s:
            return ""
        try:
            parse(s)
            return s
        except SyntaxError as e:
            print(f"Синтаксическая ошибка: {e}")
        except Exception as e:
            print(f"Ошибка разбора: {e}")


def run_once() -> Tuple[bool, bool, bool]:
    f1 = read_formula(1)
    if not f1:
        return False, False, False
    f2 = read_formula(2)
    if not f2:
        return False, False, False
    print("\nПроверка эквивалентности:")
    print(" F1:", f1)
    print(" F2:", f2)
    eq_brute, t_brute = check_equiv_bruteforce(f1, f2)
    eq_z3, t_z3 = check_equiv_z3(f1, f2)
    eq_z3p, t_z3p = check_equiv_z3_preprocessed(f1, f2)
    print(" brute:      ", "equiv" if eq_brute else "diff", f"({t_brute:.3f} ms)")
    print(" Z3 (митор):", "equiv" if eq_z3 else "diff", f"({t_z3:.3f} ms)")
    print(" Z3+pre:    ", "equiv" if eq_z3p else "diff", f"({t_z3p:.3f} ms)")
    return eq_brute, eq_z3, eq_z3p


def main():
    print("Допустимые операторы: !, &, |, ->, <-> и скобки ().")
    print("Переменные: x0, x1, x2, ...")
    print("Пустая строка вместо формулы завершает программу.\n")
    while True:
        eq_brute, eq_z3, eq_z3p = run_once()
        if eq_brute is False and eq_z3 is False and eq_z3p is False:
            break
        print()


if __name__ == "__main__":
    main()
