from __future__ import annotations

from typing import Tuple, Optional

from main import parse, check_equiv_bruteforce, check_equiv_z3


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


def run_once() -> Optional[Tuple[bool, bool]]:
    f1 = read_formula(1)
    if not f1:
        return None

    f2 = read_formula(2)
    if not f2:
        return None

    print("\nПроверка эквивалентности:")
    print(" F1:", f1)
    print(" F2:", f2)

    eq_brute, t_brute = check_equiv_bruteforce(f1, f2)
    eq_z3, t_z3 = check_equiv_z3(f1, f2)

    print(" brute: ", "equiv" if eq_brute else "diff",
          f"({t_brute:.3f} ms)")
    print(" Z3:   ", "equiv" if eq_z3 else "diff",
          f"({t_z3:.3f} ms)")

    return eq_brute, eq_z3


def main():
    print("Допустимые операторы: !, &, |, ->, <-> и скобки ().")
    print("Переменные: x0, x1, x2, ...")
    print("Пустая строка вместо формулы завершает программу.\n")

    while True:
        res = run_once()
        if res is None:
            break
        print()


if __name__ == "__main__":
    main()
