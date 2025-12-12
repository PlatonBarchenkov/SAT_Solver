from __future__ import annotations

import csv
import random

from main import random_formula, check_equiv_bruteforce, check_equiv_z3


def run_benchmark():
    random.seed(0)

    # будем менять число переменных от 2 до 25
    NUM_VARS_LIST = list(range(2, 26))
    # сколько случайных пар формул на каждое n
    TESTS_PER_POINT = 30
    # глубина случайных формул
    FORMULA_DEPTH = 6

    rows = []

    for n in NUM_VARS_LIST:
        brute_times = []
        z3_times = []

        print(f"\n=== n = {n} переменных ===")

        for i in range(TESTS_PER_POINT):
            f1 = random_formula(num_vars=n,
                                depth=FORMULA_DEPTH,
                                min_used_vars=max(1, n // 2))

            if random.random() < 0.5:
                f2 = f1
            else:
                f2 = random_formula(num_vars=n,
                                    depth=FORMULA_DEPTH,
                                    min_used_vars=max(1, n // 2))

            eq_bf, t_bf = check_equiv_bruteforce(f1, f2)
            eq_z3, t_z3 = check_equiv_z3(f1, f2)

            brute_times.append(t_bf)
            z3_times.append(t_z3)

            print(f"  test {i+1:2d}: "
                  f"brute {t_bf:8.3f} ms, "
                  f"Z3 {t_z3:8.3f} ms, "
                  f"{'OK' if eq_bf == eq_z3 else 'MISMATCH'}")

        avg_brute = sum(brute_times) / len(brute_times)
        avg_z3 = sum(z3_times) / len(z3_times)

        print(f"  -> среднее время: brute = {avg_brute:.3f} ms, Z3 = {avg_z3:.3f} ms")

        rows.append((n, avg_brute, avg_z3))

    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num_vars", "avg_bruteforce_ms", "avg_z3_ms"])
        writer.writerows(rows)

    print("\nРезультаты сохранены в benchmark_results.csv")


if __name__ == "__main__":
    run_benchmark()
