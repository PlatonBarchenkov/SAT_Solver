from __future__ import annotations

import csv
import random

from main import random_formula, check_equiv_bruteforce, check_equiv_z3, check_equiv_z3_preprocessed

https://github.com/PlatonBarchenkov/SAT-Solver.git
def run_benchmark():
    random.seed(0)
    num_vars_list = list(range(2, 35))
    tests_per_point = 30
    formula_depth = 6
    rows = []
    for n in num_vars_list:
        brute_times = []
        z3_times = []
        z3p_times = []
        print(f"\n=== n = {n} переменных ===")
        for i in range(tests_per_point):
            f1 = random_formula(num_vars=n, depth=formula_depth, min_used_vars=max(1, n // 2))
            if random.random() < 0.5:
                f2 = f1
            else:
                f2 = random_formula(num_vars=n, depth=formula_depth, min_used_vars=max(1, n // 2))
            eq_bf, t_bf = check_equiv_bruteforce(f1, f2)
            eq_z3, t_z3 = check_equiv_z3(f1, f2)
            eq_z3p, t_z3p = check_equiv_z3_preprocessed(f1, f2)
            brute_times.append(t_bf)
            z3_times.append(t_z3)
            z3p_times.append(t_z3p)
            status = "OK" if (eq_bf == eq_z3 == eq_z3p) else "MISMATCH"
            print(
                f"  test {i + 1:2d}: "
                f"brute {t_bf:8.3f} ms, "
                f"Z3 {t_z3:8.3f} ms, "
                f"Z3+pre {t_z3p:8.3f} ms, {status}"
            )
        avg_brute = sum(brute_times) / len(brute_times)
        avg_z3 = sum(z3_times) / len(z3_times)
        avg_z3p = sum(z3p_times) / len(z3p_times)
        print(f"  -> среднее время: brute = {avg_brute:.3f} ms, Z3 = {avg_z3:.3f} ms, Z3+pre = {avg_z3p:.3f} ms")
        rows.append((n, avg_brute, avg_z3, avg_z3p))
    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num_vars", "avg_bruteforce_ms", "avg_z3_ms", "avg_z3_preproc_ms"])
        writer.writerows(rows)
    print("\nРезультаты сохранены в benchmark_results.csv")


if __name__ == "__main__":
    run_benchmark()
