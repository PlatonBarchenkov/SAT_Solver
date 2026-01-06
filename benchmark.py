from __future__ import annotations
import csv
import random
from main import check_equiv_bruteforce, check_equiv_z3, check_equiv_dpll, check_equiv_cdcl


def generate_shuffled_chain(n: int) -> str:
    terms = [f"x{i}" for i in range(n)]
    random.shuffle(terms)
    return " & ".join(terms)


def run_benchmark():
    random.seed(42)
    NUM_VARS_LIST = list(range(2, 101, 2))
    TESTS_PER_POINT = 5

    print(f"Starting benchmark (N=2..100)...")
    print("-" * 60)
    print(f"{'N':<5} | {'Brute (ms)':<12} | {'DPLL (ms)':<12} | {'CDCL (ms)':<12} | {'Z3 (ms)':<12}")
    print("-" * 60)

    rows = []

    for n in NUM_VARS_LIST:
        brute_times = []
        dpll_times = []
        cdcl_times = []
        z3_times = []

        do_brute = (n <= 23)

        for _ in range(TESTS_PER_POINT):
            f1 = generate_shuffled_chain(n)
            f2 = generate_shuffled_chain(n)

            if do_brute:
                eq, t_bf = check_equiv_bruteforce(f1, f2)
                if eq is None:
                    brute_times.append(None)
                else:
                    brute_times.append(t_bf)
            else:
                brute_times.append(None)

            _, t_dpll = check_equiv_dpll(f1, f2)
            dpll_times.append(t_dpll)

            _, t_cdcl = check_equiv_cdcl(f1, f2)
            cdcl_times.append(t_cdcl)

            _, t_z3 = check_equiv_z3(f1, f2)
            z3_times.append(t_z3)

        valid_brute = [t for t in brute_times if t is not None]
        avg_brute = sum(valid_brute) / len(valid_brute) if valid_brute else None

        avg_dpll = sum(dpll_times) / len(dpll_times)
        avg_cdcl = sum(cdcl_times) / len(cdcl_times)
        avg_z3 = sum(z3_times) / len(z3_times)

        s_brute = f"{avg_brute:.3f}" if avg_brute is not None else "-"
        print(f"{n:<5} | {s_brute:<12} | {avg_dpll:<12.3f} | {avg_cdcl:<12.3f} | {avg_z3:<12.3f}")

        rows.append({
            "n": n,
            "brute": avg_brute if avg_brute is not None else "",
            "dpll": avg_dpll,
            "cdcl": avg_cdcl,
            "z3": avg_z3
        })

    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "brute", "dpll", "cdcl", "z3"])
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 60)
    print("Benchmark finished.")


if __name__ == "__main__":
    run_benchmark()
